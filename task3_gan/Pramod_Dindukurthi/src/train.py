"""Step-budgeted CycleGAN training. CLI and notebook use the same entry point."""
import argparse
import contextlib
import csv
import json
import math
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback
import uuid

import torch
from torch import nn
from torch.utils.data import DataLoader
from PIL import Image

from common import (Tee, device_for, json_write, manifest, read_config, save_environment,
                    seed_all, sha256, utc_now)
from data import UnpairedDataset, read_splits, rgb, to_image, transform
from models import build_models
from checkpointing import (ReplayPool, atomic_save, backup_file, load_checkpoint,
                           make_ema, restore_rng, rng_state, update_ema)

STOP_REQUESTED = False


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("Stop requested; finishing the current optimizer step and saving.", flush=True)


def grad_norm(model):
    grads = [p.grad.detach().float().norm(2) ** 2 for p in model.parameters() if p.grad is not None]
    return torch.stack(grads).sum().sqrt() if grads else torch.tensor(0.)


class Trainer:
    def __init__(self, config, device):
        self.config, self.device = config, device
        self.models = build_models(config, device)
        self.ema = {k: make_ema(self.models[k]) for k in ["G_A2B", "G_B2A"]}
        t = config["train"]
        self.optimizers = {
            "G": torch.optim.Adam(list(self.models["G_A2B"].parameters()) +
                                  list(self.models["G_B2A"].parameters()),
                                  lr=t["lr"], betas=(t["beta1"], t["beta2"])),
            "D": torch.optim.Adam(list(self.models["D_A"].parameters()) +
                                  list(self.models["D_B"].parameters()),
                                  lr=t["lr"], betas=(t["beta1"], t["beta2"]))}
        def schedule(step):
            return 1.0 - max(0, step-t["decay_start"]) / (t["max_steps"]-t["decay_start"])
        self.schedulers = {k: torch.optim.lr_scheduler.LambdaLR(v, schedule)
                           for k, v in self.optimizers.items()}
        self.amp = bool(t["amp"] and device.type == "cuda")
        if hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        else:
            # PyTorch 2.1/2.2 expose the CUDA scaler through this namespace.
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        self.pools = {d: ReplayPool(t["pool_size"]) for d in ["A", "B"]}
        self.step, self.training_seconds, self.examples = 0, 0.0, 0
        self.nan_count, self.inf_count = 0, 0

    def autocast(self):
        return torch.autocast("cuda", dtype=torch.float16) if self.amp else contextlib.nullcontext()

    def finite(self, values):
        for value in values:
            self.nan_count += int(torch.isnan(value).sum().item())
            self.inf_count += int(torch.isinf(value).sum().item())
        if self.nan_count or self.inf_count:
            print("NONFINITE_EVENT", json.dumps({"attempted_step": self.step+1,
                  "nan_count": self.nan_count, "inf_count": self.inf_count}), flush=True)
            raise FloatingPointError("Non-finite loss/gradient. Stop; resume the last complete checkpoint with a reviewed configuration.")

    def update(self, batch):
        A, B = (batch[d].to(self.device, non_blocking=True) for d in ["A", "B"])
        G, F = self.models["G_A2B"], self.models["G_B2A"]
        DA, DB = self.models["D_A"], self.models["D_B"]
        t = self.config["train"]
        DA.requires_grad_(False)
        DB.requires_grad_(False)
        self.optimizers["G"].zero_grad(set_to_none=True)
        with self.autocast():
            fake_B, fake_A = G(A), F(B)
            rec_A, rec_B = F(fake_B), G(fake_A)
            losses = {"adv_A2B": ((DB(fake_B)-1)**2).mean(),
                      "adv_B2A": ((DA(fake_A)-1)**2).mean(),
                      "cycle_A": (rec_A-A).abs().mean(), "cycle_B": (rec_B-B).abs().mean()}
            if t["identity_weight"]:
                losses.update(identity_A=(F(A)-A).abs().mean(), identity_B=(G(B)-B).abs().mean())
            else:
                losses.update(identity_A=A.new_tensor(0.), identity_B=B.new_tensor(0.))
            loss_G = (losses["adv_A2B"] + losses["adv_B2A"] +
                      t["cycle_weight"]*(losses["cycle_A"]+losses["cycle_B"]) +
                      t["identity_weight"]*(losses["identity_A"]+losses["identity_B"]))
        self.finite([loss_G])
        self.scaler.scale(loss_G).backward()
        self.scaler.unscale_(self.optimizers["G"])
        norms = {"grad_"+k: grad_norm(self.models[k]) for k in ["G_A2B", "G_B2A"]}
        self.finite(list(norms.values()))
        self.scaler.step(self.optimizers["G"])
        DA.requires_grad_(True)
        DB.requires_grad_(True)
        self.optimizers["D"].zero_grad(set_to_none=True)
        with self.autocast():
            old_A, old_B = self.pools["A"].query(fake_A), self.pools["B"].query(fake_B)
            losses["D_A"] = 0.5*(((DA(A)-1)**2).mean() + (DA(old_A)**2).mean())
            losses["D_B"] = 0.5*(((DB(B)-1)**2).mean() + (DB(old_B)**2).mean())
            loss_D = losses["D_A"] + losses["D_B"]
        self.finite([loss_D])
        self.scaler.scale(loss_D).backward()
        self.scaler.unscale_(self.optimizers["D"])
        norms.update({"grad_"+k: grad_norm(self.models[k]) for k in ["D_A", "D_B"]})
        self.finite(list(norms.values()))
        self.scaler.step(self.optimizers["D"])
        self.scaler.update()
        for s in self.schedulers.values():
            s.step()
        for k in self.ema:
            update_ema(self.ema[k], self.models[k], t["ema_decay"])
        self.step += 1
        self.examples += len(A) + len(B)
        losses.update(G_total=loss_G, D_total=loss_D, **norms)
        return {k: float(v.detach()) for k, v in losses.items()}

    def state(self, split_hash):
        return {"schema": 1, "step": self.step, "config": self.config, "split_sha256": split_hash,
                "models": {k: m.state_dict() for k, m in self.models.items()},
                "ema": {k: m.state_dict() for k, m in self.ema.items()},
                "optimizers": {k: o.state_dict() for k, o in self.optimizers.items()},
                "schedulers": {k: s.state_dict() for k, s in self.schedulers.items()},
                "scaler": self.scaler.state_dict(), "pools": {k: p.images for k, p in self.pools.items()},
                "rng": rng_state(), "examples": self.examples,
                "training_seconds": self.training_seconds,
                "nan_count": self.nan_count, "inf_count": self.inf_count}

    def restore(self, state, split_hash):
        if state["config"] != self.config or state["split_sha256"] != split_hash:
            raise ValueError("Resume requires identical config and split. New experiments need a separate run.")
        for k, m in self.models.items():
            m.load_state_dict(state["models"][k])
        for k, m in self.ema.items():
            m.load_state_dict(state["ema"][k])
        for k, o in self.optimizers.items():
            o.load_state_dict(state["optimizers"][k])
        for k, s in self.schedulers.items():
            s.load_state_dict(state["schedulers"][k])
        self.scaler.load_state_dict(state["scaler"])
        for k, p in self.pools.items():
            p.images = [image.cpu() for image in state["pools"][k]]
        for k in ["step", "examples", "training_seconds", "nan_count", "inf_count"]:
            setattr(self, k, state[k])
        restore_rng(state["rng"])


@torch.inference_mode()
def preview(trainer, root, splits, out):
    size = trainer.config["image_size"]
    rows, scores = [], {}
    for d, gkey, fkey in [("A", "G_A2B", "G_B2A"), ("B", "G_B2A", "G_A2B")]:
        values = []
        for rel in splits["domains"][d]["val"][:4]:
            x = transform(rgb(Path(root)/rel), size).unsqueeze(0).to(trainer.device)
            y = trainer.ema[gkey](x)
            rec = trainer.ema[fkey](y)
            values.append(float((rec-x).abs().mean()))
            row = Image.new("RGB", (size*3, size))
            for i, tensor in enumerate([x[0], y[0], rec[0]]):
                row.paste(to_image(tensor), (i*size, 0))
            rows.append(row)
        scores["preview_cycle_"+d] = sum(values)/len(values)
    grid = Image.new("RGB", (size*3, size*len(rows)))
    for i, row in enumerate(rows):
        grid.paste(row, (0, i*size))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    grid.save(out)
    return scores


def run(args):
    if args.session_hours is not None and (not math.isfinite(args.session_hours) or args.session_hours <= 0):
        raise ValueError("--session-hours must be finite and positive")
    config = read_config(args.config)
    device = device_for(args.device)
    seed_all(config["seed"], config["train"]["deterministic"])
    run_dir = Path(args.run_dir).resolve()
    ckdir = run_dir/"checkpoints"
    if (ckdir/"latest_checkpoint.pt").exists() and not args.resume:
        raise FileExistsError("Run already has a checkpoint: use --resume or a new --run-dir")
    ckdir.mkdir(parents=True, exist_ok=True)
    segment = utc_now().replace(":", "-") + "_" + uuid.uuid4().hex[:6]
    evidence = run_dir/"reproducibility"
    logs = evidence/"raw_logs"
    logs.mkdir(parents=True, exist_ok=True)
    with open(logs/f"train_{segment}.log", "x", buffering=1) as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            try:
                result = train_run(args, config, device, run_dir, segment)
            except BaseException:
                traceback.print_exc()
                print("The last complete atomic checkpoint is retained. Never resume a partially updated step.")
                raise
    if args.backup_dir:
        for path in logs.glob("*"):
            backup_file(path, Path(args.backup_dir)/"reproducibility"/"raw_logs"/path.name)
    return result


def train_run(args, config, device, run_dir, segment):
    started = time.monotonic()
    print("Started", utc_now(), "device", device, "config", json.dumps(config), flush=True)
    splits = read_splits(args.splits, args.data_root, verify=True)
    split_hash = sha256(args.splits)
    split_dest = run_dir/"data_processed"/"splits.json"
    split_dest.parent.mkdir(parents=True, exist_ok=True)
    if Path(args.splits).resolve() != split_dest.resolve():
        shutil.copy2(args.splits, split_dest)
    t = config["train"]
    session_hours = args.session_hours if args.session_hours is not None else t["max_hours"]
    trainer = Trainer(config, device)
    if args.resume:
        trainer.restore(load_checkpoint(args.resume, device), split_hash)
        print("Resumed completed optimizer step", trainer.step)
    info = manifest(config, args.code_version)
    info.update(split_sha256=split_hash, start_step=trainer.step, segment=segment,
                session_hours=session_hours,
                parameters={k: sum(p.numel() for p in m.parameters()) for k, m in trainer.models.items()})
    mpath = run_dir/"reproducibility"/"manifests"/f"run_{segment}.json"
    json_write(mpath, info)
    save_environment(mpath.with_suffix(".requirements.txt"))
    dataset = UnpairedDataset(args.data_root, splits, config["image_size"], config["load_size"], config["seed"])
    stop_step = min(t["max_steps"], trainer.step + args.stop_after) if args.stop_after else t["max_steps"]
    loader = DataLoader(dataset, batch_size=t["batch_size"],
                        sampler=range(trainer.step*t["batch_size"], stop_step*t["batch_size"]),
                        num_workers=t["num_workers"], pin_memory=device.type == "cuda",
                        generator=torch.Generator().manual_seed(config["seed"]), drop_last=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    checkpoint = run_dir/"checkpoints"/"latest_checkpoint.pt"
    last_save = time.monotonic()
    def save():
        nonlocal last_save
        atomic_save(trainer.state(split_hash), checkpoint)
        if trainer.step % t["keep_every"] == 0 and trainer.step:
            backup_file(checkpoint, checkpoint.parent/"periodic"/f"step_{trainer.step:08d}.pt")
        if args.backup_dir:
            backup = Path(args.backup_dir)
            if backup.resolve() == run_dir:
                raise ValueError("Backup directory must differ from run directory")
            backup_file(checkpoint, backup/"checkpoints"/checkpoint.name)
            backup_file(split_dest, backup/"data_processed"/"splits.json")
            backup_file(mpath, backup/"reproducibility"/"manifests"/mpath.name)
            for f in (run_dir/"reproducibility"/"raw_logs").glob("*"):
                backup_file(f, backup/"reproducibility"/"raw_logs"/f.name)
        last_save = time.monotonic()
        print("Checkpoint saved", trainer.step, flush=True)
    history_path = run_dir/"reproducibility"/"raw_logs"/f"history_{segment}.csv"
    with open(history_path, "x", newline="", buffering=1) as f:
        writer = None
        tick = time.monotonic()
        for batch in loader:
            # Timer includes data wait and optimizer work, excludes preview/checkpoint overhead.
            losses = trainer.update(batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.monotonic() - tick
            trainer.training_seconds += elapsed
            row = {"step": trainer.step, "epoch_fraction": trainer.step*t["batch_size"]/len(dataset),
                   **losses, "lr": trainer.optimizers["G"].param_groups[0]["lr"],
                   "step_seconds": elapsed, "training_seconds": trainer.training_seconds,
                   "source_images_per_second": trainer.examples/max(trainer.training_seconds, 1e-9),
                   "nan_count": trainer.nan_count, "inf_count": trainer.inf_count,
                   "peak_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0}
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row)); writer.writeheader()
            writer.writerow(row)
            if trainer.step % t["log_every"] == 0 or trainer.step == 1:
                print(json.dumps(row), flush=True)
            if trainer.step % t["preview_every"] == 0:
                vals = preview(trainer, args.data_root, splits,
                               run_dir/"outputs"/"plots"/f"preview_{trainer.step:08d}.png")
                print("Fixed EMA validation preview (not a selection score):", vals, flush=True)
            if trainer.step % t["save_every"] == 0 or time.monotonic()-last_save >= t["save_minutes"]*60:
                save()
            if STOP_REQUESTED or time.monotonic()-started >= session_hours*3600:
                print("Session time budget reached; saving for continuation.")
                break
            tick = time.monotonic()
    save()
    summary = {**info, "ended_utc": utc_now(), "end_step": trainer.step,
               "session_wall_seconds": time.monotonic()-started,
               "training_seconds": trainer.training_seconds, "source_images_seen": trainer.examples,
               "source_images_per_second": trainer.examples/max(trainer.training_seconds, 1e-9),
               "nan_count": trainer.nan_count, "inf_count": trainer.inf_count,
               "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
               "checkpoint": "checkpoints/latest_checkpoint.pt", "checkpoint_sha256": sha256(checkpoint),
               "schedule_complete": trainer.step >= t["max_steps"]}
    json_write(run_dir/"training_summary.json", summary)
    json_write(run_dir/"reproducibility"/"manifests"/f"completed_{segment}.json", summary)
    if args.backup_dir:
        backup_file(run_dir/"training_summary.json", Path(args.backup_dir)/"training_summary.json")
    print("Completed", json.dumps(summary), flush=True)
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["config", "data-root", "splits", "run-dir"]:
        p.add_argument("--"+name, required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--resume")
    p.add_argument("--backup-dir")
    p.add_argument("--code-version", default="uncommitted-source-hash-recorded")
    p.add_argument("--stop-after", type=int, help="Stop after N additional steps; preserves full LR schedule")
    p.add_argument("--session-hours", type=float,
                   help="Pause and checkpoint after this invocation's wall-clock budget; preserves full LR schedule")
    return p


if __name__ == "__main__":
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    run(parser().parse_args())
