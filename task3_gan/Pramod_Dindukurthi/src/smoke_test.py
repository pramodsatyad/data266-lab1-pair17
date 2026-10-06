"""Synthetic integration tests, not assignment results. No metric weights needed by default."""
import argparse
import copy
import csv
import json
from pathlib import Path
import random
import tempfile
import sys

import numpy as np
from PIL import Image
import torch
import yaml

from checkpointing import load_checkpoint
from common import seed_all, sha256
from data import prepare, read_splits, UnpairedDataset
from infer import export
from metrics import fid, kid, density_coverage, cosine
from models import Generator, PatchDiscriminator
from train import Trainer, parser, run
from audit import prepare as prepare_audit, summarize


def assert_nested(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: assert_nested(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): assert_nested(x, y)
    else:
        assert a == b, (a, b)


def main(full_metrics=False):
    torch.set_num_threads(1)
    root = Path(tempfile.mkdtemp(prefix="data266-task3-smoke-"))
    data_root = root/"data"
    rng = np.random.default_rng(42)
    for folder in ["monet_jpg", "photo_jpg"]:
        (data_root/folder).mkdir(parents=True)
        for i in range(40):
            Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(data_root/folder/f"{i:03}.png")
    # Exact duplicate must not enter another split.
    (data_root/"monet_jpg"/"duplicate.png").write_bytes((data_root/"monet_jpg"/"000.png").read_bytes())
    splits_path = root/"splits.json"
    splits = prepare(data_root, splits_path, val_fraction=0.4)
    assert len(splits["domains"]["A"]["duplicates"]) == 1
    assert not set(splits["domains"]["A"]["train"]) & set(splits["domains"]["A"]["val"])
    member = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((member/"configs"/"baseline.yaml").read_text())
    config.update(image_size=64, load_size=68)
    config["model"].update(channels=4, blocks=1)
    config["train"].update(max_steps=4, decay_start=2, num_workers=0, amp=False,
                           pool_size=2, save_every=2, preview_every=2, keep_every=2,
                           log_every=1, max_hours=1)
    config_path = root/"smoke.yaml"
    config_path.write_text(yaml.safe_dump(config))
    for variant in ["transpose", "resize"]:
        g = Generator(4, 1, variant)
        x = torch.randn(1, 3, 64, 64)
        y = g(x)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()
        d = PatchDiscriminator(4)
        assert d(torch.randn(1, 3, 256, 256)).shape == (1, 1, 30, 30)
    # Cycle loss alone reaches both generators; correct compositions use both mappings.
    seed_all(42)
    trainer = Trainer(config, torch.device("cpu"))
    x = torch.randn(1, 3, 64, 64)
    rec = trainer.models["G_B2A"](trainer.models["G_A2B"](x))
    (rec-x).abs().mean().backward()
    for k in ["G_A2B", "G_B2A"]:
        assert sum(float(p.grad.abs().sum()) for p in trainer.models[k].parameters() if p.grad is not None) > 0
    # Identical stateful trainer trajectory across a real save, reload and CLI-level resume.
    def args(run_dir, extra=None):
        return parser().parse_args(["--config", str(config_path), "--data-root", str(data_root),
                "--splits", str(splits_path), "--run-dir", str(run_dir), "--device", "cpu"]+(extra or []))
    uninterrupted, resumed = root/"uninterrupted", root/"resumed"
    run(args(uninterrupted))
    run(args(resumed, ["--stop-after", "2", "--backup-dir", str(root/"backup")]))
    first_log = sorted((resumed/"reproducibility"/"raw_logs").glob("*.log"))[0]
    first_log_hash = sha256(first_log)
    ck = resumed/"checkpoints"/"latest_checkpoint.pt"
    run(args(resumed, ["--resume", str(ck)]))
    assert sha256(first_log) == first_log_hash, "Resume modified an earlier raw log"
    a, b = load_checkpoint(uninterrupted/"checkpoints"/"latest_checkpoint.pt"), load_checkpoint(ck)
    for key in ["models", "ema", "optimizers", "schedulers", "scaler", "pools", "rng", "step", "examples"]:
        assert_nested(a[key], b[key])
    assert b["step"] == 4
    # A wall-time pause must preserve the same full schedule and resume trajectory.
    timed = root/"timed_pause"
    run(args(timed, ["--session-hours", "1e-12"]))
    timed_ck = timed/"checkpoints"/"latest_checkpoint.pt"
    paused = load_checkpoint(timed_ck)
    assert paused["step"] == 1 and paused["config"] == config
    run(args(timed, ["--resume", str(timed_ck), "--session-hours", "1"]))
    continued = load_checkpoint(timed_ck)
    for key in ["models", "ema", "optimizers", "schedulers", "scaler", "pools", "rng", "step", "examples"]:
        assert_nested(a[key], continued[key])
    bad = copy.deepcopy(config); bad["train"]["lr"] *= 2
    try:
        Trainer(bad, torch.device("cpu")).restore(b, sha256(splits_path))
    except ValueError: pass
    else: raise AssertionError("Changed config was accepted for exact resume")
    # Independent mathematical checks (not pretrained-model scoring).
    features = rng.normal(size=(30, 5))
    assert fid(features, features) < 1e-8
    np.testing.assert_allclose(fid(features, features+2), 20, rtol=1e-7)
    np.testing.assert_allclose(cosine(features, features), 1, atol=1e-7)
    dc = density_coverage(features, features, k=3)
    assert dc["coverage"] == 1 and dc["density"] >= 1
    assert np.isfinite(kid(features, features, subsets=3)["KID_mean"])
    exports = resumed/"outputs"/"eval_smoke"
    meta = export(ck, data_root, splits_path, exports, device="cpu")
    assert len(meta["records"]) == 32
    for row in meta["records"]:
        with Image.open(exports/row["prediction"]) as im:
            assert im.mode == "RGB" and im.size == (64, 64) and im.format == "JPEG"
    audit_dir = resumed/"outputs"/"human_audit"
    prepare_audit(exports, audit_dir)
    # Synthetic ratings exercise validation and agreement; never assignment evidence.
    for n in [1, 2]:
        path = audit_dir/"rater_packet"/f"rater_{n}.csv"
        with open(path) as f: rows = list(csv.DictReader(f))
        for row in rows:
            row.update(style="3", content="4", artifacts="2")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    assert all(r["exact_agreement_percent"] == 100 for r in summarize(audit_dir))
    if full_metrics:
        sys.path.insert(0, str(member))
        from evaluate_local import evaluate
        from report import build
        from select_checkpoint import select
        evaluate(exports, resumed, "cpu", batch_size=4, subsets=3, k=3)
        build(resumed, exports)
        select(ck, exports, resumed, "Synthetic integration test only")
    result = {"status": "PASS", "synthetic_only": True, "temporary_artifacts": str(root),
              "exact_cpu_resume": True, "full_pretrained_metrics": full_metrics}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--full-metrics", action="store_true", help="Downloads Inception/LPIPS weights on first use")
    main(p.parse_args().full_metrics)
