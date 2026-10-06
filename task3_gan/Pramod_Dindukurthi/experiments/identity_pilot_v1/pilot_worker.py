
"""Two controlled CycleGAN warm-start experiments, using the user's trainer."""
import argparse, copy, csv, gc, json, os, signal, sys, traceback
from pathlib import Path
from types import SimpleNamespace


def pilot_config(base, identity_weight):
    config = copy.deepcopy(base)
    config["train"].update(
        lr=5e-5, cycle_weight=10.0, identity_weight=float(identity_weight),
        max_steps=20000, decay_start=10000, max_hours=6.0,
        save_every=1000, keep_every=5000, preview_every=1000,
        log_every=500, save_minutes=5, amp=False,
    )
    return config


def next_target(step):
    return min(20000, (step // 5000 + 1) * 5000)


def main():
    import fcntl
    import numpy as np
    import torch
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    lock = (root / "worker.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("A pilot worker already owns this experiment.", flush=True)
        return
    plan = json.loads((root / "plan.json").read_text())
    sys.path.insert(0, str(root / "source"))
    from common import sha256, json_write, seed_all, utc_now, manifest
    from checkpointing import atomic_save, backup_file, load_checkpoint
    from data import read_splits, UnpairedDataset, rgb, transform, to_image
    from models import Generator
    import train
    import evaluate_official as official

    status_path = root / "status.json"
    def status(phase, **details):
        json_write(status_path, {"phase": phase, "updated_utc": utc_now(),
                                "pid": os.getpid(), **details})

    def stop_requested():
        return train.STOP_REQUESTED or (root / "STOP").exists()

    signal.signal(signal.SIGTERM, train.request_stop)
    signal.signal(signal.SIGINT, train.request_stop)
    try:
        status("preflight")
        if not torch.cuda.is_available():
            raise RuntimeError("This pilot requires the lab CUDA runtime.")
        for rel, digest in plan["source_sha256"].items():
            assert sha256(root / "source" / rel) == digest, "Pilot source changed."
        assert sha256(root / "pilot_worker.py") == plan["worker_sha256"]
        parent = Path(plan["parent_checkpoint"])
        assert sha256(parent) == plan["parent_checkpoint_sha256"], "Parent checkpoint changed."
        splits_path = root / "splits.json"
        split_hash = sha256(splits_path)
        data_root = Path(plan["data_root"])
        splits = read_splits(splits_path, data_root, verify=True)
        official.device = torch.device("cuda")
        evaluator_hash = sha256(Path(official.__file__))
        definitions = [("A", "A2B", "G_A2B", "photo_jpg"),
                       ("B", "B2A", "G_B2A", "monet_jpg")]
        sources, reference = {}, {}
        known = {p: h for d in splits["domains"].values() for p, h in d["sha256"].items()}
        reference_inventory = {}
        for domain, direction, _, target in definitions:
            part = splits["domains"][domain]
            sources[direction] = (part["train"] + part["val"])[:300]
            reference[direction] = official.take_n(official.list_images(str(data_root / target)), 300)
            assert len(sources[direction]) == len(reference[direction]) == 300
            reference_inventory[direction] = []
            for p in reference[direction]:
                rel = Path(p).relative_to(data_root).as_posix()
                assert rel in known and sha256(p) == known[rel], f"Reference changed: {rel}"
                reference_inventory[direction].append({"file": rel, "sha256": known[rel]})

        # The schedule counts extra updates from zero. Dataset indices continue
        # after the parent checkpoint, identically for both experimental arms.
        data_offset = int(plan["parent_step"])
        class ContinuedDataset(UnpairedDataset):
            def __getitem__(self, index):
                return super().__getitem__(index + data_offset)
        train.UnpairedDataset = ContinuedDataset

        def evaluate(checkpoint, arm, extra_step):
            digest = sha256(checkpoint)
            out = arm / "outputs/official" / f"extra_{extra_step:08d}_raw_{digest[:12]}"
            out.mkdir(parents=True, exist_ok=True)
            manifest_path = out / "export_manifest.json"
            score_path = out / "metrics.json"
            csv_path = out / "submission.csv"
            if not manifest_path.exists():
                state = load_checkpoint(checkpoint)
                assert state["split_sha256"] == split_hash
                assert state["step"] == extra_step
                records = []
                preview_rows = []
                from PIL import Image
                for domain, direction, key, _ in definitions:
                    folder = out / f"pred_{direction}"
                    folder.mkdir(exist_ok=True)
                    model = Generator(**state["config"]["model"]).cuda().eval()
                    model.load_state_dict(state["models"][key])
                    with torch.inference_mode():
                        for i, rel in enumerate(sources[direction]):
                            x = transform(rgb(data_root / rel), 256).unsqueeze(0).cuda()
                            y = model(x)
                            image_path = folder / f"{i:06d}.jpg"
                            to_image(y[0]).save(image_path, quality=95, subsampling=0)
                            records.append({"domain": domain, "direction": direction,
                                "source": rel, "prediction": image_path.relative_to(out).as_posix(),
                                "prediction_sha256": sha256(image_path)})
                            if (i + 1) % 100 == 0:
                                print(f"Generated {direction}: {i+1}/300", flush=True)
                        # Raw previews use the same first four validation inputs
                        # each time; left=input, right=translation.
                        for rel in splits["domains"][domain]["val"][:4]:
                            x = transform(rgb(data_root / rel), 256).unsqueeze(0).cuda()
                            y = model(x)
                            row = Image.new("RGB", (512, 256))
                            row.paste(to_image(x[0]), (0, 0))
                            row.paste(to_image(y[0]), (256, 0))
                            preview_rows.append(row)
                    del model, x, y
                    gc.collect()
                    torch.cuda.empty_cache()
                grid = Image.new("RGB", (512, 256 * len(preview_rows)))
                for i, row in enumerate(preview_rows):
                    grid.paste(row, (0, 256 * i))
                grid.save(out / "raw_preview.png")
                del state
                json_write(manifest_path, {"checkpoint_sha256": digest, "weights": "raw",
                    "extra_step": extra_step, "parent_step": data_offset,
                    "split_sha256": split_hash, "image_size": 256, "jpeg_quality": 95,
                    "jpeg_subsampling": 0, "records": records})
            meta = json.loads(manifest_path.read_text())
            assert meta["checkpoint_sha256"] == digest and meta["weights"] == "raw"
            assert meta["split_sha256"] == split_hash
            assert len(meta["records"]) == 600
            for _, direction, _, _ in definitions:
                records = [r for r in meta["records"] if r["direction"] == direction]
                assert [r["source"] for r in records] == sources[direction]
                assert [r["prediction"] for r in records] == [f"pred_{direction}/{i:06d}.jpg" for i in range(300)]
                for r in records:
                    assert sha256(out / r["prediction"]) == r["prediction_sha256"]
                assert len(official.list_images(str(out / f"pred_{direction}"))) == 300
            result = None
            if score_path.exists() and csv_path.exists():
                saved = json.loads(score_path.read_text())
                if (saved.get("checkpoint_sha256") == digest and
                    saved.get("evaluation_script_sha256") == evaluator_hash and
                    saved.get("export_manifest_sha256") == sha256(manifest_path) and
                    saved.get("reference_inventory") == reference_inventory):
                    result = saved
            if result is None:
                gc.collect()
                torch.cuda.empty_cache()
                directions = {}
                for _, direction, _, _ in definitions:
                    paths = official.list_images(str(out / f"pred_{direction}"))
                    fid, mifid = official.calculate_fid_mifid(reference[direction], paths, batch_size=16)
                    assert np.isfinite([fid, mifid]).all(), "Non-finite official metric."
                    directions[direction] = {"FID": float(fid), "MiFID": float(mifid)}
                submission = {"ID": 1, "FID": sum(v["FID"] for v in directions.values()) / 2,
                              "MiFID": sum(v["MiFID"] for v in directions.values()) / 2}
                with csv_path.with_suffix(".csv.tmp").open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["ID", "FID", "MiFID"])
                    writer.writeheader(); writer.writerow(submission)
                os.replace(csv_path.with_suffix(".csv.tmp"), csv_path)
                result = {"per_direction": directions, "submission": submission,
                    "checkpoint_sha256": digest, "evaluation_script_sha256": evaluator_hash,
                    "export_manifest_sha256": sha256(manifest_path),
                    "reference_inventory": reference_inventory, "n_eval": 300,
                    "weights": "raw", "created_utc": utc_now()}
                json_write(score_path, result)
            values = result["submission"]
            row = {"arm": arm.name, "identity_weight": 5.0 if arm.name == "control_id5" else 2.5,
                "extra_steps": extra_step, "parent_plus_extra": data_offset + extra_step,
                "FID_A2B": result["per_direction"]["A2B"]["FID"],
                "FID_B2A": result["per_direction"]["B2A"]["FID"],
                "FID": values["FID"], "MiFID": values["MiFID"],
                "estimated_score": -(values["FID"] + values["MiFID"]) / 2,
                "checkpoint": str(checkpoint), "submission_csv": str(csv_path),
                "preview": str(out / "raw_preview.png")}
            assert np.isfinite([row[k] for k in ("FID_A2B", "FID_B2A", "FID", "MiFID")]).all()
            json_write(arm / f"result_extra_{extra_step:08d}.json", row)
            print("PILOT_RESULT", json.dumps(row), flush=True)
            return row

        def write_comparison():
            rows = [json.loads(p.read_text()) for p in sorted(root.glob("*/result_extra_*.json"))]
            if not rows:
                return
            with (root / "comparison.csv.tmp").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            os.replace(root / "comparison.csv.tmp", root / "comparison.csv")
            best = max(rows, key=lambda r: r["estimated_score"])
            json_write(root / "best_pilot.json", {"best_pilot": best,
                "baseline_estimated_score": -51.08478503110938,
                "improved_on_baseline": best["estimated_score"] > -51.08478503110938})
            backup_root = Path(plan["backup_root"])
            for report in [root / "comparison.csv", root / "best_pilot.json"]:
                backup_file(report, backup_root / report.name)
            for report in root.glob("*/result_extra_*.json"):
                backup_file(report, backup_root / report.relative_to(root))

        for name, identity in [("control_id5", 5.0), ("experiment_id2p5", 2.5)]:
            if stop_requested():
                status("paused", reason="Stop requested; rerun launcher to continue.")
                return
            arm = root / name
            arm.mkdir(exist_ok=True)
            checkpoint = arm / "checkpoints/latest_checkpoint.pt"
            if not checkpoint.exists():
                base = load_checkpoint(parent)
                assert base["step"] == data_offset and base["split_sha256"] == split_hash
                assert base["config"]["image_size"] == 256
                assert base["config"]["train"]["batch_size"] == 1
                config = pilot_config(base["config"], identity)
                seed_all(config["seed"], config["train"]["deterministic"])
                trainer = train.Trainer(config, torch.device("cpu"))
                for key, model in trainer.models.items():
                    model.load_state_dict(base["models"][key])
                for key, model in trainer.ema.items():
                    model.load_state_dict(base["models"][key])
                assert trainer.step == 0
                assert all(not optimizer.state for optimizer in trainer.optimizers.values())
                assert all(abs(optimizer.param_groups[0]["lr"] - 5e-5) < 1e-12
                           for optimizer in trainer.optimizers.values())
                (arm / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
                json_write(arm / "warm_start.json", {
                    "parent_checkpoint_sha256": plan["parent_checkpoint_sha256"],
                    "parent_step": data_offset, "models": "raw generators and discriminators",
                    "optimizers": "fresh Adam", "replay_pools": "empty",
                    "ema": "initialized from raw generators", "dataset_index_offset": data_offset})
                atomic_save(trainer.state(split_hash), checkpoint)
                del base, trainer
                gc.collect()
            recovered = load_checkpoint(checkpoint)
            config = recovered["config"]
            assert config == pilot_config(config, identity), "Pilot config changed."
            assert recovered["split_sha256"] == split_hash
            step = int(recovered["step"])
            del recovered
            while True:
                # Recover evaluations missed by an interrupted worker before
                # proceeding to the next training interval.
                for completed in range(5000, step + 1, 5000):
                    frozen = arm / "checkpoints/evaluation" / f"step_{completed:08d}.pt"
                    archive = arm / "checkpoints/periodic" / f"step_{completed:08d}.pt"
                    if not frozen.exists():
                        source = checkpoint if completed == step else archive
                        if not source.exists():
                            raise FileNotFoundError(f"Missing pilot evaluation checkpoint: {source}")
                        backup_file(source, frozen)
                    if not (arm / f"result_extra_{completed:08d}.json").exists():
                        status("evaluating", arm=name, extra_steps=completed)
                        evaluate(frozen, arm, completed)
                        write_comparison()
                if stop_requested():
                    status("paused", arm=name, extra_steps=step)
                    return
                if step >= 20000:
                    break
                target = next_target(step)
                status("training", arm=name, extra_steps=step, next_evaluation=target)
                backup = Path(plan["backup_root"]) / name
                result = train.run(SimpleNamespace(
                    config=str(arm / "config.yaml"), data_root=str(data_root),
                    splits=str(splits_path), run_dir=str(arm), device="cuda",
                    resume=str(checkpoint), backup_dir=str(backup),
                    code_version="controlled-identity-pilot-v1", stop_after=target-step,
                    session_hours=6.0))
                step = int(result["end_step"])
                if step < target:
                    status("paused", arm=name, extra_steps=step,
                           reason="Stopped before the next evaluation; rerun launcher to resume.")
                    return
                # Source is frozen and the original baseline is untouched.
                gc.collect()
                torch.cuda.empty_cache()
            print(f"ARM_COMPLETE {name}", flush=True)
        write_comparison()
        status("complete", comparison=str(root / "comparison.csv"))
        print("BOTH_PILOTS_COMPLETE", flush=True)
    except BaseException as error:
        status("failed", error_type=type(error).__name__, error=str(error))
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
