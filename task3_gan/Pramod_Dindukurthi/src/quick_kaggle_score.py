from pathlib import Path
import csv, json, sys
import torch

# Uses a FROZEN checkpoint. For tomorrow's model, first run only Section 9's
# snapshot cell, then run this cell; no full image-export sections are needed.
default_run = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src/train.py").is_file()),
                   Path("/app/task3_gan/Pramod_Dindukurthi"))
RUN_DIR = Path(globals().get("RUN_DIR", default_run))
DATA_ROOT = Path(globals().get("DATA_ROOT", RUN_DIR.parent / "data"))
frozen = sorted(p for p in (RUN_DIR / "checkpoints/evaluation").glob("step_*.pt")
                if p.stem.removeprefix("step_").isdigit())
default_checkpoint = frozen[-1] if frozen else RUN_DIR / "checkpoints/evaluation/step_00112805.pt"
CHECKPOINT = Path(globals().get("SNAPSHOT", default_checkpoint))
WEIGHTS = globals().get("WEIGHTS", "ema")  # set "raw" for a separately recorded comparison
assert WEIGHTS in {"ema", "raw"}
N_EVAL = 300
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SRC = RUN_DIR / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from checkpointing import load_checkpoint
from common import sha256, json_write, utc_now
from data import read_splits, rgb, transform, to_image
from models import Generator
import evaluate_official as official

if not CHECKPOINT.is_file():
    raise FileNotFoundError(f"Freeze the checkpoint using Section 9's first cell: {CHECKPOINT}")
if CHECKPOINT.name == "latest_checkpoint.pt":
    raise ValueError("Use a frozen evaluation checkpoint, not the actively replaced latest_checkpoint.pt.")
SPLITS = RUN_DIR / "data_processed/splits.json"
print("Scoring frozen checkpoint:", CHECKPOINT, "| weights:", WEIGHTS, flush=True)
checkpoint_hash = sha256(CHECKPOINT)
state = load_checkpoint(CHECKPOINT)
step = state["step"]
assert state["config"]["image_size"] == 256, "Official predictions must be 256 x 256."
assert sha256(SPLITS) == state["split_sha256"], "Training split manifest differs."
tag = f"step_{step:08d}_{WEIGHTS}"
OFFICIAL_DIR = RUN_DIR / "outputs/official" / f"{tag}_quick_{checkpoint_hash[:12]}"
OFFICIAL_CSV = RUN_DIR / f"submission_{tag}_quick_{checkpoint_hash[:12]}.csv"
SCORE_JSON = OFFICIAL_CSV.with_suffix(".metrics.json")

# Reuse a prior score ONLY if its export manifest matches this checkpoint/weights,
# the evaluator source is identical, and both directions used 300 images.
prior = []
for candidate in RUN_DIR.glob(f"submission_{tag}*.metrics.json"):
    payload = json.loads(candidate.read_text())
    if payload.get("evaluation_script_sha256") != sha256(Path(official.__file__)):
        continue
    if any(payload.get("counts", {}).get(d, {}).get("matched") != N_EVAL for d in ("A2B", "B2A")):
        continue
    for manifest in (RUN_DIR / "outputs/official").glob("*/export_manifest.json"):
        if sha256(manifest) != payload.get("export_manifest_sha256"):
            continue
        metadata = json.loads(manifest.read_text())
        if metadata.get("checkpoint_sha256") == checkpoint_hash and metadata.get("weights") == WEIGHTS:
            csv_path = candidate.with_name(candidate.name.removesuffix(".metrics.json") + ".csv")
            if csv_path.is_file():
                prior.append((csv_path, payload))
            break

if prior:
    OFFICIAL_CSV, result = prior[0]
    row = result["submission"]
    del state
    print("Reusing previously computed, matching official metrics.")
else:
    manifest_path = OFFICIAL_DIR / "export_manifest.json"
    if not manifest_path.exists():
        if OFFICIAL_DIR.exists():
            raise RuntimeError(f"Incomplete earlier export exists; preserve it and choose a new output name: {OFFICIAL_DIR}")
        splits = read_splits(SPLITS)
        records = []
        for domain, direction, generator_key in (("A", "A2B", "G_A2B"), ("B", "B2A", "G_B2A")):
            sources = (splits["domains"][domain]["train"] + splits["domains"][domain]["val"])[:N_EVAL]
            assert len(sources) == N_EVAL, "Need 300 permitted source images per direction."
            folder = OFFICIAL_DIR / f"pred_{direction}"
            folder.mkdir(parents=True)
            generator = Generator(**state["config"]["model"]).to(DEVICE).eval()
            generator.load_state_dict(state["ema" if WEIGHTS == "ema" else "models"][generator_key])
            with torch.inference_mode():
                for i, rel in enumerate(sources):
                    assert sha256(DATA_ROOT / rel) == splits["domains"][domain]["sha256"][rel], f"Source changed: {rel}"
                    prediction = generator(transform(rgb(DATA_ROOT / rel), 256).unsqueeze(0).to(DEVICE))
                    path = folder / f"{i:06d}.jpg"
                    to_image(prediction[0]).save(path, quality=95, subsampling=0)
                    records.append({"domain": domain, "direction": direction, "source": rel,
                                    "prediction": path.relative_to(OFFICIAL_DIR).as_posix(),
                                    "prediction_sha256": sha256(path)})
                    if (i + 1) % 100 == 0:
                        print(f"Generated {direction}: {i + 1}/{N_EVAL}", flush=True)
            del generator, prediction
        json_write(manifest_path, {"created_utc": utc_now(), "checkpoint_sha256": checkpoint_hash,
            "step": step, "weights": WEIGHTS, "split": "all", "split_sha256": sha256(SPLITS),
            "image_size": 256, "jpeg_quality": 95, "jpeg_subsampling": 0,
            "predictions_only": True, "max_per_domain": N_EVAL,
            "input_order": "fixed split manifest order; train followed by val; first 300",
            "records": records})
    meta = json.loads(manifest_path.read_text())
    assert meta["checkpoint_sha256"] == checkpoint_hash and meta["weights"] == WEIGHTS
    for record in meta["records"]:
        assert sha256(OFFICIAL_DIR / record["prediction"]) == record["prediction_sha256"], "Prediction changed."
    del state
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    if OFFICIAL_CSV.exists() or SCORE_JSON.exists():
        raise FileExistsError("Preserve the earlier score files; use a different output name.")
    official.device = torch.device(DEVICE)
    per_direction, counts = {}, {}
    for direction, target in (("A2B", "photo_jpg"), ("B2A", "monet_jpg")):
        real = official.take_n(official.list_images(str(DATA_ROOT / target)), N_EVAL)
        generated = official.take_n(official.list_images(str(OFFICIAL_DIR / f"pred_{direction}")), N_EVAL)
        assert len(real) == len(generated) == N_EVAL
        fid, mifid = official.calculate_fid_mifid(real, generated, batch_size=BATCH_SIZE)
        assert torch.isfinite(torch.tensor([fid, mifid], dtype=torch.float64)).all(), "Non-finite metric."
        per_direction[direction] = {"FID": fid, "MiFID": mifid}
        counts[direction] = {"real_capped": len(real), "generated_capped": len(generated), "matched": N_EVAL}
    row = {"ID": 1,
           "FID": sum(x["FID"] for x in per_direction.values()) / 2,
           "MiFID": sum(x["MiFID"] for x in per_direction.values()) / 2}
    with OFFICIAL_CSV.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "FID", "MiFID"])
        writer.writeheader()
        writer.writerow(row)
    result = {"per_direction": per_direction, "submission": row, "counts": counts, "n_eval": N_EVAL,
              "evaluation_script_sha256": sha256(Path(official.__file__)),
              "export_manifest_sha256": sha256(manifest_path),
              "protocol": "instructor torchvision Inception IMAGENET1K_V1; sorted first-300; index cosine"}
    json_write(SCORE_JSON, result)

estimated_score = -(row["FID"] + row["MiFID"]) / 2
print(json.dumps(result["per_direction"], indent=2))
print(f"\nCheckpoint step: {step:,} | Weights: {WEIGHTS}")
print(f"FID: {row['FID']:.6f} | MiFID: {row['MiFID']:.6f}")
print(f"Expected Kaggle displayed score: {estimated_score:.6f} (closer to zero is better)")
print(f"Submission CSV: {OFFICIAL_CSV}")
print("No Kaggle upload was performed. This cell evaluates the chosen frozen checkpoint only.")
