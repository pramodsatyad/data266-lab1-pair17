"""Export every image in a fixed manifest; never hand-pick generated outputs."""
import argparse
import json
from pathlib import Path
import time

import torch

from checkpointing import load_checkpoint
from common import device_for, json_write, sha256, utc_now
from data import read_splits, rgb, to_image, transform
from models import Generator


@torch.inference_mode()
def export(checkpoint, data_root, splits_path, output, weights="ema", split="val", device="auto"):
    device = device_for(device)
    out, root = Path(output), Path(data_root)
    if (out/"export_manifest.json").exists() or (out/"pred_A2B").exists():
        raise FileExistsError("Export exists. Use a new output directory to preserve prior evidence.")
    state = load_checkpoint(checkpoint)
    if sha256(splits_path) != state["split_sha256"]:
        raise ValueError("Split manifest differs from the trained model's manifest")
    splits = read_splits(splits_path, root, verify=True)
    size = state["config"]["image_size"]
    models = {}
    for k in ["G_A2B", "G_B2A"]:
        models[k] = Generator(**state["config"]["model"]).to(device).eval()
        models[k].load_state_dict(state["ema" if weights == "ema" else "models"][k])
    for folder in ["pred_A2B", "pred_B2A", "input_A", "input_B", "cycle_A", "cycle_B"]:
        (out/folder).mkdir(parents=True, exist_ok=True)
    records = []
    started = time.monotonic()
    for d, gkey, fkey in [("A", "G_A2B", "G_B2A"), ("B", "G_B2A", "G_A2B")]:
        direction = "A2B" if d == "A" else "B2A"
        paths = splits["domains"][d]["train"] + splits["domains"][d]["val"] if split == "all" else splits["domains"][d][split]
        for i, rel in enumerate(paths):
            x = transform(rgb(root/rel), size).unsqueeze(0).to(device)
            y = models[gkey](x)
            rec = models[fkey](y)
            filename = f"{i:06d}"
            input_path, pred_path, cycle_path = f"input_{d}/{filename}.png", f"pred_{direction}/{filename}.jpg", f"cycle_{d}/{filename}.png"
            to_image(x[0]).save(out/input_path)
            # Fixed encoding for all candidates; no image selection/postprocessing.
            to_image(y[0]).save(out/pred_path, quality=95, subsampling=0)
            to_image(rec[0]).save(out/cycle_path)
            records.append({"domain": d, "direction": direction, "source": rel,
                            "input": input_path, "prediction": pred_path, "cycle": cycle_path,
                            "cycle_L1_tensor_0_1": float((rec-x).abs().mean()/2),
                            "input_sha256": sha256(out/input_path), "prediction_sha256": sha256(out/pred_path),
                            "cycle_sha256": sha256(out/cycle_path)})
        print("Exported", direction, len(paths), "images", flush=True)
    meta = {"created_utc": utc_now(), "checkpoint_sha256": sha256(checkpoint),
            "step": state["step"], "weights": weights, "split": split,
            "split_sha256": sha256(splits_path), "image_size": size,
            "jpeg_quality": 95, "jpeg_subsampling": 0,
            "inference_seconds_including_io": time.monotonic()-started,
            "preprocessing": "EXIF orientation, RGB, bicubic square resize",
            "records": records}
    json_write(out/"export_manifest.json", meta)
    return meta


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["checkpoint", "data-root", "splits", "output"]:
        p.add_argument("--"+key, required=True)
    p.add_argument("--weights", choices=["raw", "ema"], default="ema")
    p.add_argument("--split", choices=["val", "train", "all"], default="val")
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    export(a.checkpoint, a.data_root, a.splits, a.output, a.weights, a.split, a.device)
