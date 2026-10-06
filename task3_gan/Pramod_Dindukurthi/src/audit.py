"""Fixed 30-image audit, independent rater sheets, and exact percentage agreement."""
import argparse
import csv
import json
from pathlib import Path
import random
import shutil

from PIL import Image
import numpy as np

from common import json_write, sha256
from data import rgb


CRITERIA = ["style", "content", "artifacts"]


def prepare(export_dir, output, seed=266):
    export_dir, output = Path(export_dir), Path(output)
    if output.exists():
        raise FileExistsError("Audit exists; keep fixed samples and do not overwrite ratings")
    meta = json.loads((export_dir/"export_manifest.json").read_text())
    if meta["split"] != "val":
        raise ValueError("Use held-out validation images for the human audit")
    rng, chosen = random.Random(seed), []
    for direction in ["A2B", "B2A"]:
        group = [r for r in meta["records"] if r["direction"] == direction]
        if len(group) < 15:
            raise ValueError(f"Need at least 15 fixed validation samples in {direction} for the planned audit")
        chosen.extend(rng.sample(group, 15))
    rng.shuffle(chosen)
    (output/"rater_packet").mkdir(parents=True)
    key, rows = [], []
    for i, row in enumerate(chosen, 1):
        item = f"sample_{i:02d}"
        source, pred = rgb(export_dir/row["input"]), rgb(export_dir/row["prediction"])
        canvas = Image.new("RGB", (source.width*2, source.height))
        canvas.paste(source, (0, 0)); canvas.paste(pred, (source.width, 0))
        canvas.save(output/"rater_packet"/(item+".png"))
        rows.append({"sample_id": item, "target_domain": "Photo" if row["domain"] == "A" else "Monet",
                     "style": "", "content": "", "artifacts": "", "notes": ""})
        key.append({"sample_id": item, **row})
    for rater in [1, 2]:
        with open(output/"rater_packet"/f"rater_{rater}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (output/"rater_packet"/"INSTRUCTIONS.md").write_text(
        "# Independent blinded audit\n\nLeft: input. Right: translation. Target domain is in the CSV.\n"
        "Rate independently before discussing with the other rater. Do not disclose architecture, "
        "checkpoint, model scores or model identity. Each rater fills their own file.\n\n"
        "Use integers 1-5. Style: 1=no target style, 3=partial, 5=convincing target style. "
        "Content: 1=major scene loss, 3=partly retained, 5=structure and objects retained. "
        "Artifacts: 1=severe defects, 3=noticeable defects, 5=no visible defects. "
        "For ALL criteria higher is better. 2 and 4 are intermediate. Add observations in notes.\n")
    json_write(output/"private_key.json", {"seed": seed, "checkpoint_sha256": meta["checkpoint_sha256"],
                                         "export_manifest_sha256": sha256(export_dir/"export_manifest.json"),
                                         "samples": key})
    print("Give raters only", output/"rater_packet", "Keep private_key.json away from raters.")


def summarize(output):
    output = Path(output)
    key = json.loads((output/"private_key.json").read_text())
    ids = {r["sample_id"] for r in key["samples"]}
    ratings = []
    for n in [1, 2]:
        with open(output/"rater_packet"/f"rater_{n}.csv") as f:
            rows = list(csv.DictReader(f))
        mapping = {r["sample_id"]: r for r in rows}
        if set(mapping) != ids or len(rows) != len(ids):
            raise ValueError("Each rater must rate exactly the 30 unique sample IDs")
        for row in rows:
            for c in CRITERIA:
                if row[c] not in {"1", "2", "3", "4", "5"}:
                    raise ValueError(f"Missing/invalid rating: rater {n}, {row['sample_id']}, {c}")
        ratings.append(mapping)
    result = []
    for direction in ["A2B", "B2A", "both"]:
        group = [r["sample_id"] for r in key["samples"] if direction == "both" or r["direction"] == direction]
        for c in CRITERIA:
            a = np.array([int(ratings[0][i][c]) for i in group])
            b = np.array([int(ratings[1][i][c]) for i in group])
            result.append({"direction": direction, "criterion": c, "n_samples": len(group),
                           "rater1_mean": float(a.mean()), "rater2_mean": float(b.mean()),
                           "pooled_mean": float(np.concatenate([a, b]).mean()),
                           "exact_agreement_percent": float(100*np.mean(a == b))})
    json_write(output/"audit_summary.json", {"checkpoint_sha256": key["checkpoint_sha256"], "results": result})
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "summarize"])
    p.add_argument("--export-dir")
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=266)
    a = p.parse_args()
    if a.action == "prepare":
        if not a.export_dir:
            p.error("prepare requires --export-dir")
        prepare(a.export_dir, a.output, a.seed)
    else:
        summarize(a.output)
