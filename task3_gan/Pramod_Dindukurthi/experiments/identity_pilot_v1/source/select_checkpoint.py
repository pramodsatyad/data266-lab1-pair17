"""Record a validation-selected checkpoint and publish its COMPLETE image exports locally."""
import argparse
import json
from pathlib import Path
import shutil

from common import json_write, sha256, utc_now
from checkpointing import backup_file


def select(checkpoint, export_dir, run_dir, reason):
    checkpoint, export_dir, run_dir = Path(checkpoint), Path(export_dir), Path(run_dir)
    meta = json.loads((export_dir/"export_manifest.json").read_text())
    if meta["split"] != "val":
        raise ValueError("Select checkpoints using the fixed validation protocol")
    if sha256(checkpoint) != meta["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed since inference; use the exact archived snapshot")
    if not (export_dir/"full_metrics_report.csv").exists():
        raise ValueError("Evaluate the candidate before selecting it")
    target = run_dir/"outputs"
    for name in ["pred_A2B", "pred_B2A"]:
        if (target/name).exists():
            raise FileExistsError("Canonical outputs already exist. Archive the prior selection explicitly first.")
    backup_file(checkpoint, run_dir/"checkpoints"/"best_model.pt")
    for name in ["pred_A2B", "pred_B2A"]:
        shutil.copytree(export_dir/name, target/name)
    json_write(run_dir/"selection.json", {"selected_utc": utc_now(), "reason": reason,
              "checkpoint_sha256": meta["checkpoint_sha256"], "step": meta["step"],
              "weights": meta["weights"], "export_directory_name": export_dir.name,
              "note": "All fixed validation outputs copied; competition input requirements may differ."})
    print("Selected checkpoint and complete prediction directories saved. No Kaggle upload performed.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["checkpoint", "export-dir", "run-dir", "reason"]:
        p.add_argument("--"+key, required=True)
    a = p.parse_args()
    select(a.checkpoint, a.export_dir, a.run_dir, a.reason)
