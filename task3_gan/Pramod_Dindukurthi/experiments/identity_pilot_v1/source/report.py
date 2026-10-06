"""Produce plots and a report draft from actual artifacts, preserving raw logs."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def build(run_dir, export_dir):
    run_dir, export_dir = Path(run_dir), Path(export_dir)
    history = {}
    for path in sorted((run_dir/"reproducibility"/"raw_logs").glob("history_*.csv")):
        with open(path) as f:
            for row in csv.DictReader(f):
                history[int(row["step"])] = {k: float(v) for k, v in row.items()}
    meta = json.loads((export_dir/"export_manifest.json").read_text())
    # Resumed repeated steps use the last log segment; never edit original logs.
    rows = [history[k] for k in sorted(history) if k <= meta["step"]]
    plots = run_dir/"outputs"/"plots"
    plots.mkdir(parents=True, exist_ok=True)
    panels = {"adversarial": ["adv_A2B", "adv_B2A", "D_A", "D_B"],
              "cycle_identity": ["cycle_A", "cycle_B", "identity_A", "identity_B"],
              "gradients": ["grad_G_A2B", "grad_G_B2A", "grad_D_A", "grad_D_B"],
              "learning_rate": ["lr"]}
    for name, fields in panels.items():
        fig, ax = plt.subplots(figsize=(9, 4))
        for field in fields:
            ax.plot([r["step"] for r in rows], [r[field] for r in rows], label=field, alpha=0.8)
        ax.set(xlabel="Optimizer step", ylabel=name.replace("_", " "))
        ax.legend(); fig.tight_layout(); fig.savefig(plots/(name+".png"), dpi=150); plt.close(fig)
    with open(export_dir/"full_metrics_report.csv") as f:
        metrics = list(csv.DictReader(f))
    if rows:
        window = rows[-min(100, len(rows)):]
        for r in metrics:
            for key in ["G_total", "D_total", "adv_A2B", "adv_B2A", "D_A", "D_B",
                        "cycle_A", "cycle_B", "identity_A", "identity_B",
                        "grad_G_A2B", "grad_G_B2A", "grad_D_A", "grad_D_B"]:
                r[key+"_last100_mean"] = float(np.mean([x[key] for x in window]))
            r["training_seconds_to_checkpoint"] = rows[-1]["training_seconds"]
            r["images_per_second_to_checkpoint"] = rows[-1]["source_images_per_second"]
            r["nan_count"] = rows[-1]["nan_count"]
            r["inf_count"] = rows[-1]["inf_count"]
            r["peak_gpu_memory_bytes_through_checkpoint"] = max(x["peak_memory_bytes"] for x in rows)
    audit = run_dir/"outputs"/"human_audit"/"audit_summary.json"
    if audit.exists():
        audit_data = json.loads(audit.read_text())
        if audit_data["checkpoint_sha256"] != meta["checkpoint_sha256"]:
            raise ValueError("Human audit belongs to another checkpoint")
        for r in metrics:
            for a in audit_data["results"]:
                if a["direction"] == r["direction"]:
                    r["human_"+a["criterion"]+"_mean"] = a["pooled_mean"]
                    r["human_"+a["criterion"]+"_agreement_percent"] = a["exact_agreement_percent"]
            r["human_agreement"] = "exact agreement per criterion; see dedicated columns"
    for name in ["full_metrics_report.csv", "metrics_report.csv"]:
        with open(run_dir/name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(metrics[0])); w.writeheader(); w.writerows(metrics)
    lines = ["# Task 3 results draft", "", "A = Monet; B = Photo.", "",
             f"Checkpoint SHA-256: `{meta['checkpoint_sha256']}`; step {meta['step']}; {meta['weights']} weights.",
             f"Evaluation split: {meta['split']}. Local metrics are not verified Kaggle scores.", "",
             "| Direction | Real/generated n | FID | KID | Content cosine |", "|---|---|---|---|---|"]
    for r in metrics:
        lines.append(f"| {r['direction']} | {r['n_real']}/{r['n_generated']} | {r['FID']} | {r['KID_mean']} | {r['content_cosine_inception2048']} |")
    lines += ["", "## Interpretation to complete from evidence", "",
              "- Explain architecture, preprocessing, schedule and loss weights using the run manifest.",
              "- Discuss style/content trade-offs, small-reference-set FID uncertainty and KID subset variation.",
              "- Inspect the fixed input/translation/cycle grids; document specific failures and testable fixes.",
              "- Explain generator/discriminator and gradient curves; low cycle loss does not establish good translation.",
              "- Add independently completed human ratings and verified Kaggle scores/rank when available.",
              "- Compare with teammates under the same evaluation protocol.", "",
              "## Evidence", "", "See full_metrics_report.csv, training_summary.json, reproducibility/,",
              "outputs/plots/, evaluation_protocol.json and export_manifest.json in the selected export directory.", ""]
    # Draft has a separate name so rerunning never overwrites the student's analysis.
    (run_dir/"results_draft.md").write_text("\n".join(lines))
    print("Wrote metric reports, plots and results_draft.md to", run_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--export-dir", required=True)
    a = p.parse_args()
    build(a.run_dir, a.export_dir)
