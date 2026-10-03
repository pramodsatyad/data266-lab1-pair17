"""Full local Task 3 evaluation. Never labels a diagnostic as official MiFID."""
import argparse
import csv
import json
from pathlib import Path
import sys
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parent/"src"))
import numpy as np
from common import device_for, json_write, sha256, utc_now
from metrics import InceptionFeatures, cosine, density_coverage, fid, kid, lpips_pairs


def evaluate(export_dir, run_dir=None, device="auto", batch_size=16, seed=42, subsets=50, k=5):
    export_dir = Path(export_dir)
    meta = json.loads((export_dir/"export_manifest.json").read_text())
    if meta.get("predictions_only"):
        raise ValueError("Predictions-only exports are for official scoring. Full local metrics require a validation export with inputs and cycle reconstructions.")
    if meta["split"] != "val":
        warnings.warn("This is NOT held-out validation; report its split honestly and do not compare with validation runs.")
    records = meta["records"]
    for row in records:
        for field in ["input", "prediction", "cycle"]:
            if sha256(export_dir/row[field]) != row[field+"_sha256"]:
                raise ValueError("Export was modified after inference: " + row[field])
    device = device_for(device)
    cache = Path(run_dir)/"data_processed"/"feature_cache" if run_dir else export_dir/"feature_cache"
    extractor = InceptionFeatures(device, cache)
    feats = {}
    grouped = {d: [r for r in records if r["domain"] == d] for d in ["A", "B"]}
    for d, group in grouped.items():
        for field in ["input", "prediction"]:
            print("Extracting", d, field, len(group), flush=True)
            feats[d, field] = extractor([export_dir/r[field] for r in group], batch_size)
    results, lpips_model = [], None
    for d, target in [("A", "B"), ("B", "A")]:
        group = grouped[d]
        real, fake = feats[target, "input"], feats[d, "prediction"]
        if min(len(real), len(fake)) < 2048:
            warnings.warn("Small-sample FID is biased/unstable. Report sample counts; use KID and visuals too.")
        paths = {field: [export_dir/r[field] for r in group] for field in ["input", "prediction", "cycle"]}
        lp, lpips_model = lpips_pairs(paths["input"], paths["prediction"], device, batch_size, lpips_model)
        lpc, _ = lpips_pairs(paths["input"], paths["cycle"], device, batch_size, lpips_model)
        row = {"direction": "A2B" if d == "A" else "B2A", "source": "Monet" if d == "A" else "Photo",
               "target": "Photo" if d == "A" else "Monet", "step": meta["step"], "weights": meta["weights"],
               "split": meta["split"], "checkpoint_sha256": meta["checkpoint_sha256"],
               "n_real": len(real), "n_generated": len(fake), "FID": fid(real, fake),
               **kid(real, fake, seed, subsets), **density_coverage(real, fake, k),
               "cycle_L1_tensor_0_1": float(np.mean([r["cycle_L1_tensor_0_1"] for r in group])),
               "LPIPS_input_translation": float(lp.mean()), "LPIPS_input_cycle": float(lpc.mean()),
               "content_cosine_inception2048": float(cosine(feats[d, "input"], fake).mean()),
               "human_style_mean": "pending", "human_content_mean": "pending", "human_artifacts_mean": "pending",
               "human_agreement": "pending", "kaggle_public_score": "pending",
               "kaggle_private_score": "pending", "kaggle_rank": "pending"}
        if run_dir and (Path(run_dir)/"training_summary.json").exists():
            summary = json.loads((Path(run_dir)/"training_summary.json").read_text())
            # Training totals may describe a later step; identify the summary step explicitly.
            row.update(training_summary_step=summary["end_step"], training_seconds=summary["training_seconds"],
                       images_per_second=summary["source_images_per_second"],
                       peak_gpu_memory_bytes=summary["peak_gpu_memory_bytes"],
                       parameter_count=sum(summary["parameters"].values()),
                       nan_count=summary["nan_count"], inf_count=summary["inf_count"])
        results.append(row)
        print(json.dumps(row), flush=True)
    output = export_dir/"full_metrics_report.csv"
    with open(output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0])); w.writeheader(); w.writerows(results)
    protocol = {"created_utc": utc_now(), "inception": "torch-fidelity inception-v3-compat 2048",
                "torch_fidelity": extractor.version, "feature_weights_sha256": extractor.weights_hash,
                "LPIPS": "alex v0.1; input/translation and input/cycle separately; [-1,1]",
                "FID": "symmetric covariance PSD square root; uncorrected finite-sample estimate",
                "KID": "unscaled unbiased MMD^2; subset std is not a confidence interval",
                "density_coverage_k": k, "seed": seed, "export_manifest_sha256": sha256(export_dir/"export_manifest.json"),
                "official_kaggle_equivalent": False,
                "selection_diagnostic_mean_FID": float(np.mean([r["FID"] for r in results]))}
    json_write(export_dir/"evaluation_protocol.json", protocol)
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export-dir", required=True)
    p.add_argument("--run-dir")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--subsets", type=int, default=50)
    p.add_argument("--k", type=int, default=5)
    a = p.parse_args()
    evaluate(a.export_dir, a.run_dir, a.device, a.batch_size, a.seed, a.subsets, a.k)
