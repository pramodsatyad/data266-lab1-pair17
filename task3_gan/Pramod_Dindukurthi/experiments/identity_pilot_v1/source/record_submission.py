"""Archive an externally produced OFFICIAL evaluator CSV after validating its schema.

This does not calculate, guess, or replace the competition's metric implementation.
"""
import argparse
import csv
from pathlib import Path
import shutil

from common import json_write, sha256, utc_now


def record(official_csv, sample_csv, evaluator, official_log, export_dir, run_dir):
    official_csv, sample_csv, evaluator, official_log = map(Path, [official_csv, sample_csv, evaluator, official_log])
    export_dir, run_dir = Path(export_dir), Path(run_dir)
    for path in [official_csv, sample_csv, evaluator, official_log, export_dir/"export_manifest.json"]:
        if not path.is_file():
            raise FileNotFoundError(path)
    with open(sample_csv) as f:
        sample = list(csv.reader(f))
    with open(official_csv) as f:
        rows = list(csv.reader(f))
    if not rows or not sample or rows[0] != sample[0] or len(rows) != len(sample):
        raise ValueError("Official output headers/row count do not match sample submission")
    if any(len(r) != len(rows[0]) or any(not c.strip() for c in r) for r in rows[1:]):
        raise ValueError("Submission has missing fields")
    dest = run_dir/"submission.csv"
    if dest.exists():
        raise FileExistsError("Archive the previous submission before recording a new one")
    shutil.copy2(official_csv, dest)
    json_write(run_dir/"submission_provenance.json", {
        "recorded_utc": utc_now(), "submission_sha256": sha256(dest),
        "official_evaluator_sha256": sha256(evaluator), "official_log_sha256": sha256(official_log),
        "sample_csv_sha256": sha256(sample_csv), "export_manifest_sha256": sha256(export_dir/"export_manifest.json"),
        "status": "schema checked; user must verify evaluator was run on this export; not uploaded",
        "public_score": None, "private_score": None, "rank": None})
    print("Recorded submission.csv and provenance. No values were recomputed and nothing was uploaded.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["official-csv", "sample-csv", "evaluator", "official-log", "export-dir", "run-dir"]:
        p.add_argument("--"+name, required=True)
    a = p.parse_args()
    record(a.official_csv, a.sample_csv, a.evaluator, a.official_log, a.export_dir, a.run_dir)
