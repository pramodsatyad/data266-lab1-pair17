"""Copy immutable run evidence into the team's required top-level folders."""
import argparse
from pathlib import Path
import re
import shutil

from common import sha256


def collect(run_dir, repo_root, run_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("run-id must contain only letters, digits, underscores or hyphens")
    run_dir, repo_root = Path(run_dir), Path(repo_root)
    for kind in ["raw_logs", "manifests"]:
        for source in (run_dir/"reproducibility"/kind).glob("*"):
            target = repo_root/"reproducibility"/kind/("task3_"+run_id)/source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and sha256(target) != sha256(source):
                raise ValueError(f"Refusing to overwrite different evidence: {target}")
            if not target.exists():
                shutil.copy2(source, target)
    splits = run_dir/"data_processed"/"splits.json"
    if splits.exists():
        target = repo_root/"reproducibility"/"manifests"/("task3_"+run_id)/"splits.json"
        if target.exists() and sha256(target) != sha256(splits):
            raise ValueError("Existing split evidence differs")
        shutil.copy2(splits, target)
    print("Collected run evidence; checkpoint files must be backed up separately.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["run-dir", "repo-root", "run-id"]:
        p.add_argument("--"+key, required=True)
    a = p.parse_args()
    collect(a.run_dir, a.repo_root, a.run_id)
