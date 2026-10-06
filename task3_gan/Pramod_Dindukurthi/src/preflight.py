"""Run on the TRAINING machine; does not reserve a GPU or launch training."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from common import manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root")
    a = p.parse_args()
    print(json.dumps(manifest({}, "preflight"), indent=2))
    print("Python executable:", sys.executable)
    print("Working directory:", Path.cwd())
    print("Disk free GiB:", round(shutil.disk_usage(Path.cwd()).free/2**30, 1))
    if shutil.which("nvidia-smi"):
        subprocess.run(["nvidia-smi"], check=False)
    if a.data_root:
        for folder in ["monet_jpg", "photo_jpg"]:
            root = Path(a.data_root)/folder
            print(folder, "exists:", root.exists(), "images:",
                  sum(p.suffix.lower() in {".jpg", ".jpeg", ".png"} for p in root.rglob("*")))
    print("Confirm a persistent backup destination before long training. A saved notebook does not back up checkpoints.")


if __name__ == "__main__":
    main()
