"""Package runnable source without Git, secrets, data, checkpoints or results."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def package(output):
    member = Path(__file__).resolve().parents[1]
    repo = member.parents[1]
    files = [repo/"README.md", repo/".gitignore", member/"README.md", member/"requirements.txt",
             member/"evaluate_local.py", member/"results.md", member/"failure_analysis.md"]
    files += sorted((member/"src").glob("*.py")) + sorted((member/"src").glob("*.ipynb"))
    files += sorted((member/"configs").glob("*.yaml"))
    files += sorted((member/"reference").glob("*.ipynb"))
    files = [p for p in files if p.exists()]
    manifest = {p.relative_to(repo).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(repo).as_posix())
        z.writestr("CODE_BUNDLE_MANIFEST.json", json.dumps(manifest, indent=2))
    print("Created", output, "with", len(files), "source/config/notebook files")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="artifacts/task3-code.zip")
    package(p.parse_args().output)
