import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
src_path = ROOT / "outputs" / "gpu" / "submission.csv"
out_path = ROOT / "outputs" / "gpu" / "kaggle_submission.csv"

with open(src_path, newline="") as f:
    rows = list(csv.DictReader(f))

overall = next(r for r in rows if r["direction"] == "overall")
fid, mifid = overall["FID"], overall["MiFID"]

with open(out_path, "w", encoding="utf-8", newline="\n") as f:
    f.write("ID,FID,MiFID\n")
    f.write(f"1,{fid},{mifid}\n")

print(f"wrote {out_path}")
