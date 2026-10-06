import shutil
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.append(".")
sys.path.append("src")

from data import ROOT, load_config, make_transforms, ImageDataset
from models import ResNetGenerator
import evaluate_local as ev

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = load_config("gpu_ft_c1")
data_dir = (ROOT / cfg["paths"]["data_dir"]).resolve()
_, plain_tf = make_transforms(cfg)
monet_files = sorted(Path(data_dir / "monet_jpg").glob("*.jpg"))
photo_files = sorted(Path(data_dir / "photo_jpg").glob("*.jpg"))
monet_ds = ImageDataset(monet_files, plain_tf)
photo_ds = ImageDataset(photo_files, plain_tf)

work_dir = ROOT / "outputs" / "_diag_tmp"
scorer = ev.OfficialQuickScorer(device, monet_ds, photo_ds, data_dir, work_dir)


def build_gen(sd):
    g = ResNetGenerator(cfg["model"]["ngf"], cfg["model"]["n_res_blocks"])
    g.load_state_dict(sd)
    return g.to(device).eval()


def load_ema(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    return state["G_A2B"], state["G_B2A"]


def avg_state_dicts(sds):
    keys = sds[0].keys()
    return {k: sum(sd[k].float() for sd in sds) / len(sds) for k in keys}


rows = []


def score_and_record(name, a2b_sd, b2a_sd):
    g_a2b = build_gen(a2b_sd)
    g_b2a = build_gen(b2a_sd)
    result = scorer.score(g_a2b, g_b2a, device)
    rows.append({"name": name, **result})
    print(name, "fid_a2b", round(result["fid_a2b"], 3), "fid_b2a", round(result["fid_b2a"], 3))


ft_c1_dir = ROOT / "checkpoints" / "gpu_ft_c1"
ft_c25_dir = ROOT / "checkpoints" / "gpu_ft_c25"

ema_sds = {}
for step in [165000, 170000, 175000, 180000]:
    a2b, b2a = load_ema(ft_c1_dir / f"ema_step{step}.pt")
    ema_sds[f"ft_c1_{step}"] = (a2b, b2a)
    score_and_record(f"ft_c1_{step}", a2b, b2a)

for step in [155000, 160000]:
    a2b, b2a = load_ema(ft_c25_dir / f"ema_step{step}.pt")
    ema_sds[f"ft_c25_{step}"] = (a2b, b2a)
    score_and_record(f"ft_c25_{step}", a2b, b2a)

last_state = torch.load(ft_c1_dir / "last.pt", map_location="cpu", weights_only=False)
print("ft_c1 last.pt step:", last_state["step"])
score_and_record("ft_c1_180000_raw", last_state["models"]["G_A2B"], last_state["models"]["G_B2A"])
score_and_record("ft_c1_180000_ema_fromlast", last_state["models"]["ema_A2B"], last_state["models"]["ema_B2A"])

groups = {
    "avg_175000_180000": ["ft_c1_175000", "ft_c1_180000"],
    "avg_170000_175000_180000": ["ft_c1_170000", "ft_c1_175000", "ft_c1_180000"],
    "avg_165000_170000_175000_180000": ["ft_c1_165000", "ft_c1_170000", "ft_c1_175000", "ft_c1_180000"],
    "avg_160000_165000_170000_175000_180000": ["ft_c25_160000", "ft_c1_165000", "ft_c1_170000", "ft_c1_175000", "ft_c1_180000"],
}
for name, members in groups.items():
    a2b_avg = avg_state_dicts([ema_sds[m][0] for m in members])
    b2a_avg = avg_state_dicts([ema_sds[m][1] for m in members])
    score_and_record(name, a2b_avg, b2a_avg)

df = pd.DataFrame(rows)

individual_names = ["ft_c1_165000", "ft_c1_170000", "ft_c1_175000", "ft_c1_180000", "ft_c25_155000", "ft_c25_160000"]
avg_names = list(groups.keys())

best_single = df[df["name"].isin(individual_names)].loc[lambda d: d["quick_score"].idxmin()]
best_avg = df[df["name"].isin(avg_names)].loc[lambda d: d["quick_score"].idxmin()]
best_a2b_row = df.loc[df["fid_a2b"].idxmin()]
best_b2a_row = df.loc[df["fid_b2a"].idxmin()]
fid_avg = (best_a2b_row["fid_a2b"] + best_b2a_row["fid_b2a"]) / 2
mifid_avg = (best_a2b_row["mifid_a2b"] + best_b2a_row["mifid_b2a"]) / 2
mixed_score = (fid_avg + mifid_avg) / 2

real_photo_paths = ev.official_list_images(str(data_dir / "photo_jpg"))
real_monet_paths = ev.official_list_images(str(data_dir / "monet_jpg"))

floor_rows = []
for n in [100, 150, 200, 300]:
    a, b = real_photo_paths[:n], real_photo_paths[n:2 * n]
    fid, mifid = ev.official_calculate_fid_mifid(a, b, device)
    floor_rows.append({"name": f"floor_photo_n{n}", "domain": "photo", "n": n, "fid": fid, "mifid": mifid})
    print("floor photo n", n, "fid", round(fid, 3))
    if 2 * n <= len(real_monet_paths):
        a, b = real_monet_paths[:n], real_monet_paths[n:2 * n]
        fid, mifid = ev.official_calculate_fid_mifid(a, b, device)
        floor_rows.append({"name": f"floor_monet_n{n}", "domain": "monet", "n": n, "fid": fid, "mifid": mifid})
        print("floor monet n", n, "fid", round(fid, 3))

floor_df = pd.DataFrame(floor_rows)

shutil.rmtree(work_dir, ignore_errors=True)

full = pd.concat([df, floor_df], ignore_index=True)
full.to_csv(ROOT / "outputs" / "diagnostics.csv", index=False)

print()
print(df[["name", "fid_a2b", "mifid_a2b", "fid_b2a", "mifid_b2a", "quick_score"]].to_string(index=False))
print()
print(floor_df.to_string(index=False))
print()
print("best_single:", best_single["name"], "quick_score", round(best_single["quick_score"], 3))
print("best_avg:", best_avg["name"], "quick_score", round(best_avg["quick_score"], 3))
print("best_a2b:", best_a2b_row["name"], "fid_a2b", round(best_a2b_row["fid_a2b"], 3))
print("best_b2a:", best_b2a_row["name"], "fid_b2a", round(best_b2a_row["fid_b2a"], 3))
print("mixed_score (a2b+b2a avg, positive, lower better):", round(mixed_score, 3))
