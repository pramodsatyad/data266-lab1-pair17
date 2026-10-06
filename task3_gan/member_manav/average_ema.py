import sys
from pathlib import Path

import torch

sys.path.append(".")
sys.path.append("src")

from data import ROOT, load_config, make_transforms, ImageDataset
from models import ResNetGenerator
import evaluate_local as ev

run_name = sys.argv[1]
start_step = int(sys.argv[2])
end_step = int(sys.argv[3])

cfg = load_config(run_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = (ROOT / cfg["paths"]["data_dir"]).resolve()
_, plain_tf = make_transforms(cfg)
monet_files = sorted(Path(data_dir / "monet_jpg").glob("*.jpg"))
photo_files = sorted(Path(data_dir / "photo_jpg").glob("*.jpg"))
monet_ds = ImageDataset(monet_files, plain_tf)
photo_ds = ImageDataset(photo_files, plain_tf)

ckpt_dir = ROOT / cfg["paths"]["checkpoint_dir"]
steps = []
for p in sorted(ckpt_dir.glob("ema_step*.pt")):
    step = int(p.stem.replace("ema_step", ""))
    if start_step <= step <= end_step:
        steps.append(step)

if not steps:
    print("no ema files found in range")
    sys.exit(1)

print("averaging steps:", steps)

a2b_sds = []
b2a_sds = []
for step in steps:
    state = torch.load(ckpt_dir / f"ema_step{step}.pt", map_location="cpu", weights_only=False)
    a2b_sds.append(state["G_A2B"])
    b2a_sds.append(state["G_B2A"])


def avg_state_dicts(sds):
    keys = sds[0].keys()
    return {k: sum(sd[k].float() for sd in sds) / len(sds) for k in keys}


a2b_avg = avg_state_dicts(a2b_sds)
b2a_avg = avg_state_dicts(b2a_sds)


def build_gen(sd):
    g = ResNetGenerator(cfg["model"]["ngf"], cfg["model"]["n_res_blocks"], cfg["model"].get("upsample", "transpose"))
    g.load_state_dict(sd)
    return g.to(device).eval()


g_a2b = build_gen(a2b_avg)
g_b2a = build_gen(b2a_avg)

work_dir = ROOT / "outputs" / "_avg_ema_tmp"
scorer = ev.OfficialQuickScorer(device, monet_ds, photo_ds, data_dir, work_dir)
result = scorer.score(g_a2b, g_b2a, device)

print("fid_a2b", result["fid_a2b"])
print("fid_b2a", result["fid_b2a"])
print("mifid_a2b", result["mifid_a2b"])
print("mifid_b2a", result["mifid_b2a"])
print("quick_score", result["quick_score"])
