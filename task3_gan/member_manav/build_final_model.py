import sys
from pathlib import Path

import torch

sys.path.append(".")
sys.path.append("src")

from data import ROOT, load_config, get_data
from models import ResNetGenerator
from utils import write_predictions, get_device
import evaluate_local as ev

output_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs/gpu_final_swa"
start_step = int(sys.argv[2]) if len(sys.argv) > 2 else 215000
end_step = int(sys.argv[3]) if len(sys.argv) > 3 else 226000
run_dirs = sys.argv[4].split(",") if len(sys.argv) > 4 else ["checkpoints/gpu_swa_a", "checkpoints/gpu_swa_a2"]

cfg = load_config("gpu_swa_a")
device, device_name = get_device()
print("device:", device_name)

a2b_sds = []
b2a_sds = []
steps_used = []
for run_dir in run_dirs:
    ckpt_dir = ROOT / run_dir
    for p in sorted(ckpt_dir.glob("ema_step*.pt")):
        step = int(p.stem.replace("ema_step", ""))
        if start_step <= step <= end_step:
            state = torch.load(p, map_location="cpu", weights_only=False)
            a2b_sds.append(state["G_A2B"])
            b2a_sds.append(state["G_B2A"])
            steps_used.append(step)

print("averaging", len(steps_used), "files, steps", steps_used)


def avg_state_dicts(sds):
    keys = sds[0].keys()
    return {k: sum(sd[k].float() for sd in sds) / len(sds) for k in keys}


a2b_avg = avg_state_dicts(a2b_sds)
b2a_avg = avg_state_dicts(b2a_sds)

out_dir = ROOT / output_dir
out_dir.mkdir(parents=True, exist_ok=True)
torch.save(a2b_avg, out_dir / "G_A2B_swa.pt")
torch.save(b2a_avg, out_dir / "G_B2A_swa.pt")
print("saved averaged weights to", out_dir)


def build_gen(sd):
    g = ResNetGenerator(cfg["model"]["ngf"], cfg["model"]["n_res_blocks"], cfg["model"].get("upsample", "transpose"))
    g.load_state_dict(sd)
    return g.to(device).eval()


g_a2b = build_gen(a2b_avg)
g_b2a = build_gen(b2a_avg)

sets, sample_ids = get_data(cfg)
n_monet = write_predictions(sets["monet_plain"], g_a2b, out_dir / "pred_A2B", device, quality=95)
n_photo = write_predictions(sets["photo_plain"], g_b2a, out_dir / "pred_B2A", device, quality=95)
print("wrote", n_monet, "to pred_A2B and", n_photo, "to pred_B2A")

data_dir = (ROOT / cfg["paths"]["data_dir"]).resolve()
work_dir = out_dir / "quick_official_tmp"
scorer = ev.OfficialQuickScorer(device, sets["monet_plain"], sets["photo_plain"], data_dir, work_dir)
result = scorer.score(g_a2b, g_b2a, device)
print("fid_a2b", result["fid_a2b"])
print("fid_b2a", result["fid_b2a"])
print("mifid_a2b", result["mifid_a2b"])
print("mifid_b2a", result["mifid_b2a"])
print("quick_score", result["quick_score"])
