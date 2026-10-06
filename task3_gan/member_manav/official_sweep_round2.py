import glob
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from scipy.spatial.distance import cosine
from tqdm import tqdm

sys.path.append("src")
from data import ROOT, load_config, make_transforms, ImageDataset
from models import ResNetGenerator
from utils import write_predictions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

N_EVAL = 300
BATCH_SIZE = 32


# ===== verbatim from official_eval.ipynb (same as official_sweep.py) =====

def list_images(folder):
    exts = (".jpg", ".jpeg", ".png")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
    paths = sorted(list(set(paths)))
    return paths


def take_n(paths, n):
    if n is None:
        return paths
    return paths[:min(n, len(paths))]


_inception_cache = None


def get_inception_model():
    global _inception_cache
    if _inception_cache is not None:
        return _inception_cache
    inception = models.inception_v3(
        weights=models.Inception_V3_Weights.IMAGENET1K_V1,
        transform_input=False
    )
    inception.fc = nn.Identity()
    inception.to(device)
    inception.eval()
    _inception_cache = inception
    return inception


INCEPTION_TF = T.Compose([
    T.Resize(299),
    T.CenterCrop(299),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


def load_batch(paths):
    imgs = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        imgs.append(INCEPTION_TF(img))
    return torch.stack(imgs, dim=0)


@torch.no_grad()
def get_activations(model, image_paths, batch_size=32):
    feats = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Inception activations"):
        batch_paths = image_paths[i:i+batch_size]
        x = load_batch(batch_paths).to(device)
        f = model(x).detach().cpu().numpy()
        feats.append(f)
    return np.concatenate(feats, axis=0)


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    diff = mu1 - mu2
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean))


def calculate_fid_mifid(real_paths, gen_paths, batch_size=32, subsample_to_match=True):
    real_paths = sorted(real_paths)
    gen_paths = sorted(gen_paths)

    if subsample_to_match:
        n = min(len(real_paths), len(gen_paths))
        real_paths = real_paths[:n]
        gen_paths = gen_paths[:n]

    model = get_inception_model()

    real_act = get_activations(model, real_paths, batch_size=batch_size)
    gen_act = get_activations(model, gen_paths, batch_size=batch_size)

    mu_r, sig_r = real_act.mean(axis=0), np.cov(real_act, rowvar=False)
    mu_g, sig_g = gen_act.mean(axis=0), np.cov(gen_act, rowvar=False)

    fid = frechet_distance(mu_r, sig_r, mu_g, sig_g)

    m = min(len(real_act), len(gen_act))
    cos_dists = [cosine(real_act[i], gen_act[i]) for i in range(m)]
    mifid = float(np.mean(cos_dists))

    return fid, mifid

# ===== end verbatim section =====


cfg = load_config("gpu")
data_dir = (ROOT / cfg["paths"]["data_dir"]).resolve()
out_dir = ROOT / "outputs"
sweep_tmp_dir = out_dir / "_sweep_round2_tmp"

_, plain_tf = make_transforms(cfg)

monet_files = sorted(Path(data_dir / "monet_jpg").glob("*.jpg"))
photo_files_all = sorted(Path(data_dir / "photo_jpg").glob("*.jpg"))
photo_files_300 = photo_files_all[:300]

real_monet_paths = take_n(list_images(str(data_dir / "monet_jpg")), N_EVAL)
real_photo_paths = take_n(list_images(str(data_dir / "photo_jpg")), N_EVAL)

monet_ds = ImageDataset(monet_files, plain_tf)
photo300_ds = ImageDataset(photo_files_300, plain_tf)


def load_std_generators(ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    g_a2b = ResNetGenerator(cfg["model"]["ngf"], cfg["model"]["n_res_blocks"])
    g_a2b.load_state_dict(state["G_A2B"], strict=True)
    g_a2b.to(device).eval()

    g_b2a = ResNetGenerator(cfg["model"]["ngf"], cfg["model"]["n_res_blocks"])
    g_b2a.load_state_dict(state["G_B2A"], strict=True)
    g_b2a.to(device).eval()
    return g_a2b, g_b2a


def load_100k_generators():
    g_a2b = ResNetGenerator(cfg["model"]["ngf"], cfg["model"]["n_res_blocks"])
    g_a2b.load_state_dict(
        torch.load(ROOT / "checkpoints" / "manav_task3_G_A2B.pt", map_location="cpu", weights_only=False),
        strict=True,
    )
    g_a2b.to(device).eval()

    g_b2a = ResNetGenerator(cfg["model"]["ngf"], cfg["model"]["n_res_blocks"])
    g_b2a.load_state_dict(
        torch.load(ROOT / "checkpoints" / "manav_task3_G_B2A.pt", map_location="cpu", weights_only=False),
        strict=True,
    )
    g_b2a.to(device).eval()
    return g_a2b, g_b2a


checkpoints_to_sweep = []
for step in range(105000, 120001, 5000):
    checkpoints_to_sweep.append(("gpu_ft_id1", step, ROOT / "checkpoints" / "gpu_ft_id1" / f"ema_step{step}.pt"))
for step in range(125000, 140001, 5000):
    checkpoints_to_sweep.append(("gpu_ft_c5", step, ROOT / "checkpoints" / "gpu_ft_c5" / f"ema_step{step}.pt"))
for step in range(5000, 60001, 5000):
    checkpoints_to_sweep.append(("gpu_id05_sn", step, ROOT / "checkpoints" / "gpu_id05_sn" / f"ema_step{step}.pt"))
checkpoints_to_sweep.append(("gpu_100k_original", 100000, None))

print(f"Total checkpoints to sweep: {len(checkpoints_to_sweep)}")

rows = []
for run_name, step, ckpt_path in checkpoints_to_sweep:
    print(f"\n=== {run_name} step {step} ===")
    if ckpt_path is None:
        g_a2b, g_b2a = load_100k_generators()
    else:
        g_a2b, g_b2a = load_std_generators(ckpt_path)

    step_dir = sweep_tmp_dir / f"{run_name}_step{step}"
    pred_a2b_dir = step_dir / "pred_A2B"
    pred_b2a_dir = step_dir / "pred_B2A"

    write_predictions(monet_ds, g_a2b, pred_a2b_dir, device)
    write_predictions(photo300_ds, g_b2a, pred_b2a_dir, device)

    gen_a2b_paths = take_n(list_images(str(pred_a2b_dir)), N_EVAL)
    gen_b2a_paths = take_n(list_images(str(pred_b2a_dir)), N_EVAL)

    fid_b2a, mifid_b2a = calculate_fid_mifid(real_monet_paths, gen_b2a_paths, batch_size=BATCH_SIZE)
    print(f"[Photo->Monet] FID={fid_b2a:.3f}  MiFID={mifid_b2a:.4f}")

    fid_a2b, mifid_a2b = calculate_fid_mifid(real_photo_paths, gen_a2b_paths, batch_size=BATCH_SIZE)
    print(f"[Monet->Photo] FID={fid_a2b:.3f}  MiFID={mifid_a2b:.4f}")

    fid_avg = (fid_a2b + fid_b2a) / 2
    mifid_avg = (mifid_a2b + mifid_b2a) / 2
    leaderboard_score = -(fid_avg + mifid_avg) / 2
    print(f"leaderboard_score={leaderboard_score:.4f}")

    rows.append({
        "run": run_name,
        "step": step,
        "FID_A2B": fid_a2b,
        "MiFID_A2B": mifid_a2b,
        "FID_B2A": fid_b2a,
        "MiFID_B2A": mifid_b2a,
        "leaderboard_score": leaderboard_score,
    })

    shutil.rmtree(step_dir)
    print(f"deleted {step_dir}")

    pd.DataFrame(rows).to_csv(out_dir / "official_sweep_round2.csv", index=False)

df = pd.DataFrame(rows)
df_sorted = df.sort_values("leaderboard_score", ascending=False)
print("\n=== full table sorted by leaderboard score ===")
print(df_sorted.to_string(index=False))

if sweep_tmp_dir.exists():
    shutil.rmtree(sweep_tmp_dir)

print("\nSWEEP_ROUND2_DONE")
