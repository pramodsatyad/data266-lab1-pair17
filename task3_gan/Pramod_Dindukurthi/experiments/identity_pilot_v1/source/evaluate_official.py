"""Instructor evaluator adapted to a CLI; same features, sorting, count caps and formulas.
Reference: reference/Part3_Evaluation_Script.ipynb, retained unchanged.
Differences: CLI paths, CSV writer instead of pandas, input checks, SciPy compatibility.
These official-protocol scores are distinct from torch-fidelity local diagnostics.
"""
import os, glob
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models

import scipy.linalg
from scipy.spatial.distance import cosine

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

# Inception feature extractor
def get_inception_model():
    inception = models.inception_v3(
        weights=models.Inception_V3_Weights.IMAGENET1K_V1,
        transform_input=False
    )
    inception.fc = nn.Identity()
    inception.to(device)
    inception.eval()
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
    # Support SciPy versions with and without the deprecated disp parameter.
    try:
        covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    except TypeError:
        covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    diff = mu1 - mu2
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean))

def calculate_fid_mifid(real_paths, gen_paths, batch_size=32, subsample_to_match=True):
    # For fair comparison, match counts
    real_paths = sorted(real_paths)
    gen_paths  = sorted(gen_paths)

    if subsample_to_match:
        n = min(len(real_paths), len(gen_paths))
        real_paths = real_paths[:n]
        gen_paths  = gen_paths[:n]

    model = get_inception_model()

    real_act = get_activations(model, real_paths, batch_size=batch_size)
    gen_act  = get_activations(model, gen_paths,  batch_size=batch_size)

    mu_r, sig_r = real_act.mean(axis=0), np.cov(real_act, rowvar=False)
    mu_g, sig_g = gen_act.mean(axis=0),  np.cov(gen_act,  rowvar=False)

    fid = frechet_distance(mu_r, sig_r, mu_g, sig_g)

    # mean cosine distance between feature vectors,
    # paired by index after subsampling/matching
    m = min(len(real_act), len(gen_act))
    cos_dists = [cosine(real_act[i], gen_act[i]) for i in range(m)]
    mifid = float(np.mean(cos_dists))

    return fid, mifid

def main():
    import argparse, csv, json, hashlib
    from pathlib import Path
    global device
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True)
    p.add_argument("--export-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--n-eval", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    if a.n_eval < 2: p.error("n-eval must be >=2")
    device = torch.device(a.device)
    data, export = Path(a.data_root), Path(a.export_dir)
    result = {}
    counts = {}
    for direction, target in [("A2B", "photo_jpg"), ("B2A", "monet_jpg")]:
        real = take_n(list_images(str(data/target)), a.n_eval)
        generated = take_n(list_images(str(export/("pred_"+direction))), a.n_eval)
        if min(len(real), len(generated)) < 2:
            raise ValueError("Need >=2 real/generated images in " + direction)
        counts[direction] = {"real_capped": len(real), "generated_capped": len(generated), "matched": min(len(real), len(generated))}
        print(direction, counts[direction], flush=True)
        fid, mifid = calculate_fid_mifid(real, generated, batch_size=a.batch_size)
        if not np.isfinite([fid, mifid]).all(): raise ValueError("Non-finite official metrics")
        result[direction] = {"FID": fid, "MiFID": mifid}
    row = {"ID": 1, "FID": (result["A2B"]["FID"]+result["B2A"]["FID"])/2,
           "MiFID": (result["A2B"]["MiFID"]+result["B2A"]["MiFID"])/2}
    output = Path(a.output); output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists(): raise FileExistsError("Preserve previous submission; choose a new output filename")
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "FID", "MiFID"])
        writer.writeheader(); writer.writerow(row)
    provenance = {"per_direction": result, "submission": row, "counts": counts,
                  "n_eval": a.n_eval, "protocol": "instructor torchvision Inception_V3_Weights.IMAGENET1K_V1; sorted first-N; index cosine", 
                  "evaluation_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "note": "Real references follow instructor folders, including training images; not a held-out generalization score."}
    manifest = export/"export_manifest.json"
    if manifest.exists(): provenance["export_manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    output.with_suffix(".metrics.json").write_text(json.dumps(provenance, indent=2))
    print(row, flush=True)


if __name__ == "__main__":
    main()
