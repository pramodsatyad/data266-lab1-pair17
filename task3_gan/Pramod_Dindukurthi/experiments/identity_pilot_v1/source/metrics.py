"""Documented local diagnostics, NOT the unknown official Kaggle evaluator.

Inception: torch-fidelity TF-compatible 2048-D features, uint8 RGB input.
KID: unbiased polynomial MMD^2 (negative estimates are valid), repeated subsets.
Density/coverage: Naeem et al. https://arxiv.org/abs/2002.09797.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
import torch

from common import sha256
from data import rgb


def feature_checks(real, fake):
    real, fake = np.asarray(real, dtype=np.float64), np.asarray(fake, dtype=np.float64)
    if real.ndim != 2 or fake.ndim != 2 or real.shape[1] != fake.shape[1]:
        raise ValueError("Features must be Nxd and Mxd with matching dimensions")
    if min(len(real), len(fake)) < 2 or not np.isfinite(real).all() or not np.isfinite(fake).all():
        raise ValueError("Need >=2 finite features in both sets")
    return real, fake


def fid(real, fake):
    real, fake = feature_checks(real, fake)
    mean_distance = np.sum((real.mean(0)-fake.mean(0))**2)
    cr, cf = np.cov(real, rowvar=False), np.cov(fake, rowvar=False)
    # Symmetric PSD formulation avoids complex-valued sqrtm artifacts.
    values, vectors = np.linalg.eigh(cr)
    root = (vectors * np.sqrt(np.maximum(values, 0))) @ vectors.T
    product = root @ cf @ root
    middle = np.linalg.eigvalsh((product+product.T)*0.5)
    value = mean_distance + np.trace(cr) + np.trace(cf) - 2*np.sqrt(np.maximum(middle, 0)).sum()
    return float(max(value, 0))


def kid(real, fake, seed=42, subsets=50, subset_size=1000):
    real, fake = feature_checks(real, fake)
    n = min(subset_size, len(real), len(fake))
    if n < 2 or subsets < 1:
        raise ValueError("KID needs subset size >=2 and >=1 subsets")
    rng, values = np.random.default_rng(seed), []
    for _ in range(subsets):
        x, y = real[rng.choice(len(real), n, replace=False)], fake[rng.choice(len(fake), n, replace=False)]
        xx, yy, xy = (x@x.T/x.shape[1]+1)**3, (y@y.T/y.shape[1]+1)**3, (x@y.T/x.shape[1]+1)**3
        values.append((xx.sum()-np.trace(xx)+yy.sum()-np.trace(yy))/(n*(n-1))-2*xy.mean())
    return {"KID_mean": float(np.mean(values)), "KID_subset_std": float(np.std(values)),
            "KID_subset_size": n, "KID_subsets": subsets}


def density_coverage(real, fake, k=5, block=256):
    real, fake = feature_checks(real, fake)
    if len(real) <= k:
        raise ValueError(f"Density/coverage k={k} requires >{k} real images")
    radii = []
    for start in range(0, len(real), block):
        distances = cdist(real[start:start+block], real, metric="sqeuclidean")
        # Self distance occupies position zero; position k is kth non-self neighbor.
        radii.extend(np.partition(distances, k, axis=1)[:, k])
    radii = np.asarray(radii)
    counts, nearest = 0, np.full(len(real), np.inf)
    for start in range(0, len(fake), block):
        distances = cdist(real, fake[start:start+block], metric="sqeuclidean")
        counts += int((distances <= radii[:, None]).sum())
        nearest = np.minimum(nearest, distances.min(1))
    return {"density": counts/(k*len(fake)), "coverage": float(np.mean(nearest <= radii)), "density_k": k}


def cosine(x, y):
    if x.shape != y.shape:
        raise ValueError("Paired cosine requires identical feature shapes")
    return np.sum(x*y, axis=1) / np.maximum(np.linalg.norm(x, axis=1)*np.linalg.norm(y, axis=1), 1e-12)


class InceptionFeatures:
    def __init__(self, device, cache_dir):
        from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
        self.device, self.cache = device, Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.model = FeatureExtractorInceptionV3("inception-v3-compat", ["2048"]).to(device).eval()
        self.model.requires_grad_(False)
        h = hashlib.sha256()
        for key, value in self.model.state_dict().items():
            h.update(key.encode()); h.update(value.cpu().numpy().tobytes())
        self.weights_hash = h.hexdigest()
        self.version = importlib.metadata.version("torch-fidelity")

    @torch.inference_mode()
    def __call__(self, paths, batch_size=16):
        identity = {"sha256": [sha256(p) for p in paths], "extractor": self.version,
                    "weights_hash": self.weights_hash, "feature": "2048", "protocol": "uint8 RGB v1"}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        cache = self.cache/(key+".npy")
        if cache.exists():
            return np.load(cache)
        features = []
        for start in range(0, len(paths), batch_size):
            arrays = [np.array(rgb(p), dtype=np.uint8).transpose(2, 0, 1) for p in paths[start:start+batch_size]]
            tensor = torch.from_numpy(np.stack(arrays)).to(self.device)
            features.append(self.model(tensor)[0].cpu().numpy())
        result = np.concatenate(features)
        np.save(cache, result)
        return result


@torch.inference_mode()
def lpips_pairs(input_paths, output_paths, device, batch_size=16, model=None):
    if model is None:
        import lpips
        model = lpips.LPIPS(net="alex").to(device).eval()
    model.requires_grad_(False)
    values = []
    for start in range(0, len(input_paths), batch_size):
        def batch(paths):
            arrays = [np.array(rgb(p), dtype=np.float32).transpose(2, 0, 1)/127.5-1
                      for p in paths[start:start+batch_size]]
            return torch.from_numpy(np.stack(arrays)).to(device)
        values.extend(model(batch(input_paths), batch(output_paths)).flatten().cpu().tolist())
    return np.asarray(values), model
