import numpy as np
import torch
from scipy import linalg
from torchmetrics.image.fid import NoTrainInceptionV3


def load_inception(device):
    return NoTrainInceptionV3(name="inception-v3-compat", features_list=["2048"]).to(device).eval()


def to_uint8(images):
    return ((images.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


@torch.no_grad()
def get_features(inception, images, device, batch_size=50):
    features = []
    for i in range(0, len(images), batch_size):
        batch = to_uint8(images[i : i + batch_size]).to(device)
        features.append(inception(batch).double().cpu())
    return torch.cat(features).numpy()


def fid_from_features(a, b):
    mu_a, mu_b = a.mean(axis=0), b.mean(axis=0)
    cov_a, cov_b = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean = linalg.sqrtm(cov_a.dot(cov_b))
    if not np.isfinite(covmean).all():
        offset = np.eye(cov_a.shape[0]) * 1e-6
        covmean = linalg.sqrtm((cov_a + offset).dot(cov_b + offset))
    covmean = covmean.real
    diff = mu_a - mu_b
    return float(diff.dot(diff) + np.trace(cov_a) + np.trace(cov_b) - 2 * np.trace(covmean))


def mifid_from_features(fake, real, seed=42):
    n = min(len(fake), len(real))
    rng = np.random.default_rng(seed)
    a = fake[rng.permutation(len(fake))[:n]]
    b = real[rng.permutation(len(real))[:n]]
    cosine = (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return float((1 - cosine).mean())


class QuickScorer:
    def __init__(self, device, monets, photos, batch_size=25):
        self.device = device if device.type == "cuda" else torch.device("cpu")
        self.monets = monets
        self.photos = photos
        self.batch_size = batch_size
        self.inception = load_inception(self.device)
        self.monet_features = get_features(self.inception, monets, self.device)
        self.photo_features = get_features(self.inception, photos, self.device)

    @torch.no_grad()
    def translate(self, generator, images, device):
        out = []
        for i in range(0, len(images), self.batch_size):
            out.append(generator(images[i : i + self.batch_size].to(device)).cpu())
        return torch.cat(out)

    def score(self, g_a2b, g_b2a, device):
        fake_monet = get_features(self.inception, self.translate(g_b2a, self.photos, device), self.device)
        fake_photo = get_features(self.inception, self.translate(g_a2b, self.monets, device), self.device)
        result = {}
        for name, fake, real in [("b2a", fake_monet, self.monet_features), ("a2b", fake_photo, self.photo_features)]:
            result[f"fid_{name}"] = fid_from_features(fake, real)
            result[f"mifid_{name}"] = mifid_from_features(fake, real)
            result[f"score_{name}"] = (result[f"fid_{name}"] + result[f"mifid_{name}"]) / 2
        result["quick_score"] = (result["score_a2b"] + result["score_b2a"]) / 2
        return result
