import argparse
import sys
from pathlib import Path

import lpips
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import linalg
from torchmetrics.image.fid import NoTrainInceptionV3

sys.path.append(str(Path(__file__).resolve().parent / "src"))
from data import ROOT, list_images, load_config
from models import ResNetGenerator

SUBMISSION_COLUMNS = {"direction": "direction", "fid": "FID", "mifid": "MiFID", "score": "score"}
INCEPTION_FILE = "weights-inception-2015-12-05-6726825d.pth"
ALEXNET_FILE = "alexnet-owt-7be5be79.pth"


def load_inception(device):
    return NoTrainInceptionV3(name="inception-v3-compat", features_list=["2048"]).to(device).eval()


def to_uint8(images):
    return ((images.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


def to_float(images):
    return images.float() / 127.5 - 1


def load_uint8(files, size):
    images = []
    for file in files:
        with Image.open(file) as image:
            image = image.convert("RGB")
            if image.size != (size, size):
                image = image.resize((size, size), Image.BILINEAR)
            images.append(torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1))
    return torch.stack(images)


@torch.no_grad()
def get_features(inception, images, device, batch_size=50):
    features = []
    for i in range(0, len(images), batch_size):
        features.append(inception(images[i : i + batch_size].to(device)).double().cpu())
    return torch.cat(features).numpy()


def file_features(inception, files, size, device, batch_size=50):
    features = []
    for i in range(0, len(files), batch_size):
        batch = load_uint8(files[i : i + batch_size], size)
        features.append(get_features(inception, batch, device, batch_size))
    return np.concatenate(features)


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


def subsample_pair(fake, real, seed=42):
    n = min(len(fake), len(real))
    rng = np.random.default_rng(seed)
    return fake[rng.permutation(len(fake))[:n]], real[rng.permutation(len(real))[:n]]


def mifid_from_features(fake, real, seed=42):
    a, b = subsample_pair(fake, real, seed)
    cosine = (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return float((1 - cosine).mean())


def kid_from_features(real, fake, n_subsets=100, subset_size=1000, seed=42):
    n = min(len(real), len(fake), subset_size)
    d = real.shape[1]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_subsets):
        x = real[rng.choice(len(real), n, replace=False)]
        y = fake[rng.choice(len(fake), n, replace=False)]
        kxx = (x @ x.T / d + 1) ** 3
        kyy = (y @ y.T / d + 1) ** 3
        kxy = (x @ y.T / d + 1) ** 3
        scale = n * (n - 1)
        values.append((kxx.sum() - np.trace(kxx)) / scale + (kyy.sum() - np.trace(kyy)) / scale - 2 * kxy.mean())
    return float(np.mean(values)), float(np.std(values))


def precision_recall(real, fake, k=3, seed=42):
    fake, real = subsample_pair(fake, real, seed)
    real_t, fake_t = torch.from_numpy(real), torch.from_numpy(fake)
    real_radius = torch.cdist(real_t, real_t).sort(dim=1).values[:, k]
    fake_radius = torch.cdist(fake_t, fake_t).sort(dim=1).values[:, k]
    distances = torch.cdist(fake_t, real_t)
    precision = (distances <= real_radius[None, :]).any(dim=1).double().mean().item()
    recall = (distances.T <= fake_radius[None, :]).any(dim=1).double().mean().item()
    return precision, recall


def cosine_per_row(a, b):
    return (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def content_cosine(input_features, pred_features):
    return float(cosine_per_row(input_features, pred_features).mean())


@torch.no_grad()
def lpips_per_image(net, input_files, pred_files, size, device, batch_size=25):
    values = []
    for i in range(0, len(input_files), batch_size):
        a = to_float(load_uint8(input_files[i : i + batch_size], size)).to(device)
        b = to_float(load_uint8(pred_files[i : i + batch_size], size)).to(device)
        values.append(net(a, b).flatten().cpu())
    return torch.cat(values).numpy()


def mean_lpips(net, input_files, pred_files, size, device, batch_size=25):
    return float(lpips_per_image(net, input_files, pred_files, size, device, batch_size).mean())


@torch.no_grad()
def mean_cycle_l1(first, second, files, size, device, batch_size=25):
    total = 0.0
    for i in range(0, len(files), batch_size):
        x = to_float(load_uint8(files[i : i + batch_size], size)).to(device)
        total += (second(first(x)) - x).abs().mean(dim=[1, 2, 3]).sum().item()
    return total / len(files)


class QuickScorer:
    def __init__(self, device, monets, photos, batch_size=25):
        self.device = device if device.type == "cuda" else torch.device("cpu")
        self.monets = monets
        self.photos = photos
        self.batch_size = batch_size
        self.inception = load_inception(self.device)
        self.monet_features = get_features(self.inception, to_uint8(monets), self.device)
        self.photo_features = get_features(self.inception, to_uint8(photos), self.device)

    @torch.no_grad()
    def translate(self, generator, images, device):
        out = []
        for i in range(0, len(images), self.batch_size):
            out.append(generator(images[i : i + self.batch_size].to(device)).cpu())
        return to_uint8(torch.cat(out))

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


def load_generators(cfg, ckpt_path, device):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "models" in state:
        weights = [state["models"]["ema_A2B"], state["models"]["ema_B2A"]]
    else:
        weights = [state["G_A2B"], state["G_B2A"]]
    nets = []
    for w in weights:
        net = ResNetGenerator(cfg["model"]["ngf"], cfg["model"]["n_res_blocks"])
        net.load_state_dict(w)
        nets.append(net.to(device).eval())
    return nets


def check_predictions(folders, size):
    errors = []
    for direction, (folder, input_files) in folders.items():
        if not folder.is_dir():
            errors.append(f"{direction}: folder {folder.name} does not exist")
            continue
        names = sorted(p.name for p in folder.iterdir() if not p.name.startswith("."))
        expected = sorted(f.name for f in input_files)
        if len(names) != len(expected):
            errors.append(f"{direction}: found {len(names)} files but expected {len(expected)}")
        missing, extra = sorted(set(expected) - set(names)), sorted(set(names) - set(expected))
        if missing:
            errors.append(f"{direction}: {len(missing)} input names have no prediction, for example {missing[:3]}")
        if extra:
            errors.append(f"{direction}: {len(extra)} predictions match no input, for example {extra[:3]}")
        bad = []
        for name in names:
            try:
                with Image.open(folder / name) as image:
                    if image.format != "JPEG" or image.mode != "RGB" or image.size != (size, size):
                        bad.append(f"{name} ({image.format}, {image.mode}, {image.size})")
            except OSError:
                bad.append(f"{name} (cannot be opened)")
        if bad:
            errors.append(f"{direction}: {len(bad)} files are not {size}x{size} RGB JPG, for example {bad[:3]}")
    if errors:
        raise SystemExit("Prediction check failed:\n  " + "\n  ".join(errors))


def show_weight_paths():
    hub = Path(torch.hub.get_dir()) / "checkpoints"
    paths = [
        ("Inception-v3 (FID)", hub / INCEPTION_FILE),
        ("AlexNet (LPIPS backbone)", hub / ALEXNET_FILE),
        ("LPIPS linear layers", Path(lpips.__file__).parent / "weights" / "v0.1" / "alex.pth"),
    ]
    print("pretrained weights used only for measuring images:")
    for name, path in paths:
        size = f"{path.stat().st_size / 2**20:.1f} MB" if path.exists() else "NOT FOUND"
        print(f"  {name}: {path} ({size})")


def direction_metrics(fake, real, inputs, pred_files, input_files, lpips_net, cycle, size, device):
    precision, recall = precision_recall(real, fake)
    kid_mean, kid_std = kid_from_features(real, fake)
    fid, mifid = fid_from_features(fake, real), mifid_from_features(fake, real)
    return {
        "FID": fid,
        "MiFID": mifid,
        "score": (fid + mifid) / 2,
        "KID_mean": kid_mean,
        "KID_std": kid_std,
        "precision": precision,
        "recall": recall,
        "LPIPS": mean_lpips(lpips_net, input_files, pred_files, size, device),
        "content_cosine": content_cosine(inputs, fake),
        "cycle_L1": mean_cycle_l1(cycle[0], cycle[1], input_files, size, device),
        "n_fake": len(fake),
        "n_real": len(real),
    }


def evaluate(config, ckpt, work_dir=None, verbose=True):
    cfg = load_config(config)
    size = cfg["data"]["image_size"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = (ROOT / cfg["paths"]["data_dir"]).resolve()
    out_dir = Path(work_dir) if work_dir else ROOT / cfg["paths"]["output_dir"]
    monet_files = list_images(data_dir / "monet_jpg", cfg["data"]["monet_limit"], cfg["seed"])
    photo_files = list_images(data_dir / "photo_jpg", cfg["data"]["photo_limit"], cfg["seed"])
    pred_a2b, pred_b2a = out_dir / "pred_A2B", out_dir / "pred_B2A"
    check_predictions({"A2B": (pred_a2b, monet_files), "B2A": (pred_b2a, photo_files)}, size)
    if verbose:
        print(f"checks passed: {len(monet_files)} A2B and {len(photo_files)} B2A images, all {size}x{size} RGB JPG")

    inception = load_inception(device)
    lpips_net = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    if verbose:
        show_weight_paths()
    g_a2b, g_b2a = load_generators(cfg, ckpt, device)

    pred_a2b_files = [pred_a2b / f.name for f in monet_files]
    pred_b2a_files = [pred_b2a / f.name for f in photo_files]
    monet_real = file_features(inception, monet_files, size, device)
    photo_real = file_features(inception, photo_files, size, device)
    photo_fake = file_features(inception, pred_a2b_files, size, device)
    monet_fake = file_features(inception, pred_b2a_files, size, device)

    rows = {
        "A2B": direction_metrics(photo_fake, photo_real, monet_real, pred_a2b_files, monet_files,
                                 lpips_net, (g_a2b, g_b2a), size, device),
        "B2A": direction_metrics(monet_fake, monet_real, photo_real, pred_b2a_files, photo_files,
                                 lpips_net, (g_b2a, g_a2b), size, device),
    }
    report = pd.DataFrame(rows).T
    report.loc["overall"] = report.loc[["A2B", "B2A"]].mean()
    report.loc["overall", ["n_fake", "n_real"]] = np.nan
    report.index.name = "direction"
    report = report.reset_index()

    cols = SUBMISSION_COLUMNS
    submission = report[["direction", "FID", "MiFID", "score"]].rename(
        columns={"direction": cols["direction"], "FID": cols["fid"], "MiFID": cols["mifid"], "score": cols["score"]})
    out_dir.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_dir / "submission.csv", index=False)
    report.to_csv(out_dir / "full_metrics_report.csv", index=False)
    return submission, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="local")
    parser.add_argument("--ckpt", required=True)
    args = parser.parse_args()
    submission, report = evaluate(args.config, args.ckpt)
    print(submission.round(4).to_string(index=False))
