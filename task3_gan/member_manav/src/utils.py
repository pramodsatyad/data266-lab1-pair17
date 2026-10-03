import os
import platform
import random
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.optim.lr_scheduler import LambdaLR


def cpu_name():
    try:
        if platform.system() == "Darwin":
            cmd = ["sysctl", "-n", "machdep.cpu.brand_string"]
            return subprocess.check_output(cmd, text=True).strip()
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.cuda.get_device_name(0)
    if torch.backends.mps.is_available():
        return torch.device("mps"), f"{cpu_name()} (MPS)"
    return torch.device("cpu"), cpu_name()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Logger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, text):
        with open(self.path, "a") as f:
            f.write(text + "\n")
        print(text)


def reset_peak_memory(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def peak_memory_mb(device):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 2**20
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / 2**20
    return None


def make_scheduler(optimizer, decay_start, total_steps):
    def lr_factor(step):
        if step < decay_start:
            return 1.0
        return max(0.0, (total_steps - step) / max(1, total_steps - decay_start))

    return LambdaLR(optimizer, lr_factor)


def save_checkpoint(path, step, models, optimizers, schedulers, pools, history):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step,
        "models": {k: v.state_dict() for k, v in models.items()},
        "optimizers": {k: v.state_dict() for k, v in optimizers.items()},
        "schedulers": {k: v.state_dict() for k, v in schedulers.items()},
        "pools": {k: [image.cpu() for image in v.images] for k, v in pools.items()},
        "history": history,
    }
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, models, optimizers, schedulers, pools, device):
    state = torch.load(path, map_location=device, weights_only=False)
    for k, v in models.items():
        v.load_state_dict(state["models"][k])
    for k, v in optimizers.items():
        v.load_state_dict(state["optimizers"][k])
    for k, v in schedulers.items():
        v.load_state_dict(state["schedulers"][k])
    for k, v in pools.items():
        v.images = [image.to(device) for image in state["pools"][k]]
    return state["step"], state["history"]


def save_ema_weights(path, ema_models):
    torch.save({k: v.state_dict() for k, v in ema_models.items()}, path)


@torch.no_grad()
def save_sample_grid(path, photos, monets, g_a2b, g_b2a, device):
    photos, monets = photos.to(device), monets.to(device)
    fake_monet = g_b2a(photos)
    fake_photo = g_a2b(monets)
    rows = [
        ("photo", photos),
        ("fake Monet", fake_monet),
        ("rebuilt photo", g_a2b(fake_monet)),
        ("Monet", monets),
        ("fake photo", fake_photo),
        ("rebuilt Monet", g_b2a(fake_photo)),
    ]
    fig, axes = plt.subplots(len(rows), photos.size(0), figsize=(2.2 * photos.size(0), 2.2 * len(rows)))
    for r, (name, images) in enumerate(rows):
        for c in range(images.size(0)):
            ax = axes[r][c]
            ax.imshow(((images[c].cpu() + 1) / 2).clamp(0, 1).permute(1, 2, 0))
            ax.set_xticks([])
            ax.set_yticks([])
        axes[r][0].set_ylabel(name)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def write_predictions(dataset, generator, out_dir, device, batch_size=16, quality=95):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = dataset.files
    for start in range(0, len(files), batch_size):
        stop = min(start + batch_size, len(files))
        batch = torch.stack([dataset[i] for i in range(start, stop)]).to(device)
        fake = generator(batch).cpu()
        pixels = ((fake.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1).numpy()
        for i, array in zip(range(start, stop), pixels):
            Image.fromarray(array).save(out_dir / files[i].name, quality=quality)
    return len(files)


CRITERIA = ["style", "content", "artifacts"]


def audit_results(sheet_1, sheet_2):
    from sklearn.metrics import cohen_kappa_score

    if not (Path(sheet_1).exists() and Path(sheet_2).exists()):
        return "The audit sheets are not there yet."
    first = pd.read_csv(sheet_1).set_index("code")
    second = pd.read_csv(sheet_2).set_index("code")
    for sheet in (first, second):
        scores = sheet[CRITERIA].apply(pd.to_numeric, errors="coerce")
        if scores.isna().any().any() or not scores.isin([1, 2, 3, 4, 5]).all().all():
            return "The audit sheets are not filled in yet. Fill both with scores from 1 to 5, then run this section again."
    if set(first.index) != set(second.index):
        return "The two sheets do not have the same codes."
    second = second.loc[first.index]
    rows = {}
    for name in CRITERIA:
        a, b = first[name].astype(int), second[name].astype(int)
        rows[name] = {
            "rater 1 average": a.mean(),
            "rater 2 average": b.mean(),
            "average score": (a.mean() + b.mean()) / 2,
            "weighted kappa": cohen_kappa_score(a, b, weights="quadratic", labels=[1, 2, 3, 4, 5]),
            "agreement %": 100 * (a == b).mean(),
        }
    return pd.DataFrame(rows).T
