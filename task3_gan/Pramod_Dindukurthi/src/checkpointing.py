"""Atomic recovery checkpoints. Load only checkpoints you trust."""
import copy
import os
from pathlib import Path
import random
import shutil

import numpy as np
import torch


class ReplayPool:
    def __init__(self, capacity):
        self.capacity, self.images = capacity, []

    def query(self, batch):
        result = []
        for image in batch.detach():
            image = image.unsqueeze(0)
            if self.capacity == 0:
                result.append(image)
            elif len(self.images) < self.capacity:
                self.images.append(image.cpu().clone())
                result.append(image)
            elif random.random() < 0.5:
                i = random.randrange(self.capacity)
                result.append(self.images[i].to(image.device).clone())
                self.images[i] = image.cpu().clone()
            else:
                result.append(image)
        return torch.cat(result)


def make_ema(model):
    ema = copy.deepcopy(model).eval()
    ema.requires_grad_(False)
    return ema


@torch.no_grad()
def update_ema(ema, model, decay):
    for target, source in zip(ema.parameters(), model.parameters()):
        target.lerp_(source, 1 - decay)
    for target, source in zip(ema.buffers(), model.buffers()):
        target.copy_(source)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count changed; exact resume cannot restore all RNG states")
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def atomic_save(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def backup_file(source, destination):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, tmp)
    os.replace(tmp, destination)


def load_checkpoint(path, device="cpu"):
    # Full recovery includes Python/NumPy RNG state; weights_only cannot deserialize it.
    return torch.load(path, map_location=device, weights_only=False)
