"""Portable paths, provenance, atomic writes and hardware disclosure."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import torch
import yaml


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def read_config(path):
    config = yaml.safe_load(Path(path).read_text())
    t = config["train"]
    if not 0 <= t["decay_start"] < t["max_steps"]:
        raise ValueError("Require 0 <= decay_start < max_steps")
    if config["image_size"] % 4 or config["image_size"] < 32:
        raise ValueError("image_size must be a multiple of 4 and >=32")
    if config["load_size"] < config["image_size"]:
        raise ValueError("load_size must be >= image_size")
    if not 0 <= t["ema_decay"] < 1 or t["batch_size"] < 1:
        raise ValueError("Invalid EMA decay or batch size")
    return config


def seed_all(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def device_for(value="auto"):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in this kernel. Select the lab's CUDA PyTorch environment.")
    return device


def source_hash():
    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for p in sorted(root.rglob("*.py")):
        h.update(str(p.relative_to(root)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def manifest(config, code_version):
    versions = {}
    for name in ["torch", "torchvision", "numpy", "Pillow", "scipy", "PyYAML",
                 "matplotlib", "torch-fidelity", "lpips"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    gpu = []
    if torch.cuda.is_available():
        gpu = [{"name": torch.cuda.get_device_name(i),
                "memory_bytes": torch.cuda.get_device_properties(i).total_memory}
               for i in range(torch.cuda.device_count())]
    cpu = platform.processor() or platform.machine()
    if Path("/proc/cpuinfo").exists():
        cpu = next((s.split(":", 1)[1].strip() for s in Path("/proc/cpuinfo").read_text().splitlines()
                    if s.startswith("model name")), cpu)
    return {"created_utc": utc_now(), "python": sys.version, "platform": platform.platform(),
            "cpu": cpu, "cpu_count": os.cpu_count(), "gpu": gpu,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "packages": versions, "config": config, "code_version": code_version,
            "source_sha256": source_hash(), "domain_A": "Monet", "domain_B": "Photo"}


class Tee:
    def __init__(self, stream, file):
        self.stream, self.file = stream, file

    def write(self, text):
        self.stream.write(text)
        self.file.write(text)
        self.file.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()


def save_environment(path):
    result = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                            capture_output=True, text=True)
    Path(path).write_text(result.stdout if result.returncode == 0 else result.stderr)
