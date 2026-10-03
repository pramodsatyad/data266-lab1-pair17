import os
import platform
import subprocess
from pathlib import Path

import torch
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


def make_scheduler(optimizer, warmup_steps, total_steps):
    warmup_steps = max(1, warmup_steps)

    def lr_factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    return LambdaLR(optimizer, lr_factor)


def save_checkpoint(path, model, optimizer, scheduler, epoch, step, best_f1, bad_epochs, cfg):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "step": step,
            "best_f1": best_f1,
            "bad_epochs": bad_epochs,
            "config": cfg,
        },
        tmp,
    )
    os.replace(tmp, path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt


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
