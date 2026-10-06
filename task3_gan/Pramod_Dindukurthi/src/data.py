"""Data audit and stateless, reproducible unpaired sampling.

Preparation never uses a hidden test set. Supply only permitted development data.
Exact decoded-pixel duplicates are grouped before splitting; near duplicates require review.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset

from common import json_write, sha256


def rgb(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def transform(image, size, load_size=None, rng=None):
    side = load_size if rng else size
    image = image.resize((side, side), Image.Resampling.BICUBIC)
    if rng:
        x, y = rng.randint(0, side-size), rng.randint(0, side-size)
        image = image.crop((x, y, x+size, y+size))
        if rng.random() < 0.5:
            image = ImageOps.mirror(image)
    a = np.array(image, dtype=np.float32).transpose(2, 0, 1)
    return torch.from_numpy(a / 127.5 - 1.0)


def to_image(tensor):
    a = ((tensor.detach().cpu().float().clamp(-1, 1) + 1) * 127.5).round()
    return Image.fromarray(a.to(torch.uint8).permute(1, 2, 0).numpy())


def prepare(root, output, seed=42, val_fraction=0.15):
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError(f"Split manifest already exists: {output}. Reuse it or choose a new path.")
    if not 0 < val_fraction < 0.5:
        raise ValueError("val_fraction must lie between 0 and 0.5")
    result = {"seed": seed, "val_fraction": val_fraction, "domains": {},
              "preprocessing": "EXIF orientation -> RGB -> bicubic square resize; no pairing"}
    cross_domain = set()
    for domain, folder in [("A", "monet_jpg"), ("B", "photo_jpg")]:
        files = sorted(p for p in (root / folder).rglob("*")
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        unique, seen, duplicates, invalid, sizes = [], {}, [], [], {}
        hashes = {}
        for p in files:
            rel = p.relative_to(root).as_posix()
            try:
                im = rgb(p)
                digest = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
                if digest in cross_domain:
                    raise ValueError(f"Identical image occurs in both domains: {rel}")
                if digest in seen:
                    duplicates.append({"excluded": rel, "retained": seen[digest]})
                    continue
                seen[digest] = rel
                unique.append(rel)
                hashes[rel] = sha256(p)
                key = f"{im.width}x{im.height}"
                sizes[key] = sizes.get(key, 0) + 1
            except (OSError, SyntaxError) as e:
                invalid.append({"file": rel, "error": str(e)})
        if invalid:
            raise ValueError(f"Corrupt images in {folder}; resolve before training: {invalid[:10]}")
        if len(unique) < 8:
            raise ValueError(f"Need >=8 valid unique images in {root / folder}; found {len(unique)}")
        cross_domain.update(seen)
        random.Random(seed + (0 if domain == "A" else 1)).shuffle(unique)
        nval = min(len(unique)-2, max(2, round(len(unique)*val_fraction)))
        result["domains"][domain] = {"train": unique[nval:], "val": unique[:nval],
                                       "sha256": hashes, "duplicates": duplicates,
                                       "sizes": sizes, "total_unique": len(unique)}
    json_write(output, result)
    print(json.dumps({d: {"train": len(v["train"]), "val": len(v["val"]),
                           "duplicates_excluded": len(v["duplicates"])}
                      for d, v in result["domains"].items()}, indent=2))
    return result


def read_splits(path, root=None, verify=False):
    splits = json.loads(Path(path).read_text())
    for domain in ["A", "B"]:
        part = splits["domains"][domain]
        if set(part["train"]) & set(part["val"]):
            raise ValueError("Training/validation overlap")
        if verify:
            for rel in part["train"] + part["val"]:
                if sha256(Path(root) / rel) != part["sha256"][rel]:
                    raise ValueError(f"Dataset changed since preparation: {rel}")
    return splits


class UnpairedDataset(Dataset):
    def __init__(self, root, splits, size, load_size, seed):
        self.root, self.size, self.load_size, self.seed = Path(root), size, load_size, seed
        self.paths = {d: splits["domains"][d]["train"] for d in ["A", "B"]}
        self.epoch_size = max(map(len, self.paths.values()))
        self.cached_epoch, self.orders = None, {}

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, index):
        epoch, offset = divmod(index, self.epoch_size)
        if epoch != self.cached_epoch:
            self.orders = {}
            for i, d in enumerate(["A", "B"]):
                order = list(range(len(self.paths[d])))
                random.Random(self.seed + epoch*7919 + i*31).shuffle(order)
                self.orders[d] = order
            self.cached_epoch = epoch
        batch = {}
        for i, d in enumerate(["A", "B"]):
            path = self.paths[d][self.orders[d][offset % len(self.paths[d])]]
            rng = random.Random(self.seed + index*104729 + i*997)
            batch[d] = transform(rgb(self.root/path), self.size, self.load_size, rng)
        return batch


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.15)
    a = p.parse_args()
    prepare(a.data_root, a.output, a.seed, a.val_fraction)
