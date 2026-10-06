import json
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent


def load_config(name):
    with open(ROOT / "configs" / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def clean_story(text):
    if text is None:
        return None
    text = text.strip()
    if len(text) < 20:
        return None
    return text


def load_stories(cfg):
    d = cfg["data"]
    need = d["train_stories"] + d["val_stories"]
    ds = load_dataset(d["dataset"], split="train")
    rng = np.random.default_rng(cfg["seed"])
    order = rng.permutation(len(ds))
    stories = []
    start = 0
    while len(stories) < need and start < len(order):
        end = start + need
        picked = ds.select(order[start:end].tolist())["text"]
        for text in picked:
            text = clean_story(text)
            if text is not None:
                stories.append(text)
        start = end
    stories = stories[:need]
    if len(stories) < need:
        raise ValueError("not enough clean stories in the dataset")
    return stories[: d["train_stories"]], stories[d["train_stories"] :]


def build_vocab(text):
    chars = sorted(set(text))
    if len(chars) > 256:
        raise ValueError("vocab is too big for uint8")
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    return char_to_idx, idx_to_char


def encode(text, char_to_idx, chunk=5_000_000):
    codes = np.array([ord(c) for c in char_to_idx], dtype=np.uint32)
    ids = np.array(list(char_to_idx.values()), dtype=np.uint8)
    order = np.argsort(codes)
    codes, ids = codes[order], ids[order]
    out = np.empty(len(text), dtype=np.uint8)
    n = 0
    for start in range(0, len(text), chunk):
        part = np.frombuffer(text[start : start + chunk].encode("utf-32-le"), dtype=np.uint32)
        pos = np.searchsorted(codes, part)
        pos[pos == len(codes)] = 0
        found = ids[pos[codes[pos] == part]]
        out[n : n + len(found)] = found
        n += len(found)
    return out[:n].copy()


def decode(ids, idx_to_char):
    return "".join(idx_to_char[int(i)] for i in ids)


class CharDataset(Dataset):
    def __init__(self, ids, block_size):
        self.ids = ids
        self.block_size = block_size
        self.n = (len(ids) - 1) // block_size

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        start = i * self.block_size
        end = start + self.block_size
        x = torch.from_numpy(self.ids[start:end].astype(np.int64))
        y = torch.from_numpy(self.ids[start + 1 : end + 1].astype(np.int64))
        return x, y


def save_info(cfg, char_to_idx, train_stories, val_stories, train_ids, val_ids, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "vocab.json", "w") as f:
        json.dump({"char_to_idx": char_to_idx}, f, indent=2, ensure_ascii=False)
    info = {
        "seed": cfg["seed"],
        "dataset": cfg["data"]["dataset"],
        "train_stories": len(train_stories),
        "val_stories": len(val_stories),
        "train_chars": len(train_ids),
        "val_chars": len(val_ids),
        "vocab_size": len(char_to_idx),
        "separator": cfg["data"]["separator"],
        "block_size": cfg["model"]["block_size"],
    }
    with open(out_dir / "split_info.json", "w") as f:
        json.dump(info, f, indent=2)


def get_data(cfg):
    sep = cfg["data"]["separator"]
    block_size = cfg["model"]["block_size"]
    train_stories, val_stories = load_stories(cfg)
    train_text = sep.join(train_stories)
    val_text = sep.join(val_stories)
    char_to_idx, idx_to_char = build_vocab(train_text)
    train_ids = encode(train_text, char_to_idx)
    val_ids = encode(val_text, char_to_idx)
    save_info(cfg, char_to_idx, train_stories, val_stories, train_ids, val_ids,
              ROOT / cfg["data"]["processed_dir"])
    train_set = CharDataset(train_ids, block_size)
    val_set = CharDataset(val_ids, block_size)
    bs = cfg["train"]["batch_size"]
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=bs, shuffle=False)
    return train_loader, val_loader, char_to_idx, idx_to_char


if __name__ == "__main__":
    import sys

    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "local")
    train_loader, val_loader, char_to_idx, idx_to_char = get_data(cfg)
    print("vocab size:", len(char_to_idx))
    print("train sequences:", len(train_loader.dataset))
    print("val sequences:", len(val_loader.dataset))
    x, y = train_loader.dataset[0]
    print("input :", repr(decode(x[:60], idx_to_char)))
    print("target:", repr(decode(y[:60], idx_to_char)))
    print("shift ok:", bool((x[1:] == y[:-1]).all()))
