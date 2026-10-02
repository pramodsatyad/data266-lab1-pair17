import json
import re
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import disable_progress_bar, load_dataset
from nltk.stem import PorterStemmer
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
PAD, UNK = 0, 1

disable_progress_bar()

STOPWORDS = frozenset(
    """i me my myself we our ours ourselves you your yours yourself yourselves he him his
    himself she her hers herself it its itself they them their theirs themselves what which
    who whom this that these those am is are was were be been being have has had having do
    does did doing a an the and but if or because as until while of at by for with about
    against between into through during before after above below to from up down in out on
    off over under again further then once here there when where why how all any both each
    few more most other some such only own same so than too very s t can will just don
    should now d ll m re ve""".split()
)

URL = re.compile(r"https?://\S+|www\.\S+")
HTML = re.compile(r"<[^>]+>|&#?\w+;|\\[nrt]")
UNICODE_PAIR = re.compile(r"\\u(d[89ab][0-9a-f]{2})\\u(d[c-f][0-9a-f]{2})", re.I)
UNICODE_ESCAPE = re.compile(r"\\u[0-9a-fA-F]{4}")
ESCAPE = re.compile(r"\\u([0-9a-fA-F]{4})|\\([nrt\"'/\\])")
NOT_WORD = re.compile(r"[^a-z0-9\s]")
NOT_END = re.compile(r"n't\b")

stemmer = PorterStemmer()


@lru_cache(maxsize=None)
def stem(word):
    return stemmer.stem(word)


def load_config(name):
    with open(ROOT / "configs" / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def join_pair(m):
    high, low = int(m.group(1), 16), int(m.group(2), 16)
    return chr(0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00))


def decode_one(m):
    if m.group(1):
        code = int(m.group(1), 16)
        return " " if 0xD800 <= code <= 0xDFFF else chr(code)
    return " " if m.group(2) in "nrt" else m.group(2)


def decode_escapes(text):
    try:
        return ESCAPE.sub(decode_one, UNICODE_PAIR.sub(join_pair, text))
    except (ValueError, OverflowError):
        return text


def clean_text(text):
    text = text.lower().replace("’", "'")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = URL.sub(" ", text)
    text = HTML.sub(" ", text)
    text = text.replace("won't", "will not").replace("can't", "can not")
    text = text.replace("cannot", "can not").replace("ain't", "is not")
    text = NOT_END.sub(" not", text)
    text = NOT_WORD.sub(" ", text)
    words = [w for w in text.split() if w not in STOPWORDS]
    return " ".join(stem(w) for w in words)


def find_valid(texts, labels):
    dropped = {"not_text": 0, "empty": 0, "bad_label": 0, "escaped": 0, "unicode_escaped": 0}
    valid = []
    for i, (text, label) in enumerate(zip(texts, labels)):
        if not isinstance(text, str):
            dropped["not_text"] += 1
        elif not text.strip():
            dropped["empty"] += 1
        elif label not in (0, 1):
            dropped["bad_label"] += 1
        else:
            valid.append(i)
            dropped["escaped"] += bool(ESCAPE.search(text))
            dropped["unicode_escaped"] += bool(UNICODE_ESCAPE.search(text))
    dropped["total"] = len(texts)
    dropped["kept"] = len(valid)
    return np.array(valid), dropped


def split_train(d, perm):
    if d.get("val_samples") is not None:
        need = d["train_samples"] + d["val_samples"]
        if len(perm) < need:
            raise ValueError("not enough clean reviews for this config")
        pool = perm[:need]
        n_val = d["val_samples"]
    else:
        pool = perm if d["train_samples"] is None else perm[: d["train_samples"]]
        n_val = int(round(len(pool) * d["val_fraction"]))
    return pool[n_val:], pool[:n_val]


def build_vocab(cleaned, min_freq, max_vocab):
    counts = Counter()
    for s in cleaned:
        counts.update(s.split())
    words = [w for w, c in counts.items() if c >= min_freq]
    words.sort(key=lambda w: (-counts[w], w))
    words = words[: max_vocab - 2]
    word_to_idx = {"<pad>": PAD, "<unk>": UNK}
    for w in words:
        word_to_idx[w] = len(word_to_idx)
    if len(word_to_idx) > 65535:
        raise ValueError("vocab is too big for uint16")
    return word_to_idx


def encode(cleaned, word_to_idx, max_len):
    n = len(cleaned)
    x = np.zeros((n, max_len), dtype=np.uint16)
    n_tokens = np.zeros(n, dtype=np.uint16)
    for i, s in enumerate(cleaned):
        ids = [word_to_idx.get(w, UNK) for w in s.split()]
        n_tokens[i] = min(len(ids), 65535)
        ids = ids[:max_len]
        x[i, : len(ids)] = ids
    x[n_tokens == 0, 0] = UNK
    lengths = np.minimum(np.maximum(n_tokens, 1), max_len).astype(np.uint16)
    return x, lengths, n_tokens


def class_counts(y):
    return {"negative": int((y == 0).sum()), "positive": int((y == 1).sum())}


def build_data(cfg, out_dir):
    d = cfg["data"]
    rng = np.random.default_rng(cfg["seed"])

    ds = load_dataset(d["dataset"], split="train")
    train_texts, train_labels = ds["text"], ds["label"]
    train_valid, train_dropped = find_valid(train_texts, train_labels)
    train_idx, val_idx = split_train(d, rng.permutation(train_valid))

    ds = load_dataset(d["dataset"], split="test")
    test_texts, test_labels = ds["text"], ds["label"]
    test_valid, test_dropped = find_valid(test_texts, test_labels)
    test_idx = rng.permutation(test_valid)
    if d["test_samples"] is not None:
        test_idx = test_idx[: d["test_samples"]]

    parts = {}
    for name, texts, labels, idx in [
        ("train", train_texts, train_labels, train_idx),
        ("val", train_texts, train_labels, val_idx),
        ("test", test_texts, test_labels, test_idx),
    ]:
        decoded = [decode_escapes(texts[i]) for i in idx]
        parts[name] = {
            "text": decoded if name == "test" else None,
            "cleaned": [clean_text(t) for t in decoded],
            "y": np.array([labels[i] for i in idx], dtype=np.uint8),
            "words": np.array([len(t.split()) for t in decoded], dtype=np.uint16),
        }

    word_to_idx = build_vocab(parts["train"]["cleaned"], d["min_freq"], d["max_vocab"])

    arrays = {}
    info = {
        "seed": cfg["seed"],
        "dataset": d["dataset"],
        "data_cfg": d,
        "vocab_size": len(word_to_idx),
        "max_len": d["max_len"],
        "counts": {},
        "class_counts": {},
        "empty_after_cleaning": {},
        "cut_share": {},
        "dropped": {"train_split": train_dropped, "test_split": test_dropped},
    }
    for name, p in parts.items():
        x, lengths, n_tokens = encode(p["cleaned"], word_to_idx, d["max_len"])
        arrays[f"{name}_x"] = x
        arrays[f"{name}_len"] = lengths
        arrays[f"{name}_y"] = p["y"]
        arrays[f"{name}_words"] = p["words"]
        arrays[f"{name}_ntok"] = n_tokens
        info["counts"][name] = len(p["y"])
        info["class_counts"][name] = class_counts(p["y"])
        info["empty_after_cleaning"][name] = int((n_tokens == 0).sum())
        info["cut_share"][name] = float((n_tokens > d["max_len"]).mean())

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "arrays.npz", **arrays)
    with open(out_dir / "vocab.json", "w") as f:
        json.dump(word_to_idx, f)
    with open(out_dir / "split_info.json", "w") as f:
        json.dump(info, f, indent=2)
    test_raw = pd.DataFrame(
        {
            "orig_index": test_idx,
            "text": parts["test"]["text"],
            "label": parts["test"]["y"],
            "raw_words": parts["test"]["words"],
        }
    )
    test_raw.to_csv(out_dir / "test_raw.csv", index=False)


class ReviewDataset(Dataset):
    def __init__(self, x, lengths, y):
        self.x = x
        self.lengths = lengths
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.x[i].astype(np.int64)),
            int(self.lengths[i]),
            int(self.y[i]),
        )


def load_test_raw(cfg):
    return pd.read_csv(ROOT / cfg["paths"]["processed_dir"] / "test_raw.csv")


def get_data(cfg):
    out_dir = ROOT / cfg["paths"]["processed_dir"]
    info_path = out_dir / "split_info.json"
    cached = False
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        cached = info["data_cfg"] == cfg["data"] and info["seed"] == cfg["seed"]
    if not cached:
        build_data(cfg, out_dir)
        with open(info_path) as f:
            info = json.load(f)
    with open(out_dir / "vocab.json") as f:
        word_to_idx = json.load(f)
    arrays = np.load(out_dir / "arrays.npz")
    bs = cfg["train"]["batch_size"]
    loaders = []
    for name in ["train", "val", "test"]:
        dataset = ReviewDataset(arrays[f"{name}_x"], arrays[f"{name}_len"], arrays[f"{name}_y"])
        loaders.append(DataLoader(dataset, batch_size=bs, shuffle=(name == "train")))
    return loaders[0], loaders[1], loaders[2], word_to_idx, info
