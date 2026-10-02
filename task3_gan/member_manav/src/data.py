import random
from functools import partial
from pathlib import Path

import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent


def load_config(name):
    with open(ROOT / "configs" / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def list_images(folder, limit, seed):
    files = sorted(Path(folder).glob("*.jpg"))
    if limit is not None and limit < len(files):
        files = sorted(random.Random(seed).sample(files, limit))
    return files


def make_transforms(cfg):
    size, load = cfg["data"]["image_size"], cfg["data"]["load_size"]
    norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    train = transforms.Compose([
        transforms.Resize((load, load)),
        transforms.RandomCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        norm,
    ])
    plain = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        norm,
    ])
    return train, plain


class ImageDataset(Dataset):
    def __init__(self, files, transform):
        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return self.transform(Image.open(self.files[i]).convert("RGB"))


def pick_sample_ids(n_monet, n_photo, seed, count=4):
    rng = random.Random(seed)
    return {
        "monet": sorted(rng.sample(range(n_monet), min(count, n_monet))),
        "photo": sorted(rng.sample(range(n_photo), min(count, n_photo))),
    }


def load_samples(dataset, ids):
    return torch.stack([dataset[i] for i in ids])


def get_data(cfg):
    d = cfg["data"]
    data_dir = (ROOT / cfg["paths"]["data_dir"]).resolve()
    monet_files = list_images(data_dir / "monet_jpg", d["monet_limit"], cfg["seed"])
    photo_files = list_images(data_dir / "photo_jpg", d["photo_limit"], cfg["seed"])
    train_tf, plain_tf = make_transforms(cfg)
    sets = {
        "monet_train": ImageDataset(monet_files, train_tf),
        "photo_train": ImageDataset(photo_files, train_tf),
        "monet_plain": ImageDataset(monet_files, plain_tf),
        "photo_plain": ImageDataset(photo_files, plain_tf),
    }
    ids = pick_sample_ids(len(monet_files), len(photo_files), cfg["seed"])
    return sets, ids


def seed_worker(worker_id, base_seed):
    seed = base_seed + worker_id
    random.seed(seed)
    torch.manual_seed(seed)


def endless_loader(dataset, batch_size, seed, num_workers):
    sampler = RandomSampler(dataset, replacement=True, num_samples=10**12,
                            generator=torch.Generator().manual_seed(seed))
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                      num_workers=num_workers, drop_last=True,
                      worker_init_fn=partial(seed_worker, base_seed=seed))


def endless_pairs(cfg, sets):
    bs, workers = cfg["train"]["batch_size"], cfg["data"]["num_workers"]
    monet = endless_loader(sets["monet_train"], bs, cfg["seed"], workers)
    photo = endless_loader(sets["photo_train"], bs, cfg["seed"] + 1000, workers)
    for real_a, real_b in zip(monet, photo):
        yield real_a, real_b
