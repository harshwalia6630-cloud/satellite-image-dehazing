"""Build the clean/hazy satellite dataset from NWPU-RESISC45.

* Uses the official RESISC45 train / validation / test splits (no image is in two splits).
* Drops the "cloud" scene class: its clean images are clouds, so there is nothing to dehaze.
* Takes a class-balanced subset: 150 / 20 / 25 images per class (6,600 / 880 / 1,100).
* Training haze is generated on the fly during training (fresh haze every epoch).
  Validation gets one fixed haze sample per image (levels rotate), and every test image
  gets all three levels (thin / moderate / thick), so test = 3,300 fixed pairs.

Outputs data/processed/*.npz (uint8 arrays).
"""

import io
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dehaze.haze import LEVELS, apply_haze, sample_params  # noqa: E402

RAW, OUT = ROOT / "data/raw", ROOT / "data/processed"
URL = "https://huggingface.co/datasets/timm/resisc45/resolve/main/data/{}-00000-of-00001.parquet"
PER_CLASS = {"train": 150, "validation": 20, "test": 25}
CLOUD_LABEL = 9


def load_split(split, seed):
    path = RAW / f"resisc45_{split}.parquet"
    if not path.exists():
        print(f"Downloading {split}")
        urllib.request.urlretrieve(URL.format(split), path)
    df = pd.read_parquet(path)
    df = df[df.label != CLOUD_LABEL]
    df = df.sample(frac=1, random_state=seed).groupby("label").head(PER_CLASS[split])
    df = df.sort_values("label", kind="stable").reset_index(drop=True)
    images = np.stack([np.asarray(Image.open(io.BytesIO(b["bytes"])).convert("RGB").resize((256, 256)))
                       for b in tqdm(df.image, desc=split)])
    return images, df.label.to_numpy()


def hazy_versions(clean, levels, seed):
    rng = np.random.default_rng(seed)
    hazy, betas = [], []
    for img, level in zip(clean, levels):
        p = sample_params(256, 256, rng, LEVELS[level])
        h, _ = apply_haze(img.astype(np.float32) / 255, p)
        hazy.append(np.round(h * 255).astype(np.uint8))
        betas.append(p.beta)
    return np.stack(hazy), np.array(betas, np.float32)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    names = list(LEVELS)
    stats = {}
    for i, split in enumerate(("train", "validation", "test")):
        clean, labels = load_split(split, seed=i)
        if split == "train":
            np.savez(OUT / "train.npz", clean=clean, labels=labels)
        elif split == "validation":
            levels = np.array([names[k % 3] for k in range(len(clean))])
            hazy, betas = hazy_versions(clean, levels, seed=100)
            np.savez(OUT / "val.npz", clean=clean, hazy=hazy, labels=labels, levels=levels, betas=betas)
        else:
            reps = np.repeat(np.arange(len(clean)), 3)
            levels = np.array(names * len(clean))
            hazy, betas = hazy_versions(clean[reps], levels, seed=200)
            np.savez(OUT / "test.npz", clean=clean[reps], hazy=hazy, labels=labels[reps], levels=levels,
                     betas=betas, image_index=reps)
        stats[split] = {"images": int(len(clean)), "classes": int(len(np.unique(labels)))}
        print(split, stats[split])
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
