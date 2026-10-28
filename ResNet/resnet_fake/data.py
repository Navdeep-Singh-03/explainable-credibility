"""CSV-backed image dataset. CSV columns: id, image_path (relative to ``root``), is_fake (1 = fake)."""

from __future__ import annotations

import os
from typing import Callable, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

from .exceptions import InvalidImageError
from .preprocessing import load_image


class ImageCsvDataset(Dataset):
    def __init__(self, csv_path: str, root: str, transform: Optional[Callable] = None):
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"split file not found: {csv_path}")
        df = pd.read_csv(csv_path)
        for col in ("id", "image_path", "is_fake"):
            if col not in df.columns:
                raise ValueError(f"{csv_path} is missing required column {col!r}")
        missing = [p for p in df["image_path"] if not os.path.isfile(os.path.join(root, p))]
        if missing:
            raise FileNotFoundError(f"{len(missing)} image file(s) listed in {csv_path} are missing under {root!r}, e.g. {missing[0]!r}")
        self.df, self.root, self.transform = df.reset_index(drop=True), root, transform

    def __len__(self) -> int:
        return len(self.df)

    @property
    def labels(self):
        return [int(v) for v in self.df["is_fake"]]

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img, _ = load_image(os.path.join(self.root, row["image_path"]))
        x = self.transform(img) if self.transform else img
        return x, torch.tensor(float(row["is_fake"]))
