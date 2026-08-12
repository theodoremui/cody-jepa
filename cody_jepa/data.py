"""Health&Gait clip loading.

The manifest is a CSV of recordings; one row is one video of one participant.
Frame paths are listed once at construction so `__getitem__` only touches the
frames it actually decodes. Nothing here writes to disk.
"""

import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


MANIFEST_COLUMNS = (
    "subject_id",
    "split",
    "frame_dir",
    "num_frames",
    "gait_system",
    "speed",
    "clothing",
    "direction",
    "recording_id",
    "source_video_id",
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def list_frame_paths(frame_dir):
    """Frame files in numeric order when named numerically, else lexicographic."""
    paths = [
        path
        for path in Path(frame_dir).iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if all(path.stem.isdigit() for path in paths):
        return sorted(paths, key=lambda path: int(path.stem))
    return sorted(paths)


def read_manifest(manifest_csv, root):
    """Read manifest rows, resolving frame_dir against `root`."""
    manifest_csv = Path(manifest_csv)
    with manifest_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"manifest is empty: {manifest_csv}")
    missing = sorted(set(MANIFEST_COLUMNS) - set(rows[0]))
    if missing:
        raise ValueError(f"manifest {manifest_csv} is missing columns: {missing}")
    for row in rows:
        row["frame_dir"] = Path(root) / row["frame_dir"]
        row["num_frames"] = int(row["num_frames"])
    return rows


class HealthGaitDataset(Dataset):
    """Clips of `clip_length` grayscale frames, resized to `image_size`.

    Training draws one random window per recording per epoch with optional
    crop/flip augmentation. Evaluation draws `windows` evenly spaced windows
    with no augmentation, so repeated runs give identical features.
    """

    def __init__(
        self,
        manifest_csv,
        split,
        root=".",
        clip_length=16,
        image_size=112,
        train=False,
        windows=1,
        crop_scale=(1.0, 1.0),
        flip_prob=0.0,
        seed=0,
    ):
        self.split = split
        self.clip_length = int(clip_length)
        self.image_size = int(image_size)
        self.train = bool(train)
        self.windows = 1 if train else int(windows)
        self.crop_scale = tuple(crop_scale)
        self.flip_prob = float(flip_prob)
        self.seed = int(seed)
        self.epoch = 0

        self.samples = []
        for row in read_manifest(manifest_csv, root):
            if row["split"] != split:
                continue
            frames = list_frame_paths(row["frame_dir"])
            if len(frames) < self.clip_length:
                continue
            self.samples.append({**row, "frames": frames})
        if not self.samples:
            raise ValueError(f"no usable recordings for split {split!r}")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.samples) * self.windows

    def subject_id_at(self, index):
        return self.samples[index // self.windows]["subject_id"]

    def _window_start(self, sample, window_index, rng):
        last_start = len(sample["frames"]) - self.clip_length
        if self.train:
            return rng.randrange(last_start + 1)
        if self.windows == 1:
            return last_start // 2
        return round(window_index * last_start / (self.windows - 1))

    def _load_frame(self, path, crop, flip):
        with Image.open(path) as image:
            image = image.convert("L")
            if crop is not None:
                image = image.crop(crop)
            if flip:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            image = image.resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
            array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array)

    def __getitem__(self, index):
        sample_index, window_index = divmod(index, self.windows)
        sample = self.samples[sample_index]
        # String seeding is stable across processes; tuple seeding relies on
        # hash() and is deprecated.
        rng = random.Random(f"{self.seed}-{self.epoch}-{sample_index}")
        start = self._window_start(sample, window_index, rng)

        crop, flip = None, False
        if self.train:
            with Image.open(sample["frames"][start]) as probe:
                width, height = probe.size
            scale = rng.uniform(*self.crop_scale)
            crop_width, crop_height = round(width * scale), round(height * scale)
            left = rng.randrange(width - crop_width + 1)
            top = rng.randrange(height - crop_height + 1)
            crop = (left, top, left + crop_width, top + crop_height)
            flip = rng.random() < self.flip_prob

        frames = sample["frames"][start : start + self.clip_length]
        video = torch.stack([self._load_frame(path, crop, flip) for path in frames])
        return {
            "video": video.unsqueeze(1),  # [T, 1, H, W]
            "subject_id": sample["subject_id"],
            "split": sample["split"],
            "gait_system": sample["gait_system"],
            "speed": sample["speed"],
            "clothing": sample["clothing"],
            "direction": sample["direction"],
            "recording_id": sample["recording_id"],
            "source_video_id": sample["source_video_id"],
            "window_start": start,
        }


def build_loaders(config, manifest_csv, root=".", num_workers=4, windows=1):
    """Return (train_loader, val_loader) for a resolved training config."""
    shared = dict(
        manifest_csv=manifest_csv,
        root=root,
        clip_length=int(config["num_frames"]),
        image_size=int(config["img_size"]),
        seed=int(config.get("seed", 0)),
    )
    train_set = HealthGaitDataset(
        split="train",
        train=True,
        crop_scale=tuple(config.get("train_crop_scale", (1.0, 1.0))),
        flip_prob=float(config.get("train_horizontal_flip_prob", 0.0)),
        **shared,
    )
    val_set = HealthGaitDataset(split="val", windows=windows, **shared)
    loader_options = dict(
        batch_size=int(config["batch_size"]),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(
        train_set, shuffle=True, drop_last=True, **loader_options
    )
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_options)
    return train_loader, val_loader
