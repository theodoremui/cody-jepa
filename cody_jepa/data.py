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
from PIL import Image, ImageOps
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
    """Clips of `clip_length` grayscale frames, normalized to `image_size`.

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
        foreground_crop=True,
        foreground_threshold=8,
        foreground_margin=0.15,
        seed=0,
    ):
        self.split = split
        self.clip_length = int(clip_length)
        self.image_size = int(image_size)
        self.train = bool(train)
        self.windows = 1 if train else int(windows)
        self.crop_scale = tuple(crop_scale)
        self.flip_prob = float(flip_prob)
        self.foreground_crop = bool(foreground_crop)
        self.foreground_threshold = int(foreground_threshold)
        self.foreground_margin = float(foreground_margin)
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

    def _foreground_crop(self, paths):
        if not self.foreground_crop:
            return None
        lefts, tops, rights, bottoms = [], [], [], []
        for path in paths:
            with Image.open(path) as image:
                array = np.asarray(image.convert("L"))
            y_positions, x_positions = np.nonzero(array > self.foreground_threshold)
            if x_positions.size == 0:
                continue
            lefts.append(int(x_positions.min()))
            tops.append(int(y_positions.min()))
            rights.append(int(x_positions.max()) + 1)
            bottoms.append(int(y_positions.max()) + 1)
        if not lefts:
            return None

        left, top, right, bottom = min(lefts), min(tops), max(rights), max(bottoms)
        width, height = right - left, bottom - top
        margin_x = round(width * self.foreground_margin)
        margin_y = round(height * self.foreground_margin)
        with Image.open(paths[0]) as image:
            image_width, image_height = image.size
        return (
            max(0, left - margin_x),
            max(0, top - margin_y),
            min(image_width, right + margin_x),
            min(image_height, bottom + margin_y),
        )

    def _resize_frame(self, image):
        if bool(self.foreground_crop):
            return ImageOps.contain(
                image, (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
        return image.resize(
            (self.image_size, self.image_size), Image.Resampling.BILINEAR
        )

    def _load_frame(self, path, crop, flip):
        with Image.open(path) as image:
            image = image.convert("L")
            if crop is not None:
                image = image.crop(crop)
            if flip:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            image = self._resize_frame(image)
            canvas = Image.new("L", (self.image_size, self.image_size), 0)
            left = (self.image_size - image.width) // 2
            top = (self.image_size - image.height) // 2
            canvas.paste(image, (left, top))
            image = canvas
            array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array)

    def __getitem__(self, index):
        sample_index, window_index = divmod(index, self.windows)
        sample = self.samples[sample_index]
        # String seeding is stable across processes; tuple seeding relies on
        # hash() and is deprecated.
        rng = random.Random(f"{self.seed}-{self.epoch}-{sample_index}")
        start = self._window_start(sample, window_index, rng)

        frames = sample["frames"][start : start + self.clip_length]
        crop, flip = self._foreground_crop(frames), False
        if self.train:
            with Image.open(frames[0]) as probe:
                width, height = probe.size
            scale = rng.uniform(*self.crop_scale)
            if crop is None:
                crop_width, crop_height = round(width * scale), round(height * scale)
                left = rng.randrange(width - crop_width + 1)
                top = rng.randrange(height - crop_height + 1)
                crop = (left, top, left + crop_width, top + crop_height)
            elif scale != 1.0:
                left, top, right, bottom = crop
                crop_width, crop_height = right - left, bottom - top
                scaled_width = max(1, round(crop_width * scale))
                scaled_height = max(1, round(crop_height * scale))
                center_x = (left + right) / 2
                center_y = (top + bottom) / 2
                left = round(center_x - scaled_width / 2)
                top = round(center_y - scaled_height / 2)
                crop = (
                    max(0, left),
                    max(0, top),
                    min(width, left + scaled_width),
                    min(height, top + scaled_height),
                )
            flip = rng.random() < self.flip_prob

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
    """Return (train_loader, tune_loader) for a resolved training config."""
    shared = dict(
        manifest_csv=manifest_csv,
        root=root,
        clip_length=int(config["num_frames"]),
        image_size=int(config["img_size"]),
        foreground_crop=bool(config.get("foreground_crop", True)),
        foreground_threshold=int(config.get("foreground_threshold", 8)),
        foreground_margin=float(config.get("foreground_margin", 0.15)),
        seed=int(config.get("seed", 0)),
    )
    train_set = HealthGaitDataset(
        split="train",
        train=True,
        crop_scale=tuple(config.get("train_crop_scale", (1.0, 1.0))),
        flip_prob=float(config.get("train_horizontal_flip_prob", 0.0)),
        **shared,
    )
    tune_set = HealthGaitDataset(
        split=config.get("tune_split", "tune"), windows=windows, **shared
    )
    loader_options = dict(
        batch_size=int(config["batch_size"]),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(
        train_set, shuffle=True, drop_last=True, **loader_options
    )
    tune_loader = DataLoader(tune_set, shuffle=False, drop_last=False, **loader_options)
    return train_loader, tune_loader
