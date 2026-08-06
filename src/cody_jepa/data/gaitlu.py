"""Seekable, deterministic GaitLU-1M datasets backed by bit-packed tar records."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import random
import stat

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from .dataset import ManifestValidationError
from .frames import is_within


GAITLU_MANIFEST_VERSION = "gaitlu-indexed-bitpack-v2"
GAITLU_MANIFEST_COLUMNS = (
    "sequence_id",
    "source_group",
    "shard_path",
    "record_offset",
    "record_size",
    "num_frames",
    "height",
    "width",
    "split",
)
GAITLU_SEED_SCHEME = "splitmix64-v1"

_UINT64_MASK = (1 << 64) - 1
_SEED_COORDINATE_CONSTANTS = (
    0x9E3779B97F4A7C15,
    0xD1B54A32D192ED03,
    0x94D049BB133111EB,
    0xDB4F0B9175AE2165,
    0xBBE0563303A4615F,
)
_SAMPLER_STREAM = 1
_TRAIN_WINDOW_STREAM = 2
_TRAIN_SPATIAL_STREAM = 3
_VAL_SPATIAL_STREAM = 4
_MANIFEST_PAIR_DOMAIN = b"cody-jepa-gaitlu-manifest-pair-v1"
_HASH_CHUNK_SIZE = 1024 * 1024


def _gaitlu_seed(base_seed, epoch, sample_index, draw_index, stream):
    """Mix integer GaitLU coordinates into a deterministic unsigned 64-bit seed."""

    mixed = 0
    coordinates = (base_seed, epoch, sample_index, draw_index, stream)
    for coordinate, constant in zip(coordinates, _SEED_COORDINATE_CONSTANTS, strict=True):
        mixed ^= (int(coordinate) & _UINT64_MASK) * constant
    mixed &= _UINT64_MASK
    mixed = ((mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    mixed = ((mixed ^ (mixed >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return (mixed ^ (mixed >> 31)) & _UINT64_MASK


def gaitlu_manifest_pair_sha256(train_manifest_csv, val_manifest_csv):
    """Stream a role-sensitive digest over a GaitLU train/validation manifest pair."""

    digest = hashlib.sha256()
    digest.update(_MANIFEST_PAIR_DOMAIN)
    for role, manifest_csv in ((b"train", train_manifest_csv), (b"val", val_manifest_csv)):
        path = Path(manifest_csv).expanduser().resolve()
        length = path.stat().st_size
        digest.update(role)
        digest.update(length.to_bytes(8, "big"))
        bytes_read = 0
        with path.open("rb") as handle:
            while chunk := handle.read(_HASH_CHUNK_SIZE):
                digest.update(chunk)
                bytes_read += len(chunk)
        if bytes_read != length:
            raise OSError(f"GaitLU manifest changed while hashing: {path}")
    return digest.hexdigest()


class FixedExposureSampler(Sampler):
    """Draw a fixed number of examples per virtual epoch with replacement.

    The yielded draw index becomes part of the clip augmentation seed. Repeated
    selections of one short-pool sequence therefore receive different temporal
    and spatial transforms within the same virtual epoch.
    """

    def __init__(self, dataset: Dataset, num_samples: int, seed: int):
        self.dataset = dataset
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        if len(dataset) <= 0:
            raise ValueError("fixed-exposure dataset must be nonempty")
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        epoch = int(getattr(self.dataset, "epoch", 0))
        rng = random.Random(
            _gaitlu_seed(self.seed, epoch, 0, 0, _SAMPLER_STREAM)
        )
        return iter(
            (rng.randrange(len(self.dataset)), draw_index)
            for draw_index in range(self.num_samples)
        )


class GaitLUIndexedDataset(Dataset):
    """Random-access GaitLU sequences stored inside uncompressed tar shards."""

    def __init__(
        self,
        manifest_csv,
        *,
        data_root,
        split,
        clip_length=16,
        image_size=(112, 112),
        random_windows=False,
        base_seed=0,
        crop_scale=(1.0, 1.0),
        horizontal_flip_prob=0.0,
        deterministic_windows=1,
    ):
        self.manifest_csv = Path(manifest_csv).expanduser().resolve()
        self.data_root = Path(data_root).expanduser().resolve()
        self.split = str(split)
        self.clip_length = int(clip_length)
        self.image_size = tuple(int(value) for value in image_size)
        self.random_windows = bool(random_windows)
        self.base_seed = int(base_seed)
        self.crop_scale = tuple(float(value) for value in crop_scale)
        self.horizontal_flip_prob = float(horizontal_flip_prob)
        self.deterministic_windows = int(deterministic_windows)
        self.epoch = 0
        self._handles: dict[Path, object] = {}
        self._validate_options()
        self.samples = self._load_manifest()

    def _validate_options(self):
        if not self.data_root.is_dir():
            raise ValueError(f"GaitLU data_root is not a directory: {self.data_root}")
        if not self.manifest_csv.is_file():
            raise ManifestValidationError(
                f"GaitLU manifest does not exist: {self.manifest_csv}"
            )
        if self.split not in {"train", "val"}:
            raise ValueError("GaitLU split must be 'train' or 'val'")
        if self.clip_length <= 0:
            raise ValueError("clip_length must be positive")
        if len(self.image_size) != 2 or any(value <= 0 for value in self.image_size):
            raise ValueError("image_size must contain positive (height, width)")
        if (
            len(self.crop_scale) != 2
            or not 0 < self.crop_scale[0] <= self.crop_scale[1] <= 1
        ):
            raise ValueError("crop_scale must be an ordered pair in (0, 1]")
        if not 0 <= self.horizontal_flip_prob <= 1:
            raise ValueError("horizontal_flip_prob must be in [0, 1]")
        if self.deterministic_windows <= 0:
            raise ValueError("deterministic_windows must be positive")
        if self.random_windows and self.deterministic_windows != 1:
            raise ValueError("random datasets must use deterministic_windows=1")

    @staticmethod
    def _positive_int(value, label, line_no):
        try:
            result = int(value)
        except (TypeError, ValueError) as error:
            raise ManifestValidationError(
                f"row {line_no}: {label} must be an integer"
            ) from error
        if result <= 0:
            raise ManifestValidationError(f"row {line_no}: {label} must be positive")
        return result

    def _load_manifest(self):
        samples = []
        seen_ids = {}
        shards = {}
        shard_paths = []
        with self.manifest_csv.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != GAITLU_MANIFEST_COLUMNS:
                raise ManifestValidationError(
                    "GaitLU manifest must have exactly these columns in order: "
                    + ",".join(GAITLU_MANIFEST_COLUMNS)
                )
            for line_no, row in enumerate(reader, start=2):
                sequence_id = row["sequence_id"]
                source_group = row["source_group"]
                if not sequence_id or sequence_id != sequence_id.strip():
                    raise ManifestValidationError(
                        f"row {line_no}: sequence_id must be nonempty without outer whitespace"
                    )
                if not source_group or source_group != source_group.strip():
                    raise ManifestValidationError(
                        f"row {line_no}: source_group must be nonempty without outer whitespace"
                    )
                canonical_id = sequence_id.casefold()
                if canonical_id in seen_ids:
                    raise ManifestValidationError(
                        f"row {line_no}: duplicate/case-colliding sequence_id {sequence_id!r}"
                    )
                seen_ids[canonical_id] = line_no
                if row["split"] != self.split:
                    raise ManifestValidationError(
                        f"row {line_no}: expected split {self.split!r}, got {row['split']!r}"
                    )
                relative_shard = Path(row["shard_path"])
                if relative_shard.is_absolute() or ".." in relative_shard.parts:
                    raise ManifestValidationError(
                        f"row {line_no}: shard_path must be a safe relative path"
                    )
                relative_shard_key = relative_shard.as_posix()
                cached_shard = shards.get(relative_shard_key)
                if cached_shard is None:
                    shard = (self.data_root / relative_shard).resolve()
                    if not is_within(shard, self.data_root):
                        raise ManifestValidationError(
                            f"row {line_no}: shard is missing or escapes data_root: {shard}"
                        )
                    try:
                        shard_stat = shard.stat()
                    except OSError as error:
                        raise ManifestValidationError(
                            f"row {line_no}: shard is missing or escapes data_root: {shard}"
                        ) from error
                    if not stat.S_ISREG(shard_stat.st_mode):
                        raise ManifestValidationError(
                            f"row {line_no}: shard is missing or escapes data_root: {shard}"
                        )
                    shard_size = shard_stat.st_size
                    shard_index = len(shard_paths)
                    shard_paths.append(shard)
                    cached_shard = (shard_index, shard_size)
                    shards[relative_shard_key] = cached_shard
                shard_index, shard_size = cached_shard
                if shard_paths[shard_index].suffix != ".tar":
                    raise ManifestValidationError(
                        f"row {line_no}: prepared shards must be uncompressed .tar files"
                    )
                offset = self._positive_int(row["record_offset"], "record_offset", line_no)
                size = self._positive_int(row["record_size"], "record_size", line_no)
                num_frames = self._positive_int(row["num_frames"], "num_frames", line_no)
                height = self._positive_int(row["height"], "height", line_no)
                width = self._positive_int(row["width"], "width", line_no)
                expected_size = math.ceil(num_frames * height * width / 8)
                if size != expected_size:
                    raise ManifestValidationError(
                        f"row {line_no}: record_size={size}, expected {expected_size}"
                    )
                if offset + size > shard_size:
                    raise ManifestValidationError(
                        f"row {line_no}: record extends beyond shard {shard_paths[shard_index]}"
                    )
                if num_frames < self.clip_length:
                    raise ManifestValidationError(
                        f"row {line_no}: {num_frames} frames is shorter than clip_length"
                    )
                available_windows = num_frames - self.clip_length + 1
                if not self.random_windows and self.deterministic_windows > available_windows:
                    raise ManifestValidationError(
                        f"row {line_no}: only {available_windows} distinct windows are available"
                    )
                samples.append(
                    {
                        "sequence_id": sequence_id,
                        "source_group": source_group,
                        "shard_index": shard_index,
                        "record_offset": offset,
                        "record_size": size,
                        "num_frames": num_frames,
                        "height": height,
                        "width": width,
                    }
                )
        if not samples:
            raise ManifestValidationError("GaitLU manifest contains no samples")
        self._shard_paths = tuple(shard_paths)
        return samples

    def _open_shard(self, path):
        handle = self._handles.get(path)
        if handle is None or handle.closed:
            handle = path.open("rb", buffering=0)
            self._handles[path] = handle
        return handle

    def _read_packed(self, sample):
        handle = self._open_shard(self._shard_paths[sample["shard_index"]])
        payload = os.pread(
            handle.fileno(), sample["record_size"], sample["record_offset"]
        )
        if len(payload) != sample["record_size"]:
            raise ManifestValidationError(
                f"short read for GaitLU sequence {sample['sequence_id']!r}"
            )
        return payload

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_handles"] = {}
        return state

    def close(self):
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __del__(self):
        self.close()

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _decode(self, sample):
        payload = self._read_packed(sample)
        count = sample["num_frames"] * sample["height"] * sample["width"]
        bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="little", count=count)
        return bits.reshape(sample["num_frames"], sample["height"], sample["width"])

    def _window_start(self, sample_index, sample, draw_index, window_index):
        count = sample["num_frames"] - self.clip_length + 1
        if self.random_windows:
            seed = _gaitlu_seed(
                self.base_seed,
                self.epoch,
                sample_index,
                draw_index,
                _TRAIN_WINDOW_STREAM,
            )
            return random.Random(seed).randrange(count)
        if self.deterministic_windows == 1:
            return (count - 1) // 2
        return round(window_index * (count - 1) / (self.deterministic_windows - 1))

    def _transform(self, video, sample_index, draw_index):
        stream = _TRAIN_SPATIAL_STREAM if self.split == "train" else _VAL_SPATIAL_STREAM
        rng = random.Random(
            _gaitlu_seed(
                self.base_seed, self.epoch, sample_index, draw_index, stream
            )
        )
        scale = rng.uniform(*self.crop_scale)
        _, _, height, width = video.shape
        crop_height = max(1, round(height * scale))
        crop_width = max(1, round(width * scale))
        top = round(rng.random() * (height - crop_height))
        left = round(rng.random() * (width - crop_width))
        video = video[:, :, top : top + crop_height, left : left + crop_width]
        if rng.random() < self.horizontal_flip_prob:
            video = video.flip(-1)
        return F.interpolate(video, size=self.image_size, mode="bilinear", align_corners=False)

    def __len__(self):
        return len(self.samples) * self.deterministic_windows

    def subject_id_at(self, idx):
        if isinstance(idx, tuple):
            idx = idx[0]
        sample_index, _ = divmod(int(idx), self.deterministic_windows)
        return self.samples[sample_index]["source_group"]

    def __getitem__(self, idx):
        draw_index = 0
        if isinstance(idx, tuple):
            idx, draw_index = idx
        sample_index, window_index = divmod(int(idx), self.deterministic_windows)
        sample = self.samples[sample_index]
        start = self._window_start(sample_index, sample, int(draw_index), window_index)
        frames = self._decode(sample)[start : start + self.clip_length]
        video = torch.from_numpy(frames.astype(np.float32, copy=False)).unsqueeze(1)
        video = self._transform(video, sample_index, int(draw_index))
        return {
            "video": video,
            "split": self.split,
            "sequence_id": sample["sequence_id"],
            "subject_id": sample["source_group"],
            "source_group": sample["source_group"],
            "window_start": start,
            "window_index": window_index,
            "num_frames": sample["num_frames"],
        }

    def description(self):
        return {
            "version": GAITLU_MANIFEST_VERSION,
            "manifest_name": self.manifest_csv.name,
            "clip_length": self.clip_length,
            "image_size": list(self.image_size),
            "base_seed": self.base_seed,
            "random_windows": self.random_windows,
            "crop_scale": list(self.crop_scale),
            "horizontal_flip_prob": self.horizontal_flip_prob,
            "deterministic_windows": self.deterministic_windows,
            "split": self.split,
            "sequence_count": len(self.samples),
            "sample_count": len(self),
        }


@dataclass(frozen=True)
class GaitLULoaderConfig:
    train_manifest_csv: Path
    val_manifest_csv: Path
    data_root: Path
    clip_length: int = 16
    image_size: tuple[int, int] = (112, 112)
    seed: int = 0
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    train_crop_scale: tuple[float, float] = (0.9, 1.0)
    train_horizontal_flip_prob: float = 0.0
    eval_windows: int = 1
    epoch_examples: int = 65_536

    def __post_init__(self):
        root = Path(self.data_root).expanduser().resolve()
        object.__setattr__(self, "data_root", root)
        for field in ("train_manifest_csv", "val_manifest_csv"):
            path = Path(getattr(self, field)).expanduser()
            path = path.resolve() if path.is_absolute() else (root / path).resolve()
            object.__setattr__(self, field, path)
        object.__setattr__(self, "clip_length", int(self.clip_length))
        object.__setattr__(self, "image_size", tuple(map(int, self.image_size)))
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "batch_size", int(self.batch_size))
        object.__setattr__(self, "num_workers", int(self.num_workers))
        object.__setattr__(self, "prefetch_factor", int(self.prefetch_factor))
        object.__setattr__(self, "eval_windows", int(self.eval_windows))
        object.__setattr__(self, "epoch_examples", int(self.epoch_examples))
        if self.batch_size <= 0 or self.num_workers < 0 or self.prefetch_factor <= 0:
            raise ValueError("invalid GaitLU DataLoader size/worker settings")
        if self.epoch_examples <= 0 or self.epoch_examples % self.batch_size:
            raise ValueError("epoch_examples must be positive and divisible by batch_size")
        if self.eval_windows <= 0:
            raise ValueError("eval_windows must be positive")

    def as_dict(self):
        return {
            "clip_length": self.clip_length,
            "image_size": list(self.image_size),
            "seed": self.seed,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "prefetch_factor": self.prefetch_factor,
            "train_crop_scale": list(self.train_crop_scale),
            "train_horizontal_flip_prob": self.train_horizontal_flip_prob,
            "eval_windows": self.eval_windows,
            "epoch_examples": self.epoch_examples,
        }


def build_gaitlu_datasets_from_config(config: GaitLULoaderConfig):
    train = GaitLUIndexedDataset(
        config.train_manifest_csv,
        data_root=config.data_root,
        split="train",
        clip_length=config.clip_length,
        image_size=config.image_size,
        random_windows=True,
        base_seed=config.seed,
        crop_scale=config.train_crop_scale,
        horizontal_flip_prob=config.train_horizontal_flip_prob,
    )
    val = GaitLUIndexedDataset(
        config.val_manifest_csv,
        data_root=config.data_root,
        split="val",
        clip_length=config.clip_length,
        image_size=config.image_size,
        random_windows=False,
        base_seed=config.seed,
        deterministic_windows=config.eval_windows,
    )
    return train, val


def build_gaitlu_loaders_from_config(config: GaitLULoaderConfig, datasets=None):
    train, val = build_gaitlu_datasets_from_config(config) if datasets is None else datasets
    generator = torch.Generator().manual_seed(config.seed)
    worker_options = (
        {"prefetch_factor": config.prefetch_factor} if config.num_workers > 0 else {}
    )
    sampler = FixedExposureSampler(train, config.epoch_examples, config.seed)
    train_loader = DataLoader(
        train,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        generator=generator,
        drop_last=True,
        persistent_workers=False,
        **worker_options,
    )
    val_loader = DataLoader(
        val,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=False,
        **worker_options,
    )
    return train_loader, val_loader


__all__ = [
    "FixedExposureSampler",
    "GAITLU_MANIFEST_COLUMNS",
    "GAITLU_MANIFEST_VERSION",
    "GAITLU_SEED_SCHEME",
    "GaitLUIndexedDataset",
    "GaitLULoaderConfig",
    "build_gaitlu_datasets_from_config",
    "build_gaitlu_loaders_from_config",
    "gaitlu_manifest_pair_sha256",
]
