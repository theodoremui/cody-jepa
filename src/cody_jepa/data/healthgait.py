from pathlib import Path
import csv
from dataclasses import dataclass, replace

import torch
from torch.utils.data import DataLoader

from .dataset import HealthGaitManifestDataset, VALID_SPLITS
from .healthgait_diagnostics import (
    run_healthgait_motion_diagnostics,
    summarize_healthgait_manifest,
    write_healthgait_dummy_probe_exports,
    write_healthgait_metadata_summary,
)


DEFAULT_BATCH_SIZE = 4
DEFAULT_CLIP_LENGTH = 16
DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_MANIFEST_SEED = 0
DEFAULT_MODALITY = "silhouette"
DEFAULT_BASE_SEED = 0
DEFAULT_CHANNELS = 1
DEFAULT_WINDOW_POLICY = "train_random_val_center"
VALID_WINDOW_POLICIES = frozenset({"train_random_val_center", "random", "center"})


@dataclass(frozen=True)
class HealthGaitLoaderConfig:
    manifest_csv: Path
    repo_root: Path
    split: str = "train"
    clip_length: int = DEFAULT_CLIP_LENGTH
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE
    channels: int = DEFAULT_CHANNELS
    seed: int = DEFAULT_BASE_SEED
    window_policy: str = DEFAULT_WINDOW_POLICY
    strict_validation: bool = True
    batch_size: int = DEFAULT_BATCH_SIZE
    num_workers: int = 0
    pin_memory: bool = False

    def __post_init__(self):
        object.__setattr__(self, "manifest_csv", Path(self.manifest_csv))
        object.__setattr__(self, "repo_root", Path(self.repo_root))
        object.__setattr__(self, "clip_length", int(self.clip_length))
        object.__setattr__(self, "image_size", tuple(int(value) for value in self.image_size))
        object.__setattr__(self, "channels", int(self.channels))
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "strict_validation", bool(self.strict_validation))
        object.__setattr__(self, "batch_size", int(self.batch_size))
        object.__setattr__(self, "num_workers", int(self.num_workers))
        object.__setattr__(self, "pin_memory", bool(self.pin_memory))

        if self.split not in VALID_SPLITS:
            valid = ", ".join(sorted(VALID_SPLITS))
            raise ValueError(f"split must be one of {{{valid}}}; got {self.split!r}")
        if self.clip_length <= 0:
            raise ValueError(f"clip_length must be positive; got {self.clip_length}")
        if len(self.image_size) != 2 or any(value <= 0 for value in self.image_size):
            raise ValueError(f"image_size must contain two positive integers; got {self.image_size}")
        if self.channels != DEFAULT_CHANNELS:
            raise ValueError("Health&Gait silhouette loader currently supports one grayscale channel")
        if self.window_policy not in VALID_WINDOW_POLICIES:
            valid = ", ".join(sorted(VALID_WINDOW_POLICIES))
            raise ValueError(
                f"window_policy must be one of {{{valid}}}; got {self.window_policy!r}"
            )
        if not self.strict_validation:
            raise ValueError("Health&Gait loader currently requires strict_validation=True")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive; got {self.batch_size}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be non-negative; got {self.num_workers}")

    def for_split(self, split):
        return replace(self, split=split)

    def uses_random_windows(self):
        if self.window_policy == "random":
            return True
        if self.window_policy == "center":
            return False
        return self.split == "train"

    def as_dict(self):
        return {
            "manifest_csv": str(self.manifest_csv),
            "repo_root": str(self.repo_root),
            "split": self.split,
            "clip_length": self.clip_length,
            "image_size": list(self.image_size),
            "channels": self.channels,
            "seed": self.seed,
            "window_policy": self.window_policy,
            "strict_validation": self.strict_validation,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }


def find_repo_root(start=None):
    """Find the repo root from the repo directory or a direct child directory."""
    cwd = Path.cwd() if start is None else Path(start).resolve()
    candidates = [cwd, *cwd.parents]

    for candidate in candidates:
        if (candidate / "data" / "healthgait").exists():
            return candidate

    raise FileNotFoundError("Could not find data/healthgait from the current directory")


def healthgait_manifest_path(
    repo_root=None,
    modality=DEFAULT_MODALITY,
    seed=DEFAULT_MANIFEST_SEED,
):
    """Return the default manifest path for a Health&Gait modality and split seed."""
    root = find_repo_root() if repo_root is None else Path(repo_root)
    filename = f"{modality}_subject_split_seed{seed}.csv"
    return root / "data" / "healthgait" / "manifests" / filename


def preview_manifest(manifest_csv, n=3):
    """Read the first n rows of a manifest without constructing a dataset."""
    manifest_csv = Path(manifest_csv)

    with manifest_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        return [row for _, row in zip(range(n), reader)]


def build_healthgait_datasets(
    manifest_csv=None,
    repo_root=None,
    clip_length=DEFAULT_CLIP_LENGTH,
    image_size=DEFAULT_IMAGE_SIZE,
    base_seed=DEFAULT_BASE_SEED,
):
    """Build train and validation Health&Gait datasets from a manifest."""
    root = find_repo_root() if repo_root is None else Path(repo_root)
    manifest = healthgait_manifest_path(root) if manifest_csv is None else Path(manifest_csv)

    train_ds = HealthGaitManifestDataset(
        manifest,
        split="train",
        repo_root=root,
        clip_length=clip_length,
        image_size=image_size,
        random_windows=True,
        base_seed=base_seed,
    )
    val_ds = HealthGaitManifestDataset(
        manifest,
        split="val",
        repo_root=root,
        clip_length=clip_length,
        image_size=image_size,
        random_windows=False,
        base_seed=base_seed,
    )

    return train_ds, val_ds


def build_healthgait_dataset_from_config(config):
    """Build one Health&Gait dataset from an explicit loader config."""
    return HealthGaitManifestDataset(
        config.manifest_csv,
        split=config.split,
        repo_root=config.repo_root,
        clip_length=config.clip_length,
        image_size=config.image_size,
        random_windows=config.uses_random_windows(),
        base_seed=config.seed,
    )


def build_healthgait_datasets_from_config(config):
    """Build train and validation Health&Gait datasets from one reproducible config."""
    train_ds = build_healthgait_dataset_from_config(config.for_split("train"))
    val_ds = build_healthgait_dataset_from_config(config.for_split("val"))
    return train_ds, val_ds


def build_healthgait_loaders(
    manifest_csv=None,
    repo_root=None,
    clip_length=DEFAULT_CLIP_LENGTH,
    image_size=DEFAULT_IMAGE_SIZE,
    batch_size=DEFAULT_BATCH_SIZE,
    num_workers=0,
    pin_memory=False,
    base_seed=DEFAULT_BASE_SEED,
):
    """Build train and validation DataLoaders with notebook-safe defaults."""
    train_ds, val_ds = build_healthgait_datasets(
        manifest_csv=manifest_csv,
        repo_root=repo_root,
        clip_length=clip_length,
        image_size=image_size,
        base_seed=base_seed,
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(int(base_seed))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader


def build_healthgait_loaders_from_config(config):
    """Build train and validation DataLoaders from one reproducible config."""
    train_ds, val_ds = build_healthgait_datasets_from_config(config)

    train_generator = torch.Generator()
    train_generator.manual_seed(config.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    return train_loader, val_loader


__all__ = [
    "DEFAULT_BASE_SEED",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_CHANNELS",
    "DEFAULT_CLIP_LENGTH",
    "DEFAULT_IMAGE_SIZE",
    "DEFAULT_MANIFEST_SEED",
    "DEFAULT_MODALITY",
    "DEFAULT_WINDOW_POLICY",
    "HealthGaitLoaderConfig",
    "VALID_WINDOW_POLICIES",
    "build_healthgait_dataset_from_config",
    "build_healthgait_datasets",
    "build_healthgait_datasets_from_config",
    "build_healthgait_loaders",
    "build_healthgait_loaders_from_config",
    "find_repo_root",
    "healthgait_manifest_path",
    "preview_manifest",
    "run_healthgait_motion_diagnostics",
    "summarize_healthgait_manifest",
    "write_healthgait_dummy_probe_exports",
    "write_healthgait_metadata_summary",
]
