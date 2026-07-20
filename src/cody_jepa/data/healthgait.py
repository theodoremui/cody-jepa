from pathlib import Path
import csv
from dataclasses import dataclass, replace

import torch
from torch.utils.data import DataLoader

from .dataset import HealthGaitManifestDataset, ManifestValidationError, VALID_SPLITS
from .healthgait_diagnostics import (
    run_healthgait_motion_diagnostics,
    summarize_healthgait_manifest,
    write_healthgait_dummy_probe_exports,
    write_healthgait_metadata_summary,
)


DEFAULT_BATCH_SIZE = 16
DEFAULT_CLIP_LENGTH = 16
DEFAULT_IMAGE_SIZE = (72, 72)
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
    prefetch_factor: int = 2
    train_crop_scale: tuple[float, float] = (1.0, 1.0)
    train_horizontal_flip_prob: float = 0.0
    expected_modality: str = DEFAULT_MODALITY
    strict_frame_sequence: bool = True
    image_verify_mode: str = "none"
    inventory_hash_mode: str = "sample"
    allowed_data_root: Path | None = None
    eval_windows: int = 1
    drop_last_train: bool = False

    def __post_init__(self):
        if not isinstance(self.strict_validation, bool):
            raise TypeError("strict_validation must be a bool")
        if not isinstance(self.pin_memory, bool):
            raise TypeError("pin_memory must be a bool")
        if not isinstance(self.strict_frame_sequence, bool):
            raise TypeError("strict_frame_sequence must be a bool")
        if not isinstance(self.drop_last_train, bool):
            raise TypeError("drop_last_train must be a bool")
        repo_root = Path(self.repo_root).expanduser().resolve()
        allowed_data_root = (
            repo_root
            if self.allowed_data_root is None
            else Path(self.allowed_data_root).expanduser().resolve()
        )
        manifest_csv = Path(self.manifest_csv).expanduser()
        manifest_csv = (
            manifest_csv.resolve()
            if manifest_csv.is_absolute()
            else (repo_root / manifest_csv).resolve()
        )
        object.__setattr__(self, "manifest_csv", manifest_csv)
        object.__setattr__(self, "repo_root", repo_root)
        object.__setattr__(self, "allowed_data_root", allowed_data_root)
        object.__setattr__(self, "clip_length", int(self.clip_length))
        object.__setattr__(self, "image_size", tuple(int(value) for value in self.image_size))
        object.__setattr__(self, "channels", int(self.channels))
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "strict_validation", bool(self.strict_validation))
        object.__setattr__(self, "batch_size", int(self.batch_size))
        object.__setattr__(self, "num_workers", int(self.num_workers))
        object.__setattr__(self, "pin_memory", bool(self.pin_memory))
        object.__setattr__(self, "prefetch_factor", int(self.prefetch_factor))
        object.__setattr__(
            self, "train_crop_scale", tuple(float(value) for value in self.train_crop_scale)
        )
        object.__setattr__(
            self, "train_horizontal_flip_prob", float(self.train_horizontal_flip_prob)
        )
        object.__setattr__(self, "expected_modality", str(self.expected_modality).strip())
        object.__setattr__(self, "strict_frame_sequence", bool(self.strict_frame_sequence))
        object.__setattr__(self, "image_verify_mode", str(self.image_verify_mode))
        object.__setattr__(self, "inventory_hash_mode", str(self.inventory_hash_mode))
        object.__setattr__(self, "eval_windows", int(self.eval_windows))

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
        if self.prefetch_factor <= 0:
            raise ValueError("prefetch_factor must be positive")
        if (
            len(self.train_crop_scale) != 2
            or not 0.0 < self.train_crop_scale[0] <= self.train_crop_scale[1] <= 1.0
        ):
            raise ValueError("train_crop_scale must be an ordered pair in (0, 1]")
        if not 0.0 <= self.train_horizontal_flip_prob <= 1.0:
            raise ValueError("train_horizontal_flip_prob must be in [0, 1]")
        if not self.expected_modality:
            raise ValueError("expected_modality must be nonempty")
        if self.image_verify_mode not in {"none", "sample", "all"}:
            raise ValueError("image_verify_mode must be one of {all, none, sample}")
        if self.inventory_hash_mode not in {"sample", "full"}:
            raise ValueError("inventory_hash_mode must be one of {full, sample}")
        if self.eval_windows <= 0:
            raise ValueError("eval_windows must be positive")

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
            "prefetch_factor": self.prefetch_factor,
            "train_crop_scale": list(self.train_crop_scale),
            "train_horizontal_flip_prob": self.train_horizontal_flip_prob,
            "expected_modality": self.expected_modality,
            "strict_frame_sequence": self.strict_frame_sequence,
            "image_verify_mode": self.image_verify_mode,
            "inventory_hash_mode": self.inventory_hash_mode,
            "allowed_data_root": str(self.allowed_data_root),
            "eval_windows": self.eval_windows,
            "drop_last_train": self.drop_last_train,
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


def build_healthgait_dataset_from_config(
    config, *, _validated_samples=None, _manifest_sha256=None, _inventory_sha256=None
):
    """Build one Health&Gait dataset from an explicit loader config."""
    return HealthGaitManifestDataset(
        config.manifest_csv,
        split=config.split,
        repo_root=config.repo_root,
        clip_length=config.clip_length,
        image_size=config.image_size,
        random_windows=config.uses_random_windows(),
        base_seed=config.seed,
        crop_scale=(
            config.train_crop_scale
            if config.split == "train" and config.uses_random_windows()
            else (1.0, 1.0)
        ),
        horizontal_flip_prob=(
            config.train_horizontal_flip_prob
            if config.split == "train" and config.uses_random_windows()
            else 0.0
        ),
        expected_modality=config.expected_modality,
        strict_frame_sequence=config.strict_frame_sequence,
        image_verify_mode=config.image_verify_mode,
        inventory_hash_mode=config.inventory_hash_mode,
        allowed_data_root=config.allowed_data_root,
        deterministic_windows=(config.eval_windows if config.split == "val" else 1),
        _validated_samples=_validated_samples,
        _manifest_sha256=_manifest_sha256,
        _inventory_sha256=_inventory_sha256,
    )


def build_healthgait_datasets_from_config(config):
    """Build train and validation Health&Gait datasets from one reproducible config."""
    train_ds = build_healthgait_dataset_from_config(config.for_split("train"))
    val_ds = build_healthgait_dataset_from_config(
        config.for_split("val"),
        _validated_samples=train_ds._all_samples,
        _manifest_sha256=train_ds._manifest_sha256,
        _inventory_sha256=train_ds._inventory_sha256,
    )
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


def build_healthgait_loaders_from_config(config, datasets=None):
    """Build train and validation DataLoaders from one reproducible config."""
    train_ds, val_ds = (
        build_healthgait_datasets_from_config(config) if datasets is None else datasets
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(config.seed)

    worker_options = (
        {"prefetch_factor": config.prefetch_factor}
        if config.num_workers > 0
        else {}
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        generator=train_generator,
        drop_last=config.drop_last_train,
        persistent_workers=False,
        **worker_options,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=False,
        **worker_options,
    )

    return train_loader, val_loader


@torch.inference_mode()
def audit_healthgait_clip_quality(
    datasets,
    *,
    min_motion_energy=1e-5,
    min_foreground_fraction=1e-4,
    max_foreground_fraction=0.95,
    foreground_threshold=0.05,
):
    """Decode one deterministic clip from every sequence in both splits.

    This is deliberately an all-sequence preflight rather than a first-batch
    spot check. It catches blank or static sequences before a long training run.
    """
    if len(datasets) != 2:
        raise ValueError("datasets must be the (train, val) dataset pair")
    thresholds = (
        float(min_motion_energy),
        float(min_foreground_fraction),
        float(max_foreground_fraction),
        float(foreground_threshold),
    )
    if not 0 <= thresholds[0] or not 0 <= thresholds[1] < thresholds[2] <= 1:
        raise ValueError("invalid clip-quality thresholds")
    if not 0 <= thresholds[3] <= 1:
        raise ValueError("foreground_threshold must be in [0, 1]")

    summaries = {}
    failures = []
    for dataset in datasets:
        previous_epoch = dataset.epoch
        dataset.set_epoch(0)
        motion_values = []
        foreground_values = []
        try:
            for sample_index, sample in enumerate(dataset.samples):
                window_index = dataset.deterministic_windows // 2
                item_index = sample_index * dataset.deterministic_windows + window_index
                video = dataset[item_index]["video"]
                motion = float((video[1:] - video[:-1]).abs().mean())
                foreground = float((video > foreground_threshold).float().mean())
                motion_values.append(motion)
                foreground_values.append(foreground)
                reasons = []
                if motion < min_motion_energy:
                    reasons.append(f"motion={motion:.3g}")
                if not min_foreground_fraction <= foreground <= max_foreground_fraction:
                    reasons.append(f"foreground={foreground:.3g}")
                if reasons:
                    failures.append(
                        f"{dataset.split}:{sample['sequence_id']} ({', '.join(reasons)})"
                    )
        finally:
            dataset.set_epoch(previous_epoch)
        summaries[dataset.split] = {
            "sequences_checked": len(motion_values),
            "min_motion_energy": min(motion_values),
            "mean_motion_energy": sum(motion_values) / len(motion_values),
            "min_foreground_fraction": min(foreground_values),
            "max_foreground_fraction": max(foreground_values),
            "mean_foreground_fraction": sum(foreground_values) / len(foreground_values),
        }
    if failures:
        shown = failures[:20]
        remainder = len(failures) - len(shown)
        suffix = f"\n- ... and {remainder} more" if remainder else ""
        raise ManifestValidationError(
            "Health&Gait clip-quality preflight failed:\n- "
            + "\n- ".join(shown)
            + suffix
        )
    return summaries


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
    "audit_healthgait_clip_quality",
    "find_repo_root",
    "healthgait_manifest_path",
    "preview_manifest",
    "run_healthgait_motion_diagnostics",
    "summarize_healthgait_manifest",
    "write_healthgait_dummy_probe_exports",
    "write_healthgait_metadata_summary",
]
