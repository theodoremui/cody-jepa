from pathlib import Path
import csv

from torch.utils.data import DataLoader

from .dataset import HealthGaitManifestDataset


DEFAULT_BATCH_SIZE = 4
DEFAULT_CLIP_LENGTH = 16
DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_MANIFEST_SEED = 0
DEFAULT_MODALITY = "silhouette"


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
    )
    val_ds = HealthGaitManifestDataset(
        manifest,
        split="val",
        repo_root=root,
        clip_length=clip_length,
        image_size=image_size,
        random_windows=False,
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
):
    """Build train and validation DataLoaders with notebook-safe defaults."""
    train_ds, val_ds = build_healthgait_datasets(
        manifest_csv=manifest_csv,
        repo_root=repo_root,
        clip_length=clip_length,
        image_size=image_size,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_CLIP_LENGTH",
    "DEFAULT_IMAGE_SIZE",
    "DEFAULT_MANIFEST_SEED",
    "DEFAULT_MODALITY",
    "build_healthgait_datasets",
    "build_healthgait_loaders",
    "find_repo_root",
    "healthgait_manifest_path",
    "preview_manifest",
]
