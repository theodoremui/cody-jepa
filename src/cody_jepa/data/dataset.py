from pathlib import Path
import csv
import hashlib

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


REQUIRED_MANIFEST_COLUMNS = frozenset({
    "subject_id",
    "modality",
    "gait_system",
    "trial",
    "frame_dir",
    "num_frames",
    "split",
})
VALID_SPLITS = frozenset({"train", "val"})


class ManifestValidationError(ValueError):
    """Raised when a Health&Gait manifest cannot produce trustworthy clips."""


class HealthGaitManifestDataset(Dataset):
    def __init__(
        self,
        manifest_csv,
        split,
        repo_root,
        clip_length=16,
        image_size=(224, 224),
        random_windows=False,
        base_seed=0,
    ):
        self.manifest_csv = Path(manifest_csv)
        self.repo_root = Path(repo_root)
        self.split = split
        self.clip_length = clip_length
        self.image_size = image_size
        self.random_windows = random_windows
        self.base_seed = int(base_seed)
        self.epoch = 0

        if self.split not in VALID_SPLITS:
            valid = ", ".join(sorted(VALID_SPLITS))
            raise ValueError(f"split must be one of {{{valid}}}; got {self.split!r}")

        # Store prepared sample metadata here. The actual image tensors are
        # loaded later in __getitem__, one clip at a time.
        self.samples = [
            sample
            for sample in self._load_validated_manifest_samples()
            if sample["split"] == self.split
        ]

    def _load_validated_manifest_samples(self):
        if not self.manifest_csv.exists():
            raise ManifestValidationError(f"Manifest does not exist: {self.manifest_csv}")

        errors = []
        samples = []
        split_counts = {split: 0 for split in VALID_SPLITS}
        split_subjects = {split: set() for split in VALID_SPLITS}

        with self.manifest_csv.open(newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames or [])
            missing = sorted(REQUIRED_MANIFEST_COLUMNS - fieldnames)
            if missing:
                missing_text = ", ".join(missing)
                raise ManifestValidationError(
                    f"Invalid Health&Gait manifest {self.manifest_csv}: "
                    f"missing required columns: {missing_text}"
                )

            for line_no, raw_row in enumerate(reader, start=2):
                row = {
                    key: value.strip() if isinstance(value, str) else value
                    for key, value in raw_row.items()
                }
                row_errors = []

                split = row["split"]
                if split not in VALID_SPLITS:
                    valid = ", ".join(sorted(VALID_SPLITS))
                    row_errors.append(
                        f"row {line_no}: invalid split {split!r}; expected one of {{{valid}}}"
                    )
                else:
                    split_counts[split] += 1
                    if row["subject_id"]:
                        split_subjects[split].add(row["subject_id"])

                frame_dir_value = row["frame_dir"]
                frame_dir = self._resolve_frame_dir(frame_dir_value) if frame_dir_value else None
                frame_paths = []
                if not frame_dir_value:
                    row_errors.append(f"row {line_no}: frame_dir is empty")
                elif not frame_dir.exists():
                    row_errors.append(f"row {line_no}: frame_dir does not exist: {frame_dir}")
                elif not frame_dir.is_dir():
                    row_errors.append(f"row {line_no}: frame_dir is not a directory: {frame_dir}")
                else:
                    frame_paths = self._list_frame_paths(frame_dir)

                declared_num_frames = None
                try:
                    declared_num_frames = int(row["num_frames"])
                except (TypeError, ValueError):
                    row_errors.append(
                        f"row {line_no}: num_frames must be an integer; got {row['num_frames']!r}"
                    )

                actual_num_frames = len(frame_paths)
                frame_dir_is_valid = (
                    frame_dir is not None and frame_dir.exists() and frame_dir.is_dir()
                )
                if frame_dir_is_valid and declared_num_frames is not None:
                    if declared_num_frames != actual_num_frames:
                        row_errors.append(
                            f"row {line_no}: num_frames={declared_num_frames} but found "
                            f"{actual_num_frames} frame files in {frame_dir}"
                        )

                if frame_dir_is_valid:
                    if actual_num_frames < self.clip_length:
                        row_errors.append(
                            f"row {line_no}: only {actual_num_frames} frame files in {frame_dir}; "
                            f"clip_length requires at least {self.clip_length}"
                        )

                if row_errors:
                    errors.extend(row_errors)
                    continue

                samples.append({
                    "sequence_id": self._sequence_id(row),
                    "split": split,
                    "modality": row["modality"],
                    "subject_id": row["subject_id"],
                    "trial": row["trial"],
                    "gait_system": row["gait_system"],
                    "num_frames": declared_num_frames,
                    "frame_dir": frame_dir,
                    "frame_paths": frame_paths,
                })

        for split in sorted(VALID_SPLITS):
            if split_counts[split] == 0:
                errors.append(f"manifest has no rows for split {split!r}")

        overlap = split_subjects["train"] & split_subjects["val"]
        if overlap:
            subjects = ", ".join(sorted(overlap))
            errors.append(f"subject overlap between train and val splits: {subjects}")

        if errors:
            message = "\n- ".join(errors)
            raise ManifestValidationError(
                f"Invalid Health&Gait manifest {self.manifest_csv}:\n- {message}"
            )

        return samples

    def _resolve_frame_dir(self, frame_dir):
        """Resolve manifest paths against the repo root when they are relative."""
        path = Path(frame_dir)
        return path if path.is_absolute() else self.repo_root / path

    def _list_frame_paths(self, frame_dir):
        """Return image frames in temporal order."""
        frames = list(frame_dir.glob("*.jpg")) + list(frame_dir.glob("*.png"))
        return sorted(frames, key=self._frame_sort_key)

    @staticmethod
    def _frame_sort_key(path):
        # Health&Gait names frames as 001.jpg, 002.jpg, and so on.
        # Numeric sorting preserves time order even if a filename lacks zero padding.
        return int(path.stem) if path.stem.isdigit() else path.name

    @staticmethod
    def _sequence_id(row):
        if row.get("sequence_id"):
            return row["sequence_id"]

        if row.get("sample_id"):
            return row["sample_id"]

        return "::".join([
            row.get("subject_id", ""),
            row.get("modality", ""),
            row.get("gait_system", ""),
            row.get("trial", ""),
            row.get("frame_dir", ""),
        ])

    def set_epoch(self, epoch):
        """Set the epoch used by deterministic train-window sampling."""
        self.epoch = int(epoch)

    def _stable_seed(self, sequence_id):
        key = "\0".join([
            str(self.base_seed),
            str(sequence_id),
            str(self.epoch),
            str(self.split),
        ])
        digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="big", signed=False)

    def _choose_window_start(self, sequence_id, num_frames):
        """Choose where the contiguous clip window starts."""
        max_start = num_frames - self.clip_length

        if self.random_windows and max_start > 0:
            return self._stable_seed(sequence_id) % (max_start + 1)

        # Validation should be repeatable, so use the center window.
        return max_start // 2

    @staticmethod
    def _frame_index(path, fallback_index):
        return int(path.stem) if path.stem.isdigit() else fallback_index

    def _clip_metadata(self, sample, start, selected_paths):
        frame_indices = [
            self._frame_index(path, start + offset)
            for offset, path in enumerate(selected_paths)
        ]

        return {
            "sequence_id": sample["sequence_id"],
            "split": sample["split"],
            "modality": sample["modality"],
            "subject_id": sample["subject_id"],
            "gait_system": sample["gait_system"],
            "trial": sample["trial"],
            "window_start": start,
            "frame_indices": frame_indices,
            "num_frames": sample["num_frames"],
            "frame_dir": str(sample["frame_dir"]),
        }

    def _load_frame(self, path):
        """Load one image as a normalized grayscale tensor [C, H, W]."""
        with Image.open(path) as img:
            img = img.convert("L")
            img = img.resize(self.image_size, Image.BILINEAR)
            array = np.asarray(img, dtype=np.float32) / 255.0

        # The unsqueeze adds the channel dimension: [H, W] -> [1, H, W].
        return torch.from_numpy(array).unsqueeze(0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frame_paths = sample["frame_paths"]

        stable_id = sample["sequence_id"]
        start = self._choose_window_start(stable_id, len(frame_paths))
        selected_paths = frame_paths[start : start + self.clip_length]

        frames = [self._load_frame(path) for path in selected_paths]
        video = torch.stack(frames, dim=0)  # [T, C, H, W]

        return {
            "video": video,
            **self._clip_metadata(sample, start, selected_paths),
        }


__all__ = [
    "HealthGaitManifestDataset",
    "ManifestValidationError",
    "REQUIRED_MANIFEST_COLUMNS",
    "VALID_SPLITS",
]
