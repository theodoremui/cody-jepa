from pathlib import Path
import csv
import hashlib
import json
import random

import numpy as np
from PIL import Image, UnidentifiedImageError
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
REQUIRED_NONEMPTY_FIELDS = frozenset({
    "subject_id",
    "modality",
    "gait_system",
    "trial",
})
VALID_SPLITS = frozenset({"train", "val"})
VALID_IMAGE_VERIFY_MODES = frozenset({"none", "sample", "all"})
VALID_INVENTORY_HASH_MODES = frozenset({"sample", "full"})
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})


class ManifestValidationError(ValueError):
    """Raised when a Health&Gait manifest cannot produce trustworthy clips."""


class HealthGaitManifestDataset(Dataset):
    """Strict, deterministic Health&Gait clip dataset.

    ``image_size`` is public API order ``(height, width)``. Spatial augmentation
    is clip-consistent and stateless: every frame in one clip receives the same
    crop and flip, derived from seed, sequence id, split, and epoch. Validation
    therefore remains deterministic across DataLoader worker counts.
    """

    def __init__(
        self,
        manifest_csv,
        split,
        repo_root,
        clip_length=16,
        image_size=(224, 224),
        random_windows=False,
        base_seed=0,
        crop_scale=(1.0, 1.0),
        horizontal_flip_prob=0.0,
        expected_modality="silhouette",
        strict_frame_sequence=True,
        image_verify_mode="none",
        inventory_hash_mode="sample",
        allowed_data_root=None,
        deterministic_windows=1,
        _validated_samples=None,
        _manifest_sha256=None,
        _inventory_sha256=None,
    ):
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.allowed_data_root = (
            self.repo_root
            if allowed_data_root is None
            else Path(allowed_data_root).expanduser().resolve()
        )
        manifest_path = Path(manifest_csv).expanduser()
        self.manifest_csv = (
            manifest_path.resolve()
            if manifest_path.is_absolute()
            else (self.repo_root / manifest_path).resolve()
        )
        if not isinstance(random_windows, bool):
            raise TypeError("random_windows must be a bool")
        if not isinstance(strict_frame_sequence, bool):
            raise TypeError("strict_frame_sequence must be a bool")
        self.split = str(split)
        self.clip_length = int(clip_length)
        self.image_size = tuple(int(value) for value in image_size)
        self.random_windows = random_windows
        self.base_seed = int(base_seed)
        self.crop_scale = tuple(float(value) for value in crop_scale)
        self.horizontal_flip_prob = float(horizontal_flip_prob)
        self.expected_modality = str(expected_modality).strip()
        self.strict_frame_sequence = strict_frame_sequence
        self.image_verify_mode = str(image_verify_mode)
        self.inventory_hash_mode = str(inventory_hash_mode)
        self.deterministic_windows = int(deterministic_windows)
        self.epoch = 0

        self._validate_options()
        if _validated_samples is None:
            self._all_samples = self._load_validated_manifest_samples()
            self._manifest_sha256 = self._hash_file(self.manifest_csv)
            self._inventory_sha256 = self._hash_inventory(self._all_samples)
        else:
            self._all_samples = list(_validated_samples)
            self._manifest_sha256 = str(_manifest_sha256)
            self._inventory_sha256 = str(_inventory_sha256)
        self.samples = [
            sample for sample in self._all_samples if sample["split"] == self.split
        ]
        if not self.samples:
            raise ManifestValidationError(
                f"manifest has no validated samples for split {self.split!r}"
            )
        self._signature = self._build_signature()

    def _validate_options(self):
        if not self.repo_root.exists() or not self.repo_root.is_dir():
            raise ValueError(f"repo_root must be an existing directory: {self.repo_root}")
        if not self.allowed_data_root.exists() or not self.allowed_data_root.is_dir():
            raise ValueError(
                "allowed_data_root must be an existing directory: "
                f"{self.allowed_data_root}"
            )
        if self.split not in VALID_SPLITS:
            valid = ", ".join(sorted(VALID_SPLITS))
            raise ValueError(f"split must be one of {{{valid}}}; got {self.split!r}")
        if self.clip_length <= 0:
            raise ValueError(f"clip_length must be positive; got {self.clip_length}")
        if len(self.image_size) != 2 or any(value <= 0 for value in self.image_size):
            raise ValueError(
                f"image_size must contain positive (height, width); got {self.image_size}"
            )
        if (
            len(self.crop_scale) != 2
            or not 0.0 < self.crop_scale[0] <= self.crop_scale[1] <= 1.0
        ):
            raise ValueError(
                "crop_scale must be an ordered pair in (0, 1]; "
                f"got {self.crop_scale}"
            )
        if not 0.0 <= self.horizontal_flip_prob <= 1.0:
            raise ValueError("horizontal_flip_prob must be in [0, 1]")
        if not self.expected_modality:
            raise ValueError("expected_modality must be nonempty")
        if self.image_verify_mode not in VALID_IMAGE_VERIFY_MODES:
            valid = ", ".join(sorted(VALID_IMAGE_VERIFY_MODES))
            raise ValueError(
                f"image_verify_mode must be one of {{{valid}}}; got {self.image_verify_mode!r}"
            )
        if self.inventory_hash_mode not in VALID_INVENTORY_HASH_MODES:
            valid = ", ".join(sorted(VALID_INVENTORY_HASH_MODES))
            raise ValueError(
                f"inventory_hash_mode must be one of {{{valid}}}; "
                f"got {self.inventory_hash_mode!r}"
            )
        if self.deterministic_windows <= 0:
            raise ValueError("deterministic_windows must be positive")
        if self.random_windows and self.deterministic_windows != 1:
            raise ValueError(
                "deterministic_windows must be 1 when random_windows=True"
            )

    @staticmethod
    def _hash_file(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _canonical_subject(subject_id):
        return subject_id.strip().casefold()

    def _hash_inventory(self, samples):
        """Hash the frame inventory and either sampled or complete file contents."""
        digest = hashlib.sha256()
        for sample in samples:
            try:
                portable_dir = sample["frame_dir"].relative_to(self.repo_root)
            except ValueError:
                portable_dir = sample["frame_dir"]
            header = (
                sample["sequence_id"],
                sample["split"],
                self._canonical_subject(sample["subject_id"]),
                str(portable_dir),
            )
            digest.update(json.dumps(header, separators=(",", ":")).encode("utf-8"))
            frame_paths = sample["frame_paths"]
            for path in frame_paths:
                digest.update(path.name.encode("utf-8"))
                digest.update(str(path.stat().st_size).encode("ascii"))
            content_paths = (
                frame_paths
                if self.inventory_hash_mode == "full"
                else [frame_paths[0], frame_paths[len(frame_paths) // 2], frame_paths[-1]]
            )
            for path in dict.fromkeys(content_paths):
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
        return digest.hexdigest()

    def _load_validated_manifest_samples(self):
        if not self.manifest_csv.exists():
            raise ManifestValidationError(f"Manifest does not exist: {self.manifest_csv}")

        errors = []
        samples = []
        split_counts = {split: 0 for split in VALID_SPLITS}
        split_subjects = {split: {} for split in VALID_SPLITS}
        seen_sequence_ids = {}
        seen_frame_dirs = {}

        with self.manifest_csv.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or [])
            missing = sorted(REQUIRED_MANIFEST_COLUMNS - fieldnames)
            if missing:
                raise ManifestValidationError(
                    f"Invalid Health&Gait manifest {self.manifest_csv}: "
                    f"missing required columns: {', '.join(missing)}"
                )

            for line_no, raw_row in enumerate(reader, start=2):
                row = {
                    key: (value.strip() if isinstance(value, str) else "" if value is None else value)
                    for key, value in raw_row.items()
                    if key is not None
                }
                row_errors = []

                if None in raw_row:
                    row_errors.append(f"row {line_no}: contains more fields than the header")

                for field in sorted(REQUIRED_NONEMPTY_FIELDS):
                    if not row.get(field):
                        row_errors.append(f"row {line_no}: {field} is empty")

                split = row.get("split", "")
                if split not in VALID_SPLITS:
                    valid = ", ".join(sorted(VALID_SPLITS))
                    row_errors.append(
                        f"row {line_no}: invalid split {split!r}; expected one of {{{valid}}}"
                    )
                else:
                    split_counts[split] += 1
                    if row.get("subject_id"):
                        canonical_subject = self._canonical_subject(row["subject_id"])
                        split_subjects[split].setdefault(
                            canonical_subject, row["subject_id"]
                        )

                modality = row.get("modality", "")
                if modality and modality.casefold() != self.expected_modality.casefold():
                    row_errors.append(
                        f"row {line_no}: modality {modality!r} does not match "
                        f"expected {self.expected_modality!r}"
                    )

                frame_dir_value = row.get("frame_dir", "")
                frame_dir = self._resolve_frame_dir(frame_dir_value) if frame_dir_value else None
                frame_paths = []
                if not frame_dir_value:
                    row_errors.append(f"row {line_no}: frame_dir is empty")
                elif not frame_dir.exists():
                    row_errors.append(f"row {line_no}: frame_dir does not exist: {frame_dir}")
                elif not frame_dir.is_dir():
                    row_errors.append(f"row {line_no}: frame_dir is not a directory: {frame_dir}")
                elif not self._is_within_allowed_data_root(frame_dir):
                    row_errors.append(
                        f"row {line_no}: frame_dir escapes allowed_data_root "
                        f"{self.allowed_data_root}: {frame_dir}"
                    )
                else:
                    frame_paths = self._list_frame_paths(frame_dir)
                    self._validate_frame_sequence(frame_paths, line_no, row_errors)
                    self._verify_frame_images(frame_paths, line_no, row_errors)

                declared_num_frames = None
                try:
                    declared_num_frames = int(row.get("num_frames", ""))
                    if declared_num_frames <= 0:
                        raise ValueError
                except (TypeError, ValueError):
                    row_errors.append(
                        f"row {line_no}: num_frames must be a positive integer; "
                        f"got {row.get('num_frames')!r}"
                    )

                actual_num_frames = len(frame_paths)
                if frame_dir is not None and frame_dir.is_dir():
                    if (
                        declared_num_frames is not None
                        and declared_num_frames != actual_num_frames
                    ):
                        row_errors.append(
                            f"row {line_no}: num_frames={declared_num_frames} but found "
                            f"{actual_num_frames} frame files in {frame_dir}"
                        )
                    if actual_num_frames < self.clip_length:
                        row_errors.append(
                            f"row {line_no}: only {actual_num_frames} frame files in {frame_dir}; "
                            f"clip_length requires at least {self.clip_length}"
                        )
                valid_window_starts = self._valid_window_starts(frame_paths)
                if frame_paths and not valid_window_starts:
                    row_errors.append(
                        f"row {line_no}: no contiguous {self.clip_length}-frame window"
                    )

                sequence_id = self._sequence_id(row)
                if sequence_id:
                    previous = seen_sequence_ids.get(sequence_id)
                    if previous is not None:
                        row_errors.append(
                            f"row {line_no}: duplicate sequence_id {sequence_id!r}; "
                            f"already used on row {previous}"
                        )
                    else:
                        seen_sequence_ids[sequence_id] = line_no

                if frame_dir is not None:
                    canonical_dir = str(frame_dir)
                    previous = seen_frame_dirs.get(canonical_dir)
                    if previous is not None:
                        row_errors.append(
                            f"row {line_no}: duplicate canonical frame_dir {frame_dir}; "
                            f"already used on row {previous}"
                        )
                    else:
                        seen_frame_dirs[canonical_dir] = line_no

                if row_errors:
                    errors.extend(row_errors)
                    continue

                samples.append({
                    "sequence_id": sequence_id,
                    "split": split,
                    "modality": modality,
                    "subject_id": row["subject_id"],
                    "trial": row["trial"],
                    "gait_system": row["gait_system"],
                    "num_frames": declared_num_frames,
                    "frame_dir": frame_dir,
                    "frame_paths": frame_paths,
                    "valid_window_starts": valid_window_starts,
                })

        for split in sorted(VALID_SPLITS):
            if split_counts[split] == 0:
                errors.append(f"manifest has no rows for split {split!r}")

        overlap = split_subjects["train"].keys() & split_subjects["val"].keys()
        if overlap:
            display_subjects = [split_subjects["train"][key] for key in sorted(overlap)]
            errors.append(
                "subject overlap between train and val splits: "
                + ", ".join(display_subjects)
            )

        if errors:
            raise ManifestValidationError(
                f"Invalid Health&Gait manifest {self.manifest_csv}:\n- "
                + "\n- ".join(errors)
            )
        return samples

    def _resolve_frame_dir(self, frame_dir):
        path = Path(frame_dir).expanduser()
        return path.resolve() if path.is_absolute() else (self.repo_root / path).resolve()

    def _is_within_allowed_data_root(self, path):
        try:
            Path(path).relative_to(self.allowed_data_root)
        except ValueError:
            return False
        return True

    def _list_frame_paths(self, frame_dir):
        frames = [
            path
            for path in frame_dir.iterdir()
            if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
        ]
        return sorted(frames, key=self._frame_sort_key)

    @staticmethod
    def _frame_sort_key(path):
        return (0, int(path.stem)) if path.stem.isdigit() else (1, path.name.casefold())

    def _validate_frame_sequence(self, frame_paths, line_no, errors):
        if not self.strict_frame_sequence or not frame_paths:
            return
        nonnumeric = [path.name for path in frame_paths if not path.stem.isdigit()]
        if nonnumeric:
            errors.append(
                f"row {line_no}: frame names must have numeric stems; "
                f"found {nonnumeric[:3]}"
            )
            return
        indices = [int(path.stem) for path in frame_paths]
        if len(indices) != len(set(indices)):
            errors.append(f"row {line_no}: duplicate numeric frame indices")
            return
        # Gaps are allowed in a source sequence, but __getitem__ only samples
        # windows whose numeric indices are contiguous. This avoids inventing a
        # large motion jump while retaining valid runs from partially missing data.

    def _valid_window_starts(self, frame_paths):
        if len(frame_paths) < self.clip_length:
            return []
        if not self.strict_frame_sequence:
            return list(range(len(frame_paths) - self.clip_length + 1))
        if any(not path.stem.isdigit() for path in frame_paths):
            return []
        indices = [int(path.stem) for path in frame_paths]
        return [
            start
            for start in range(len(frame_paths) - self.clip_length + 1)
            if indices[start + self.clip_length - 1] - indices[start]
            == self.clip_length - 1
            and len(set(indices[start : start + self.clip_length])) == self.clip_length
        ]

    def _verify_frame_images(self, frame_paths, line_no, errors):
        if self.image_verify_mode == "none" or not frame_paths:
            return
        if self.image_verify_mode == "all":
            selected = frame_paths
        else:
            selected = [frame_paths[0], frame_paths[len(frame_paths) // 2], frame_paths[-1]]
            selected = list(dict.fromkeys(selected))
        expected_size = None
        for path in selected:
            try:
                with Image.open(path) as image:
                    size = image.size
                    image.verify()
                if expected_size is None:
                    expected_size = size
                elif size != expected_size:
                    errors.append(
                        f"row {line_no}: inconsistent frame dimensions; "
                        f"{path} is {size}, expected {expected_size}"
                    )
            except (OSError, ValueError, UnidentifiedImageError) as error:
                errors.append(f"row {line_no}: corrupt image {path}: {error}")

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

    def _build_signature(self):
        try:
            portable_allowed_root = str(self.allowed_data_root.relative_to(self.repo_root))
        except ValueError:
            portable_allowed_root = str(self.allowed_data_root)
        contract = {
            "manifest_sha256": self._manifest_sha256,
            "inventory_sha256": self._inventory_sha256,
            "clip_length": self.clip_length,
            "image_size": self.image_size,
            "base_seed": self.base_seed,
            "random_windows": self.random_windows,
            "crop_scale": self.crop_scale,
            "horizontal_flip_prob": self.horizontal_flip_prob,
            "expected_modality": self.expected_modality,
            "strict_frame_sequence": self.strict_frame_sequence,
            "image_verify_mode": self.image_verify_mode,
            "inventory_hash_mode": self.inventory_hash_mode,
            "allowed_data_root": portable_allowed_root,
            "deterministic_windows": self.deterministic_windows,
            "split": self.split,
        }
        payload = json.dumps(contract, sort_keys=True, separators=(",", ":"))
        return {
            "schema": 1,
            "manifest_sha256": self._manifest_sha256,
            "inventory_sha256": self._inventory_sha256,
            "dataset_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            "split": self.split,
            "sequence_count": len(self.samples),
            "sample_count": len(self),
        }

    def signature(self):
        return dict(self._signature)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _stable_seed(self, sequence_id, purpose):
        key = "\0".join([
            str(self.base_seed),
            str(sequence_id),
            str(self.epoch),
            str(self.split),
            str(purpose),
        ])
        digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="big", signed=False)

    def _choose_window_start(
        self, sequence_id, num_frames, window_index=0, valid_window_starts=None
    ):
        starts = (
            list(valid_window_starts)
            if valid_window_starts is not None
            else list(range(num_frames - self.clip_length + 1))
        )
        if not starts:
            raise ManifestValidationError(
                f"sequence {sequence_id!r} has no valid {self.clip_length}-frame window"
            )
        if self.random_windows and len(starts) > 1:
            selected = self._stable_seed(sequence_id, "window") % len(starts)
            return starts[selected]
        if self.deterministic_windows > 1 and len(starts) > 1:
            selected = round(window_index * (len(starts) - 1) / (self.deterministic_windows - 1))
            return starts[selected]
        return starts[len(starts) // 2]

    def _clip_transform(self, sequence_id):
        rng = random.Random(self._stable_seed(sequence_id, "spatial-transform"))
        scale = rng.uniform(*self.crop_scale)
        return {
            "scale": scale,
            "top_fraction": rng.random(),
            "left_fraction": rng.random(),
            "flip": rng.random() < self.horizontal_flip_prob,
        }

    @staticmethod
    def _frame_index(path, fallback_index):
        return int(path.stem) if path.stem.isdigit() else fallback_index

    def _clip_metadata(self, sample, start, selected_paths, window_index=0):
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
            "window_index": window_index,
            "frame_indices": frame_indices,
            "num_frames": sample["num_frames"],
            "frame_dir": str(sample["frame_dir"]),
        }

    def _load_frame(self, path, transform):
        try:
            with Image.open(path) as image:
                image = image.convert("L")
                width, height = image.size
                crop_width = max(1, round(width * transform["scale"]))
                crop_height = max(1, round(height * transform["scale"]))
                max_left = width - crop_width
                max_top = height - crop_height
                left = round(transform["left_fraction"] * max_left)
                top = round(transform["top_fraction"] * max_top)
                image = image.crop((left, top, left + crop_width, top + crop_height))
                if transform["flip"]:
                    image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                out_height, out_width = self.image_size
                image = image.resize((out_width, out_height), Image.Resampling.BILINEAR)
                array = np.array(image, dtype=np.float32, copy=True) / 255.0
        except (OSError, ValueError, UnidentifiedImageError) as error:
            raise ManifestValidationError(f"failed to decode frame {path}: {error}") from error
        return torch.from_numpy(array).unsqueeze(0)

    def __len__(self):
        return len(self.samples) * self.deterministic_windows

    def __getitem__(self, idx):
        sample_index, window_index = divmod(idx, self.deterministic_windows)
        sample = self.samples[sample_index]
        frame_paths = sample["frame_paths"]
        stable_id = sample["sequence_id"]
        start = self._choose_window_start(
            stable_id,
            len(frame_paths),
            window_index,
            sample["valid_window_starts"],
        )
        selected_paths = frame_paths[start : start + self.clip_length]
        transform = self._clip_transform(stable_id)
        video = torch.stack(
            [self._load_frame(path, transform) for path in selected_paths], dim=0
        )
        return {
            "video": video,
            **self._clip_metadata(sample, start, selected_paths, window_index),
        }


__all__ = [
    "HealthGaitManifestDataset",
    "ManifestValidationError",
    "REQUIRED_MANIFEST_COLUMNS",
    "REQUIRED_NONEMPTY_FIELDS",
    "VALID_IMAGE_VERIFY_MODES",
    "VALID_INVENTORY_HASH_MODES",
    "VALID_SPLITS",
]
