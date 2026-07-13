from pathlib import Path
import csv
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class HealthGaitManifestDataset(Dataset):
    def __init__(
        self,
        manifest_csv,
        split,
        repo_root,
        clip_length=16,
        image_size=(224, 224),
        random_windows=False,
    ):
        self.manifest_csv = Path(manifest_csv)
        self.repo_root = Path(repo_root)
        self.split = split
        self.clip_length = clip_length
        self.image_size = image_size
        self.random_windows = random_windows

        # Store prepared sample metadata here. The actual image tensors are
        # loaded later in __getitem__, one clip at a time.
        self.samples = []

        with self.manifest_csv.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] != split:
                    continue

                frame_dir = self._resolve_frame_dir(row["frame_dir"])
                frame_paths = self._list_frame_paths(frame_dir)

                # A short trial cannot provide a full training window.
                if len(frame_paths) < self.clip_length:
                    continue

                self.samples.append({
                    "subject_id": row["subject_id"],
                    "trial": row["trial"],
                    "gait_system": row["gait_system"],
                    "num_frames": int(row["num_frames"]),
                    "frame_dir": frame_dir,
                    "frame_paths": frame_paths,
                })

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

    def _choose_window_start(self, num_frames):
        """Choose where the contiguous clip window starts."""
        max_start = num_frames - self.clip_length

        if self.random_windows and max_start > 0:
            return random.randint(0, max_start)

        # Validation should be repeatable, so use the center window.
        return max_start // 2

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

        start = self._choose_window_start(len(frame_paths))
        selected_paths = frame_paths[start : start + self.clip_length]

        frames = [self._load_frame(path) for path in selected_paths]
        video = torch.stack(frames, dim=0)  # [T, C, H, W]

        return {
            "video": video,
            "subject_id": sample["subject_id"],
            "trial": sample["trial"],
            "gait_system": sample["gait_system"],
            "num_frames": sample["num_frames"],
            "frame_dir": str(sample["frame_dir"]),
        }


__all__ = ["HealthGaitManifestDataset"]

