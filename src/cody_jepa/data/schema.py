"""Canonical Health&Gait manifest schema and factor vocabulary."""

FACTOR_COLUMNS = ("recording_id", "speed", "clothing", "direction")
RECORDING_HIERARCHY_COLUMNS = ("source_video_id", "direction_clip_id")
SHORTCUT_FEATURE_COLUMNS = (
    "shortcut_log_frame_count",
    "shortcut_duration_seconds",
    "shortcut_horizontal_centroid_drift_signed",
    "shortcut_horizontal_centroid_drift_absolute",
    "shortcut_foreground_area_mean",
    "shortcut_foreground_area_std",
    "shortcut_foreground_area_q25",
    "shortcut_foreground_area_median",
    "shortcut_foreground_area_q75",
)
REQUIRED_MANIFEST_COLUMNS = frozenset(
    {
        "subject_id",
        "modality",
        "gait_system",
        "trial",
        "frame_dir",
        "num_frames",
        "split",
        "fps",
        *FACTOR_COLUMNS,
        *RECORDING_HIERARCHY_COLUMNS,
        *SHORTCUT_FEATURE_COLUMNS,
    }
)
REQUIRED_NONEMPTY_FIELDS = frozenset(
    {
        "subject_id",
        "modality",
        "gait_system",
        "trial",
        *FACTOR_COLUMNS,
        *RECORDING_HIERARCHY_COLUMNS,
    }
)
VALID_SPLITS = frozenset({"train", "val"})
VALID_IMAGE_VERIFY_MODES = frozenset({"none", "sample", "all"})
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
VALID_SPEEDS = frozenset({"UGS", "FGS"})
VALID_CLOTHING = frozenset({"WoJ", "WJ"})
VALID_DIRECTIONS = frozenset({"R2L", "L2R"})


__all__ = [
    "FACTOR_COLUMNS",
    "IMAGE_SUFFIXES",
    "RECORDING_HIERARCHY_COLUMNS",
    "REQUIRED_MANIFEST_COLUMNS",
    "REQUIRED_NONEMPTY_FIELDS",
    "SHORTCUT_FEATURE_COLUMNS",
    "VALID_CLOTHING",
    "VALID_DIRECTIONS",
    "VALID_IMAGE_VERIFY_MODES",
    "VALID_SPEEDS",
    "VALID_SPLITS",
]
