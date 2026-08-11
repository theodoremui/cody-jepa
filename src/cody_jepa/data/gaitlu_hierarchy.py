"""Finalize nested GaitLU pools for the hierarchical-diversity experiment."""

from __future__ import annotations

import csv
import os
from pathlib import Path
import random
import shutil
import tempfile

from .gaitlu import GAITLU_MANIFEST_COLUMNS


HIERARCHY_REGISTRY_COLUMNS = (
    "model_label",
    "replicate",
    "sequence_support",
    "window_policy",
    "train_manifest",
    "val_manifest",
    "pool_seed",
    "optimization_seed",
    "replicate_seed",
    "unique_sequences",
    "training_exposure",
    "anchor_spacing",
)
HIERARCHY_REPLICATES = tuple(range(8))
HIERARCHY_SUPPORT_LEVELS = ("low", "high")
HIERARCHY_WINDOW_POLICIES = ("frozen_random", "resampled_anchor")
_INTEGER_REGISTRY_COLUMNS = (
    "replicate",
    "pool_seed",
    "optimization_seed",
    "replicate_seed",
    "unique_sequences",
    "training_exposure",
    "anchor_spacing",
)
_INVENTORY_REQUIRED_COLUMNS = {
    "sequence_id",
    "source_group",
    "shard_path",
    "record_offset",
    "record_size",
    "num_frames",
    "height",
    "width",
    "eligible",
}


def _write_csv(path: Path, fieldnames, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _read_inventory(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = _INVENTORY_REQUIRED_COLUMNS.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"inventory is missing required columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("inventory contains no rows")

    seen_ids: dict[str, int] = {}
    for line_no, row in enumerate(rows, start=2):
        sequence_id = row["sequence_id"]
        source_group = row["source_group"]
        if not sequence_id or sequence_id != sequence_id.strip():
            raise ValueError(
                f"inventory row {line_no}: sequence_id must be nonempty without outer whitespace"
            )
        canonical_id = sequence_id.casefold()
        if canonical_id in seen_ids:
            raise ValueError(
                f"inventory row {line_no}: duplicate/case-colliding sequence_id "
                f"{sequence_id!r}"
            )
        seen_ids[canonical_id] = line_no
        if row["eligible"].casefold() not in {"true", "false"}:
            raise ValueError(f"inventory row {line_no}: eligible must be true or false")
        if row["eligible"].casefold() == "true" and (
            not source_group or source_group != source_group.strip()
        ):
            raise ValueError(
                f"inventory row {line_no}: eligible source_group must be nonempty "
                "without outer whitespace"
            )
    return rows


def _anchor_count(num_frames: int, *, clip_length: int, anchor_spacing: int) -> int:
    if num_frames < clip_length:
        return 0
    return (num_frames - clip_length) // anchor_spacing + 1


def _group_rows(rows) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row["source_group"], []).append(row)
    return groups


def _closest_group_prefix(group_names, groups, target: int) -> list[str]:
    chosen: list[str] = []
    count = 0
    for name in group_names:
        size = len(groups[name])
        if chosen and abs(count - target) <= abs(count + size - target):
            break
        chosen.append(name)
        count += size
        if count == target:
            break
    return chosen


def _manifest_rows(rows, split: str) -> list[dict[str, str]]:
    return [
        {
            "sequence_id": row["sequence_id"],
            "source_group": row["source_group"],
            "shard_path": row["shard_path"],
            "record_offset": row["record_offset"],
            "record_size": row["record_size"],
            "num_frames": row["num_frames"],
            "height": row["height"],
            "width": row["width"],
            "split": split,
        }
        for row in sorted(rows, key=lambda item: item["sequence_id"])
    ]


def _resolve_manifest(registry_path: Path, value: str) -> Path:
    relative = Path(value)
    if (
        not value
        or value != relative.as_posix()
        or relative.is_absolute()
        or ".." in relative.parts
    ):
        raise ValueError(
            f"registry manifest path must be a canonical safe relative path: {value!r}"
        )
    resolved = (registry_path.parent / relative).resolve()
    try:
        resolved.relative_to(registry_path.parent.resolve())
    except ValueError as error:
        raise ValueError(f"registry manifest escapes hierarchy directory: {value!r}") from error
    return resolved


def _read_manifest(
    path: Path, split: str
) -> tuple[set[str], set[str], dict[str, set[str]]]:
    if not path.is_file():
        raise ValueError(f"registry manifest does not exist: {path}")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != GAITLU_MANIFEST_COLUMNS:
            raise ValueError(
                f"manifest {path} must have exactly these columns in order: "
                + ",".join(GAITLU_MANIFEST_COLUMNS)
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"manifest contains no rows: {path}")
    sequence_ids: set[str] = set()
    canonical_ids: set[str] = set()
    groups: set[str] = set()
    group_members: dict[str, set[str]] = {}
    for line_no, row in enumerate(rows, start=2):
        sequence_id = row["sequence_id"]
        source_group = row["source_group"]
        canonical_id = sequence_id.casefold()
        if not sequence_id or canonical_id in canonical_ids:
            raise ValueError(f"manifest {path} row {line_no}: duplicate or empty sequence_id")
        if not source_group:
            raise ValueError(f"manifest {path} row {line_no}: source_group is empty")
        if row["split"] != split:
            raise ValueError(
                f"manifest {path} row {line_no}: expected split {split!r}, "
                f"got {row['split']!r}"
            )
        canonical_ids.add(canonical_id)
        sequence_ids.add(sequence_id)
        groups.add(source_group)
        group_members.setdefault(source_group, set()).add(sequence_id)
    return sequence_ids, groups, group_members


def read_hierarchy_registry(
    registry_path, *, clip_length: int = 16
) -> list[dict[str, object]]:
    """Read and strictly validate the complete 32-row hierarchy registry.

    Integer fields are returned as integers. Manifest paths remain safe paths relative
    to the registry directory; every referenced manifest is reopened and validated.
    """

    registry_path = Path(registry_path).expanduser().resolve()
    clip_length = int(clip_length)
    if clip_length <= 0:
        raise ValueError("clip_length must be positive")
    if not registry_path.is_file():
        raise FileNotFoundError(registry_path)
    with registry_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != HIERARCHY_REGISTRY_COLUMNS:
            raise ValueError(
                "hierarchy registry must have exactly these columns in order: "
                + ",".join(HIERARCHY_REGISTRY_COLUMNS)
            )
        raw_rows = list(reader)
    if len(raw_rows) != 32:
        raise ValueError(f"hierarchy registry must contain exactly 32 rows, found {len(raw_rows)}")

    rows: list[dict[str, object]] = []
    labels: set[str] = set()
    cells: dict[tuple[int, str, str], dict[str, object]] = {}
    for line_no, raw in enumerate(raw_rows, start=2):
        row: dict[str, object] = dict(raw)
        for column in _INTEGER_REGISTRY_COLUMNS:
            try:
                row[column] = int(raw[column])
            except ValueError as error:
                raise ValueError(
                    f"hierarchy registry row {line_no}: {column} must be an integer"
                ) from error
        label = str(row["model_label"])
        if (
            not label
            or label != label.strip()
            or label in {".", ".."}
            or "/" in label
            or "\\" in label
            or label in labels
        ):
            raise ValueError(
                f"hierarchy registry row {line_no}: model_label must be a unique, "
                "safe path component"
            )
        labels.add(label)
        replicate = int(row["replicate"])
        support = str(row["sequence_support"])
        policy = str(row["window_policy"])
        key = (replicate, support, policy)
        if support not in HIERARCHY_SUPPORT_LEVELS:
            raise ValueError(f"hierarchy registry row {line_no}: unknown support {support!r}")
        if policy not in HIERARCHY_WINDOW_POLICIES:
            raise ValueError(f"hierarchy registry row {line_no}: unknown policy {policy!r}")
        if key in cells:
            raise ValueError(f"hierarchy registry contains duplicate cell {key}")
        if int(row["unique_sequences"]) <= 0:
            raise ValueError(f"hierarchy registry row {line_no}: unique_sequences must be positive")
        if int(row["training_exposure"]) <= 0:
            raise ValueError(f"hierarchy registry row {line_no}: training_exposure must be positive")
        if int(row["anchor_spacing"]) <= 0:
            raise ValueError(f"hierarchy registry row {line_no}: anchor_spacing must be positive")
        for column in ("pool_seed", "optimization_seed", "replicate_seed"):
            if int(row[column]) < 0:
                raise ValueError(
                    f"hierarchy registry row {line_no}: {column} must be nonnegative"
                )
        cells[key] = row
        rows.append(row)

    expected_cells = {
        (replicate, support, policy)
        for replicate in HIERARCHY_REPLICATES
        for support in HIERARCHY_SUPPORT_LEVELS
        for policy in HIERARCHY_WINDOW_POLICIES
    }
    if set(cells) != expected_cells:
        missing = sorted(expected_cells.difference(cells))
        extra = sorted(set(cells).difference(expected_cells))
        raise ValueError(f"hierarchy registry cells are incomplete: missing={missing}, extra={extra}")

    exposures = {int(row["training_exposure"]) for row in rows}
    spacings = {int(row["anchor_spacing"]) for row in rows}
    val_manifests = {str(row["val_manifest"]) for row in rows}
    if len(exposures) != 1:
        raise ValueError("all hierarchy rows must use the same training exposure")
    if len(spacings) != 1:
        raise ValueError("all hierarchy rows must use the same anchor spacing")
    if len(val_manifests) != 1:
        raise ValueError("all hierarchy rows must use one common validation manifest")

    anchor_spacing = next(iter(spacings))
    inventory_rows = _read_inventory(registry_path.parent.parent / "inventory.csv")
    canonical_group_members: dict[str, set[str]] = {}
    for inventory_row in inventory_rows:
        if inventory_row["eligible"].casefold() != "true":
            continue
        try:
            num_frames = int(inventory_row["num_frames"])
        except ValueError as error:
            raise ValueError(
                f"eligible sequence {inventory_row['sequence_id']!r} has invalid num_frames"
            ) from error
        if _anchor_count(
            num_frames,
            clip_length=clip_length,
            anchor_spacing=anchor_spacing,
        ) >= 2:
            canonical_group_members.setdefault(
                inventory_row["source_group"], set()
            ).add(inventory_row["sequence_id"])

    def validate_whole_groups(label, group_members):
        for group, members in group_members.items():
            if canonical_group_members.get(group) != members:
                raise ValueError(f"{label}: manifest must contain whole source groups")

    val_path = _resolve_manifest(registry_path, next(iter(val_manifests)))
    val_ids, val_groups, val_members = _read_manifest(val_path, "val")
    validate_whole_groups("validation", val_members)
    train_paths: set[Path] = set()
    manifest_cache: dict[Path, tuple[set[str], set[str], dict[str, set[str]]]] = {}
    for replicate in HIERARCHY_REPLICATES:
        block = [row for row in rows if int(row["replicate"]) == replicate]
        for field in ("pool_seed", "optimization_seed", "replicate_seed"):
            if len({int(row[field]) for row in block}) != 1:
                raise ValueError(f"replicate {replicate}: {field} must match across all four cells")
        support_values: dict[
            str, tuple[set[str], set[str], dict[str, set[str]]]
        ] = {}
        for support in HIERARCHY_SUPPORT_LEVELS:
            paired = [cells[(replicate, support, policy)] for policy in HIERARCHY_WINDOW_POLICIES]
            train_names = {str(row["train_manifest"]) for row in paired}
            counts = {int(row["unique_sequences"]) for row in paired}
            if len(train_names) != 1 or len(counts) != 1:
                raise ValueError(
                    f"replicate {replicate} {support}: policies must reuse one manifest and count"
                )
            train_name = next(iter(train_names))
            train_path = _resolve_manifest(registry_path, train_name)
            train_paths.add(train_path)
            if train_path not in manifest_cache:
                manifest_cache[train_path] = _read_manifest(train_path, "train")
            sequence_ids, groups, group_members = manifest_cache[train_path]
            validate_whole_groups(f"replicate {replicate} {support}", group_members)
            if len(sequence_ids) != next(iter(counts)):
                raise ValueError(
                    f"replicate {replicate} {support}: registry sequence count does not "
                    "match its manifest"
                )
            if sequence_ids & val_ids or groups & val_groups:
                raise ValueError(
                    f"replicate {replicate} {support}: training and holdout are not group-disjoint"
                )
            support_values[support] = (sequence_ids, groups, group_members)
        low_ids, low_groups, low_members = support_values["low"]
        high_ids, high_groups, high_members = support_values["high"]
        if not low_ids < high_ids or not low_groups < high_groups:
            raise ValueError(
                f"replicate {replicate}: low sequences and groups must be strict subsets of high"
            )
        if any(low_members[group] != high_members[group] for group in low_groups):
            raise ValueError(
                f"replicate {replicate}: low must contain whole source groups from high"
            )
    if len(train_paths) != 16:
        raise ValueError(
            f"hierarchy registry must reference exactly 16 training manifests, found {len(train_paths)}"
        )
    return rows


def finalize_gaitlu_hierarchy(
    prepared_root,
    *,
    training_exposure: int,
    holdout_target: int = 10_000,
    holdout_seed: int = 20_260_806,
    pool_seeds=HIERARCHY_REPLICATES,
    low_target: int = 2_500,
    high_target: int = 250_000,
    clip_length: int = 16,
    anchor_spacing: int = 8,
    optimization_seeds=None,
    replicate_seeds=None,
) -> dict[str, object]:
    """Create the common holdout, 16 nested manifests, and 32-row registry."""

    prepared_root = Path(prepared_root).expanduser().resolve()
    if not prepared_root.is_dir():
        raise ValueError(f"prepared_root is not a directory: {prepared_root}")
    hierarchy_root = prepared_root / "hierarchy"
    if hierarchy_root.exists():
        raise FileExistsError(f"refusing to overwrite existing hierarchy output: {hierarchy_root}")

    pool_seeds = tuple(map(int, pool_seeds))
    if len(pool_seeds) != 8 or len(set(pool_seeds)) != 8:
        raise ValueError("hierarchy finalization requires exactly eight distinct pool seeds")
    optimization_seeds = tuple(
        pool_seeds if optimization_seeds is None else map(int, optimization_seeds)
    )
    replicate_seeds = tuple(pool_seeds if replicate_seeds is None else map(int, replicate_seeds))
    if len(optimization_seeds) != 8 or len(replicate_seeds) != 8:
        raise ValueError("optimization_seeds and replicate_seeds must each contain eight values")
    if min(
        int(training_exposure),
        int(holdout_target),
        int(low_target),
        int(high_target),
        int(clip_length),
        int(anchor_spacing),
    ) <= 0:
        raise ValueError("exposure, targets, clip length, and anchor spacing must be positive")
    if int(low_target) >= int(high_target):
        raise ValueError("low_target must be smaller than high_target")

    inventory_rows = _read_inventory(prepared_root / "inventory.csv")
    eligible_rows: list[dict[str, str]] = []
    for row in inventory_rows:
        if row["eligible"].casefold() != "true":
            continue
        try:
            num_frames = int(row["num_frames"])
        except ValueError as error:
            raise ValueError(
                f"eligible sequence {row['sequence_id']!r} has invalid num_frames"
            ) from error
        if _anchor_count(
            num_frames,
            clip_length=int(clip_length),
            anchor_spacing=int(anchor_spacing),
        ) >= 2:
            eligible_rows.append(row)
    if not eligible_rows:
        raise ValueError("no exact-content-eligible sequences satisfy the two-anchor rule")

    groups = _group_rows(eligible_rows)
    ordered_groups = sorted(groups)
    random.Random(int(holdout_seed)).shuffle(ordered_groups)
    holdout_groups = set(
        _closest_group_prefix(ordered_groups, groups, int(holdout_target))
    )
    holdout_rows = [row for row in eligible_rows if row["source_group"] in holdout_groups]
    training_rows = [row for row in eligible_rows if row["source_group"] not in holdout_groups]
    if len(training_rows) < int(high_target):
        raise ValueError(
            f"only {len(training_rows)} hierarchy-training sequences remain for high target "
            f"{int(high_target)}"
        )
    training_groups = _group_rows(training_rows)

    selections: list[tuple[int, int, set[str], set[str]]] = []
    for replicate, pool_seed in enumerate(pool_seeds):
        group_order = sorted(training_groups)
        random.Random(pool_seed).shuffle(group_order)
        low_groups = set(
            _closest_group_prefix(group_order, training_groups, int(low_target))
        )
        high_groups = set(
            _closest_group_prefix(group_order, training_groups, int(high_target))
        )
        if not low_groups < high_groups:
            raise ValueError(
                f"replicate {replicate}: source-group granularity cannot produce "
                "strictly nested low and high pools"
            )
        selections.append((replicate, pool_seed, low_groups, high_groups))

    temporary_root = Path(tempfile.mkdtemp(prefix=".hierarchy.", dir=prepared_root))
    try:
        val_name = "manifests/common-holdout.csv"
        _write_csv(
            temporary_root / val_name,
            GAITLU_MANIFEST_COLUMNS,
            _manifest_rows(holdout_rows, "val"),
        )
        registry: list[dict[str, object]] = []
        pool_counts: list[dict[str, int | str]] = []
        for replicate, pool_seed, low_groups, high_groups in selections:
            for support, selected_groups in (("low", low_groups), ("high", high_groups)):
                selected_rows = [
                    row for row in training_rows if row["source_group"] in selected_groups
                ]
                train_name = f"manifests/replicate-{replicate}-{support}.csv"
                _write_csv(
                    temporary_root / train_name,
                    GAITLU_MANIFEST_COLUMNS,
                    _manifest_rows(selected_rows, "train"),
                )
                pool_counts.append(
                    {
                        "replicate": replicate,
                        "sequence_support": support,
                        "unique_sequences": len(selected_rows),
                        "source_groups": len(selected_groups),
                    }
                )
                for policy in HIERARCHY_WINDOW_POLICIES:
                    registry.append(
                        {
                            "model_label": (
                                f"replicate-{replicate}-{support}-{policy.replace('_', '-')}"
                            ),
                            "replicate": replicate,
                            "sequence_support": support,
                            "window_policy": policy,
                            "train_manifest": train_name,
                            "val_manifest": val_name,
                            "pool_seed": pool_seed,
                            "optimization_seed": optimization_seeds[replicate],
                            "replicate_seed": replicate_seeds[replicate],
                            "unique_sequences": len(selected_rows),
                            "training_exposure": int(training_exposure),
                            "anchor_spacing": int(anchor_spacing),
                        }
                    )
        registry_path = temporary_root / "training_registry.csv"
        _write_csv(registry_path, HIERARCHY_REGISTRY_COLUMNS, registry)
        read_hierarchy_registry(registry_path, clip_length=int(clip_length))
        os.replace(temporary_root, hierarchy_root)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise

    return {
        "version": "gaitlu-hierarchy-pools-v1",
        "registry": str(hierarchy_root / "training_registry.csv"),
        "holdout_manifest": str(hierarchy_root / val_name),
        "holdout_sequences": len(holdout_rows),
        "eligible_after_temporal_rule": len(eligible_rows),
        "excluded_by_temporal_rule": sum(
            row["eligible"].casefold() == "true" for row in inventory_rows
        )
        - len(eligible_rows),
        "training_manifests": 16,
        "registry_rows": 32,
        "training_exposure": int(training_exposure),
        "pools": pool_counts,
    }


__all__ = [
    "HIERARCHY_REGISTRY_COLUMNS",
    "HIERARCHY_REPLICATES",
    "HIERARCHY_SUPPORT_LEVELS",
    "HIERARCHY_WINDOW_POLICIES",
    "finalize_gaitlu_hierarchy",
    "read_hierarchy_registry",
]
