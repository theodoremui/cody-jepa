"""Trusted-pickle conversion and deterministic GaitLU scaling-pool construction."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import pickle
import random
import tarfile
import tempfile

import numpy as np

from .gaitlu import GAITLU_MANIFEST_COLUMNS


INVENTORY_COLUMNS = (
    "sequence_id",
    "source_shard",
    "source_member",
    "shard_path",
    "record_offset",
    "record_size",
    "num_frames",
    "height",
    "width",
    "content_sha256",
    "foreground_fraction",
    "empty_frame_fraction",
    "intermediate_fraction",
    "valid",
    "exclusion_reason",
)
FINAL_INVENTORY_COLUMNS = INVENTORY_COLUMNS + (
    "source_group",
    "duplicate_of",
    "eligible",
    "cohort",
)
TRAINING_REGISTRY_COLUMNS = (
    "model_label",
    "ladder",
    "rung",
    "train_manifest",
    "val_manifest",
    "pool_seed",
    "optimization_seed",
    "unique_sequences",
    "training_exposure",
)
DEFAULT_POOL_SIZES = (2_500, 25_000, 250_000)
RUNG_NAMES = ("small", "medium", "large", "full")


def _packed_sequence_sha256(packed: bytes, shape: tuple[int, int, int]) -> str:
    """Hash bit-packed content together with its unambiguous tensor shape."""

    header = b"gaitlu-bitpack-v1\0" + b"\0".join(
        str(value).encode() for value in shape
    )
    return hashlib.sha256(header + b"\0" + packed).hexdigest()


def _atomic_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    with tempfile.NamedTemporaryFile(
        mode="w", newline="", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sequence_id(member_name: str) -> str:
    path = PurePosixPath(member_name)
    if path.is_absolute() or ".." in path.parts or "\\" in member_name:
        raise ValueError(f"unsafe tar member path: {member_name!r}")
    if path.suffix.casefold() != ".pkl":
        raise ValueError(f"not a pickle member: {member_name!r}")
    # OpenGait convention is label/type/view/view.pkl. The sequence is the
    # containing label/type/view path, not merely the repeated leaf filename.
    value = path.parent.as_posix() if path.parent != PurePosixPath(".") else path.stem
    if not value or value in {".", "/"}:
        raise ValueError(f"cannot derive sequence id from {member_name!r}")
    return value


def _coerce_frame_array(value) -> np.ndarray:
    if isinstance(value, dict):
        recognized = [key for key in ("silhouettes", "frames", "data") if key in value]
        if len(recognized) != 1:
            raise ValueError(
                "pickle mappings must contain exactly one of silhouettes, frames, or data"
            )
        value = value[recognized[0]]
    if isinstance(value, (list, tuple)) and len(value) == 1:
        candidate = np.asarray(value[0])
        if candidate.ndim >= 3:
            value = candidate
    array = np.asarray(value)
    if array.dtype == object:
        raise ValueError("sequence has object dtype")
    if array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0]
    elif array.ndim == 4 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 3:
        raise ValueError(f"sequence must have shape [T,H,W], got {array.shape}")
    if any(int(value) <= 0 for value in array.shape):
        raise ValueError(f"sequence has an empty dimension: {array.shape}")
    if not np.issubdtype(array.dtype, np.number) and array.dtype != np.bool_:
        raise ValueError(f"unsupported sequence dtype: {array.dtype}")
    return array


def validate_and_pack_sequence(
    value,
    *,
    min_frames=16,
    max_empty_frame_fraction=0.20,
    min_foreground_fraction=1e-4,
    max_intermediate_fraction=0.05,
):
    """Validate one decoded pickle and return a deterministic bit-packed record."""

    array = _coerce_frame_array(value)
    if array.shape[0] < int(min_frames):
        raise ValueError(f"sequence has {array.shape[0]} frames; at least {min_frames} required")
    numeric = array.astype(np.float32, copy=False)
    if not np.isfinite(numeric).all():
        raise ValueError("sequence contains non-finite pixels")
    minimum = float(numeric.min())
    maximum = float(numeric.max())
    if minimum < -1e-6:
        raise ValueError(f"sequence has negative pixels: minimum={minimum}")
    if maximum <= 1.0 + 1e-6:
        normalized = numeric
    elif maximum <= 255.0 + 1e-6:
        normalized = numeric / 255.0
    else:
        raise ValueError(f"sequence pixel maximum exceeds 255: {maximum}")
    intermediate = float(((normalized > 0.05) & (normalized < 0.95)).mean())
    if intermediate > float(max_intermediate_fraction):
        raise ValueError(
            f"sequence is not binary-looking: intermediate_fraction={intermediate:.6g}"
        )
    binary = normalized >= 0.5
    per_frame_foreground = binary.reshape(binary.shape[0], -1).mean(axis=1)
    empty_fraction = float((per_frame_foreground < float(min_foreground_fraction)).mean())
    if empty_fraction > float(max_empty_frame_fraction):
        raise ValueError(
            f"too many empty frames: empty_frame_fraction={empty_fraction:.6g}"
        )
    foreground = float(binary.mean())
    packed = np.packbits(binary.reshape(-1), bitorder="little").tobytes()
    shape = tuple(map(int, binary.shape))
    return {
        "packed": packed,
        "shape": shape,
        "content_sha256": _packed_sequence_sha256(packed, shape),
        "foreground_fraction": foreground,
        "empty_frame_fraction": empty_fraction,
        "intermediate_fraction": intermediate,
    }


def _output_stem(input_path: Path) -> str:
    name = input_path.name
    if name.endswith(".tar.gz"):
        return name[:-7]
    if name.endswith(".tgz"):
        return name[:-4]
    raise ValueError("input shard must end in .tar.gz or .tgz")


def pack_gaitlu_shard(
    input_path,
    prepared_root,
    *,
    trust_pickles=False,
    min_frames=16,
    max_empty_frame_fraction=0.20,
    min_foreground_fraction=1e-4,
    max_intermediate_fraction=0.05,
    max_pickle_bytes=256 * 1024 * 1024,
):
    """Convert one gzip pickle shard without extracting its tiny files.

    ``trust_pickles`` must be explicit because unpickling can execute arbitrary
    code. Only use this on the official, trusted GaitLU release.
    """

    if not trust_pickles:
        raise PermissionError(
            "pickle loading is unsafe; pass trust_pickles=True only for the trusted release"
        )
    input_path = Path(input_path).expanduser().resolve()
    prepared_root = Path(prepared_root).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    stem = _output_stem(input_path)
    shard_dir = prepared_root / "shards"
    inventory_dir = prepared_root / "inventories"
    shard_dir.mkdir(parents=True, exist_ok=True)
    inventory_dir.mkdir(parents=True, exist_ok=True)
    output_tar = shard_dir / f"{stem}.tar"
    output_inventory = inventory_dir / f"{stem}.csv"
    for path in (output_tar, output_inventory):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {path}")

    rows = []
    seen_ids = {}
    local_records = {}
    with tempfile.NamedTemporaryFile(
        dir=shard_dir, prefix=f".{stem}.", suffix=".tar.tmp", delete=False
    ) as handle:
        temporary_tar = Path(handle.name)
    try:
        with (
            tarfile.open(input_path, mode="r:gz") as source,
            tarfile.open(temporary_tar, mode="w", format=tarfile.USTAR_FORMAT) as target,
        ):
            members = sorted(
                (member for member in source.getmembers() if member.isfile()),
                key=lambda member: member.name,
            )
            for member in members:
                if PurePosixPath(member.name).suffix.casefold() != ".pkl":
                    continue
                sequence_id = _sequence_id(member.name)
                canonical_id = sequence_id.casefold()
                if canonical_id in seen_ids:
                    raise ValueError(
                        f"duplicate/case-colliding sequence id {sequence_id!r} in "
                        f"{member.name!r} and {seen_ids[canonical_id]!r}"
                    )
                seen_ids[canonical_id] = member.name
                base = {
                    "sequence_id": sequence_id,
                    "source_shard": input_path.name,
                    "source_member": member.name,
                    "shard_path": f"shards/{stem}.tar",
                    "record_offset": "",
                    "record_size": "",
                    "num_frames": "",
                    "height": "",
                    "width": "",
                    "content_sha256": "",
                    "foreground_fraction": "",
                    "empty_frame_fraction": "",
                    "intermediate_fraction": "",
                    "valid": "false",
                    "exclusion_reason": "",
                }
                try:
                    if member.size <= 0 or member.size > int(max_pickle_bytes):
                        raise ValueError(
                            f"pickle size {member.size} is outside (0, {max_pickle_bytes}]"
                        )
                    extracted = source.extractfile(member)
                    if extracted is None:
                        raise ValueError("tar member has no readable payload")
                    payload = extracted.read(int(max_pickle_bytes) + 1)
                    if len(payload) != member.size:
                        raise ValueError(
                            f"tar member declared {member.size} bytes but read {len(payload)}"
                        )
                    # nosec B301 -- guarded by the explicit trust_pickles contract above.
                    value = pickle.load(io.BytesIO(payload))
                    record = validate_and_pack_sequence(
                        value,
                        min_frames=min_frames,
                        max_empty_frame_fraction=max_empty_frame_fraction,
                        min_foreground_fraction=min_foreground_fraction,
                        max_intermediate_fraction=max_intermediate_fraction,
                    )
                    packed = record["packed"]
                    digest = record["content_sha256"]
                    if digest in local_records:
                        offset, size = local_records[digest]
                    else:
                        info = tarfile.TarInfo(f"records/{digest}.bits")
                        info.size = len(packed)
                        info.mtime = info.uid = info.gid = 0
                        info.uname = info.gname = ""
                        offset = target.offset + tarfile.BLOCKSIZE
                        target.addfile(info, io.BytesIO(packed))
                        size = len(packed)
                        local_records[digest] = (offset, size)
                    frames, height, width = record["shape"]
                    base.update(
                        {
                            "record_offset": offset,
                            "record_size": size,
                            "num_frames": frames,
                            "height": height,
                            "width": width,
                            "content_sha256": digest,
                            "foreground_fraction": f"{record['foreground_fraction']:.12g}",
                            "empty_frame_fraction": f"{record['empty_frame_fraction']:.12g}",
                            "intermediate_fraction": f"{record['intermediate_fraction']:.12g}",
                            "valid": "true",
                        }
                    )
                except Exception as error:  # retain every rejected sequence in the audit
                    base["exclusion_reason"] = f"{type(error).__name__}: {error}"
                rows.append(base)
        os.replace(temporary_tar, output_tar)
        _atomic_csv(output_inventory, INVENTORY_COLUMNS, rows)
    finally:
        temporary_tar.unlink(missing_ok=True)

    valid = sum(row["valid"] == "true" for row in rows)
    summary = {
        "input": str(input_path),
        "output_tar": str(output_tar),
        "inventory": str(output_inventory),
        "sequences": len(rows),
        "valid_sequences": valid,
        "excluded_sequences": len(rows) - valid,
        "unique_records_in_shard": len(local_records),
    }
    _atomic_json(inventory_dir / f"{stem}.summary.json", summary)
    return summary


def _read_inventories(prepared_root: Path):
    paths = sorted((prepared_root / "inventories").glob("gaitlu-*.csv"))
    if not paths:
        raise FileNotFoundError("no gaitlu-*.csv inventories found")
    rows = []
    for path in paths:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != INVENTORY_COLUMNS:
                raise ValueError(f"inventory has the wrong schema: {path}")
            rows.extend(dict(row) for row in reader)
    return paths, rows


def _source_groups(path: Path | None, sequence_ids):
    if path is None:
        return {sequence_id: sequence_id for sequence_id in sequence_ids}, "singleton"
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != ("sequence_id", "source_group"):
            raise ValueError("source-group CSV must contain exactly sequence_id,source_group")
        mapping = {}
        for line_no, row in enumerate(reader, start=2):
            sequence_id = row["sequence_id"]
            source_group = row["source_group"]
            if (
                not sequence_id
                or not source_group
                or sequence_id.strip() != sequence_id
                or source_group.strip() != source_group
            ):
                raise ValueError(f"invalid source-group mapping on row {line_no}")
            if sequence_id in mapping:
                raise ValueError(f"duplicate source-group sequence_id {sequence_id!r}")
            mapping[sequence_id] = source_group
    expected = set(sequence_ids)
    if set(mapping) != expected:
        raise ValueError(
            "source-group CSV coverage differs from inventory: "
            f"missing={len(expected - set(mapping))}, extra={len(set(mapping) - expected)}"
        )
    return mapping, "provided"


def _group_rows(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["source_group"], []).append(row)
    return groups


def _closest_prefix(group_names, groups, target):
    if target <= 0:
        raise ValueError("pool targets must be positive")
    chosen = []
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


def _manifest_row(row, split):
    return {
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


def _write_manifest(path, rows, split):
    selected = sorted((_manifest_row(row, split) for row in rows), key=lambda row: row["sequence_id"])
    _atomic_csv(path, GAITLU_MANIFEST_COLUMNS, selected)


def finalize_gaitlu_study(
    prepared_root,
    *,
    holdout_size=10_000,
    holdout_seed=20_260_806,
    pool_seeds=(0, 1, 2, 3, 4),
    pool_sizes=DEFAULT_POOL_SIZES,
    training_exposure=8_192_000,
    source_groups_csv=None,
    expected_shards=100,
):
    """Deduplicate inventories and create the common holdout plus five ladders."""

    prepared_root = Path(prepared_root).expanduser().resolve()
    inventory_paths, rows = _read_inventories(prepared_root)
    if len(inventory_paths) != int(expected_shards):
        raise ValueError(
            f"expected exactly {int(expected_shards)} shard inventories, "
            f"found {len(inventory_paths)}"
        )
    pool_seeds = tuple(map(int, pool_seeds))
    pool_sizes = tuple(map(int, pool_sizes))
    if len(pool_seeds) != 5 or len(set(pool_seeds)) != 5:
        raise ValueError("the scaling study requires exactly five distinct pool seeds")
    if (
        len(pool_sizes) != 3
        or any(target <= 0 for target in pool_sizes)
        or not all(left < right for left, right in zip(pool_sizes, pool_sizes[1:]))
    ):
        raise ValueError("pool_sizes must contain three positive, strictly increasing targets")
    if int(training_exposure) <= 0:
        raise ValueError("training_exposure must be positive")

    rows.sort(key=lambda row: (row["source_shard"], row["source_member"]))
    seen_ids = {}
    for row in rows:
        canonical = row["sequence_id"].casefold()
        if canonical in seen_ids:
            raise ValueError(
                f"duplicate/case-colliding sequence ids across shards: "
                f"{row['sequence_id']!r} and {seen_ids[canonical]!r}"
            )
        seen_ids[canonical] = row["sequence_id"]
    mapping, grouping_policy = _source_groups(
        Path(source_groups_csv).expanduser().resolve() if source_groups_csv else None,
        [row["sequence_id"] for row in rows],
    )
    canonical_by_hash = {}
    for row in rows:
        row["source_group"] = mapping[row["sequence_id"]]
        row["duplicate_of"] = ""
        row["eligible"] = "false"
        row["cohort"] = "excluded"
        if row["valid"] != "true":
            continue
        previous = canonical_by_hash.get(row["content_sha256"])
        if previous is not None:
            row["duplicate_of"] = previous["sequence_id"]
            row["exclusion_reason"] = "exact_content_duplicate"
            continue
        canonical_by_hash[row["content_sha256"]] = row
        row["eligible"] = "true"

    canonical_rows = [row for row in rows if row["eligible"] == "true"]
    groups = _group_rows(canonical_rows)
    group_names = sorted(groups)
    random.Random(int(holdout_seed)).shuffle(group_names)
    holdout_group_names = set(_closest_prefix(group_names, groups, int(holdout_size)))
    holdout_rows = [
        row for row in canonical_rows if row["source_group"] in holdout_group_names
    ]
    training_rows = [
        row for row in canonical_rows if row["source_group"] not in holdout_group_names
    ]
    if len(training_rows) < pool_sizes[-1]:
        raise ValueError(
            f"only {len(training_rows)} eligible training sequences remain; "
            f"large target is {pool_sizes[-1]}"
        )
    for row in holdout_rows:
        row["cohort"] = "holdout"
    for row in training_rows:
        row["cohort"] = "training"

    training_groups = _group_rows(training_rows)
    ladder_selections = []
    for ladder_index, pool_seed in enumerate(pool_seeds):
        ordered_groups = sorted(training_groups)
        random.Random(pool_seed).shuffle(ordered_groups)
        selections = []
        for target in pool_sizes:
            names = set(_closest_prefix(ordered_groups, training_groups, target))
            if selections and not selections[-1] < names:
                raise ValueError(
                    "source-group granularity cannot produce strictly nested pool rungs"
                )
            selections.append(names)
        full_names = set(ordered_groups)
        if not selections[-1] < full_names:
            raise ValueError(
                "source-group granularity cannot produce a full pool larger than the large rung"
            )
        selections.append(full_names)
        ladder_selections.append((ladder_index, pool_seed, selections))

    manifests_dir = prepared_root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    holdout_path = manifests_dir / "common-holdout.csv"
    _write_manifest(holdout_path, holdout_rows, "val")
    smoke_holdout_rows = sorted(holdout_rows, key=lambda row: row["sequence_id"])[:64]
    smoke_holdout_path = manifests_dir / "smoke-holdout.csv"
    _write_manifest(smoke_holdout_path, smoke_holdout_rows, "val")

    registry = []
    pool_summaries = []
    for ladder_index, pool_seed, selections in ladder_selections:
        ladder = f"ladder-{ladder_index}"
        for rung, names in zip(RUNG_NAMES, selections):
            selected = [
                row for row in training_rows if row["source_group"] in names
            ]
            manifest_relative = f"manifests/{ladder}-{rung}.csv"
            _write_manifest(prepared_root / manifest_relative, selected, "train")
            model_label = f"{ladder}-{rung}"
            registry.append(
                {
                    "model_label": model_label,
                    "ladder": ladder,
                    "rung": rung,
                    "train_manifest": manifest_relative,
                    "val_manifest": "manifests/common-holdout.csv",
                    "pool_seed": pool_seed,
                    "optimization_seed": pool_seed,
                    "unique_sequences": len(selected),
                    "training_exposure": int(training_exposure),
                }
            )
            pool_summaries.append(
                {
                    "ladder": ladder,
                    "rung": rung,
                    "pool_seed": pool_seed,
                    "target_sequences": None if rung == "full" else pool_sizes[RUNG_NAMES.index(rung)],
                    "actual_sequences": len(selected),
                    "manifest": manifest_relative,
                }
            )

    _atomic_csv(prepared_root / "inventory.csv", FINAL_INVENTORY_COLUMNS, rows)
    _atomic_csv(
        prepared_root / "training_registry.csv", TRAINING_REGISTRY_COLUMNS, registry
    )
    summary = {
        "version": "gaitlu-scaling-pools-v2",
        "source_inventory_count": len(inventory_paths),
        "source_sequence_count": len(rows),
        "valid_before_deduplication": sum(row["valid"] == "true" for row in rows),
        "exact_duplicates": sum(bool(row["duplicate_of"]) for row in rows),
        "invalid_or_excluded": sum(row["valid"] != "true" for row in rows),
        "source_grouping_policy": grouping_policy,
        "holdout_seed": int(holdout_seed),
        "holdout_sequences": len(holdout_rows),
        "holdout_manifest": "manifests/common-holdout.csv",
        "smoke_holdout_manifest": "manifests/smoke-holdout.csv",
        "eligible_training_sequences": len(training_rows),
        "training_exposure": int(training_exposure),
        "pools": pool_summaries,
    }
    _atomic_json(prepared_root / "study_pools.json", summary)
    return summary


__all__ = [
    "DEFAULT_POOL_SIZES",
    "FINAL_INVENTORY_COLUMNS",
    "INVENTORY_COLUMNS",
    "TRAINING_REGISTRY_COLUMNS",
    "finalize_gaitlu_study",
    "pack_gaitlu_shard",
    "validate_and_pack_sequence",
]
