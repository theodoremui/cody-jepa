"""Phase-0 reproducibility contract, artifact validation, and compact reports."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import csv
import json
import os
import tempfile

from .probes import (
    FEATURE_FORMULA,
    FEATURE_SOURCE,
    METADATA_COLUMNS,
    checkpoint_sha256,
    evaluate_all_probes,
    read_feature_table,
    validate_feature_metadata,
)
from .single_stream_jepa import CHECKPOINT_SCHEMA, MODEL_ARCHITECTURE, load_checkpoint


PROTOCOL_PATH = Path("protocols/phase0-baseline.json")
REPRODUCIBILITY_CODE_PATHS = (
    "notebooks/single-stream-jepa.ipynb",
    "scripts/export_single_stream_features.py",
    "scripts/eval_probes.py",
    "scripts/run_phase0_pipeline.py",
    "src/cody_jepa/data/__init__.py",
    "src/cody_jepa/data/dataset.py",
    "src/cody_jepa/data/healthgait.py",
    "src/cody_jepa/phase0.py",
    "src/cody_jepa/probes.py",
    "src/cody_jepa/single_stream_jepa.py",
)
VALIDATION_METRIC_KEYS = (
    "loss",
    "subject_balanced_loss",
    "effective_rank",
    "effective_rank_ratio",
    "feature_std",
    "near_zero_variance_fraction",
    "subject_balanced_context_shuffle_loss_gap",
    "representations_healthy",
    "health_issues",
)


def _resolved(path: Path, repo_root: Path) -> Path:
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _relative(path: Path, repo_root: Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def portable_path(path, repo_root) -> str:
    """Return a checkout-independent path when an artifact is inside the repo."""
    return _relative(Path(path), Path(repo_root))


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def guard_research_path(path, repo_root, *, write: bool) -> Path:
    """Reject retired paths and writes into the immutable retained baseline."""
    repo_root = Path(repo_root).resolve()
    resolved = _resolved(Path(path), repo_root)
    retired = repo_root / "outputs" / "jepa-v3"
    baseline = repo_root / "outputs" / "jepa-v4"
    if _is_within(resolved, retired):
        raise ValueError("outputs/jepa-v3 is retired and must not be recreated or used")
    if write and _is_within(resolved, baseline):
        raise ValueError("outputs/jepa-v4 is read-only baseline evidence")
    return resolved


def require_unchanged_hash(path, expected_sha256, description="artifact") -> None:
    actual = checkpoint_sha256(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"{description} changed during the operation: "
            f"before={expected_sha256}, after={actual}"
        )


def prepare_empty_directory(path, repo_root) -> Path:
    path = guard_research_path(path, repo_root, write=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.mkdir()
    except FileExistsError:
        raise FileExistsError(f"artifact directory must not already exist: {path}") from None
    return path


def load_protocol(repo_root, path=PROTOCOL_PATH) -> dict:
    repo_root = Path(repo_root).resolve()
    protocol_path = _resolved(Path(path), repo_root)
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("schema_version") != 1:
        raise ValueError("unsupported Phase-0 protocol schema")
    return protocol


def validate_manifest(protocol: Mapping, repo_root) -> dict:
    repo_root = Path(repo_root).resolve()
    contract = protocol["manifest"]
    path = _resolved(Path(contract["path"]), repo_root)
    actual_hash = checkpoint_sha256(path)
    if actual_hash != contract["sha256"]:
        raise ValueError(
            f"manifest hash drift: expected={contract['sha256']}, actual={actual_hash}"
        )
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != contract["metadata_schema"]:
            raise ValueError(
                "manifest metadata schema drift: "
                f"expected={contract['metadata_schema']}, actual={reader.fieldnames}"
            )
        rows = list(reader)
    actual_splits = {}
    subjects_by_split = {}
    for split, expected in contract["splits"].items():
        selected = [row for row in rows if row["split"] == split]
        subjects = {row["subject_id"].casefold() for row in selected}
        actual = {"sequences": len(selected), "subjects": len(subjects)}
        if actual != expected:
            raise ValueError(
                f"manifest {split} counts drift: expected={expected}, actual={actual}"
            )
        actual_splits[split] = actual
        subjects_by_split[split] = subjects
    overlap = subjects_by_split["train"] & subjects_by_split["val"]
    if overlap:
        raise ValueError("manifest train/val subjects overlap")
    return {"path": _relative(path, repo_root), "sha256": actual_hash, "splits": actual_splits}


def checkpoint_record(path, protocol: Mapping | None = None, expected_name=None) -> dict:
    path = Path(path).resolve()
    digest = checkpoint_sha256(path)
    checkpoint = load_checkpoint(path)
    require_unchanged_hash(path, digest, "checkpoint")
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(f"unsupported checkpoint schema in {path}")
    if checkpoint.get("architecture") != MODEL_ARCHITECTURE:
        raise ValueError(f"unsupported checkpoint architecture in {path}")
    if protocol is not None:
        if protocol.get("checkpoint_schema") != CHECKPOINT_SCHEMA:
            raise ValueError(
                "frozen protocol checkpoint schema does not match the supported schema"
            )
        if protocol.get("checkpoint_architecture") != MODEL_ARCHITECTURE:
            raise ValueError(
                "frozen protocol checkpoint architecture does not match the supported architecture"
            )
    completed_epochs = checkpoint.get("completed_epochs")
    global_step = checkpoint.get("global_step")
    if not isinstance(completed_epochs, int) or completed_epochs <= 0:
        raise ValueError(f"checkpoint is not at a completed epoch boundary: {path}")
    if not isinstance(global_step, int) or global_step <= 0:
        raise ValueError(f"checkpoint has no completed optimizer steps: {path}")
    data_contract = checkpoint.get("data_contract")
    if not isinstance(data_contract, Mapping):
        raise ValueError(f"checkpoint has no data contract: {path}")
    train = data_contract.get("train_dataset", {})
    val = data_contract.get("val_dataset", {})
    if train.get("manifest_sha256") != val.get("manifest_sha256"):
        raise ValueError("checkpoint train/val manifest hashes differ")
    if protocol is not None:
        name = expected_name or path.name
        expected_record = protocol["candidate_checkpoints"].get(name)
        if expected_record is None:
            raise ValueError(f"checkpoint {name!r} is not in the frozen baseline contract")
        expected = {
            key: expected_record[key]
            for key in ("sha256", "completed_epochs", "global_step")
        }
        actual = {
            "sha256": digest,
            "completed_epochs": completed_epochs,
            "global_step": global_step,
        }
        if actual != expected:
            raise ValueError(
                f"baseline checkpoint drift for {name}: expected={expected}, actual={actual}"
            )
        if train.get("manifest_sha256") != protocol["manifest"]["sha256"]:
            raise ValueError(f"checkpoint {name} does not use the frozen manifest")
        for split, dataset in (("train", train), ("val", val)):
            expected_count = protocol["manifest"]["splits"][split]["sequences"]
            if dataset.get("sequence_count") != expected_count:
                raise ValueError(f"checkpoint {name} {split} sequence count drift")
    history = checkpoint.get("history", [])
    validation = {}
    if history and isinstance(history[-1], Mapping):
        final_history = history[-1]
        if final_history.get("epoch") == completed_epochs:
            final_validation = final_history.get("val")
            if isinstance(final_validation, Mapping):
                validation = {
                    key: final_validation[key]
                    for key in VALIDATION_METRIC_KEYS
                    if key in final_validation
                }
    return {
        "filename": expected_name or path.name,
        "sha256": digest,
        "identifier": f"sha256:{digest}",
        "completed_epochs": completed_epochs,
        "global_step": global_step,
        "best_epoch": checkpoint.get("best_epoch"),
        "best_val_loss": checkpoint.get("best_val_loss"),
        "best_healthy_epoch": checkpoint.get("best_healthy_epoch"),
        "manifest_sha256": train.get("manifest_sha256"),
        "validation": validation,
    }


def validate_completed_run(latest_path) -> dict:
    """Prove training reached its declared epoch or optimizer-step boundary."""
    latest_path = Path(latest_path).resolve()
    digest = checkpoint_sha256(latest_path)
    checkpoint = load_checkpoint(latest_path)
    record = checkpoint_record(latest_path)
    if record["sha256"] != digest:
        raise RuntimeError("latest checkpoint changed during completed-run validation")
    require_unchanged_hash(latest_path, digest, "latest checkpoint")
    config = checkpoint.get("config", {})
    by_epoch = record["completed_epochs"] >= int(config.get("num_epochs", 0)) > 0
    by_step = record["global_step"] >= int(config.get("steps", 0)) > 0
    if not (by_epoch or by_step):
        raise ValueError(f"latest checkpoint does not represent a completed run: {latest_path}")
    history = checkpoint.get("history", [])
    if history:
        final = history[-1]
        if final.get("epoch") != record["completed_epochs"]:
            raise ValueError("latest checkpoint history does not match completed_epochs")
    return record


def _require_equal(actual, expected, description):
    if actual != expected:
        raise ValueError(f"{description} mismatch: expected={expected!r}, actual={actual!r}")


def validate_candidate_artifacts(
    checkpoint_path, feature_path, probe_json_path, protocol: Mapping, candidate_name
) -> dict:
    checkpoint = checkpoint_record(checkpoint_path, protocol, candidate_name)
    feature_path = Path(feature_path).resolve()
    feature_hash = checkpoint_sha256(feature_path)
    sidecar_path = feature_path.with_suffix(feature_path.suffix + ".metadata.json")
    if not sidecar_path.is_file():
        raise ValueError(f"feature metadata sidecar is required for {feature_path}")
    sidecar_hash = checkpoint_sha256(sidecar_path)
    table, metadata = read_feature_table(feature_path)
    validate_feature_metadata(table, feature_path, metadata)
    feature_contract = protocol["feature_export"]
    _require_equal(metadata["checkpoint_sha256"], checkpoint["sha256"], "feature checkpoint hash")
    _require_equal(metadata["feature_source"], FEATURE_SOURCE, "feature source")
    _require_equal(metadata["feature_formula"], FEATURE_FORMULA, "feature formula")
    _require_equal(metadata.get("windows_per_sequence"), feature_contract["windows_per_sequence"], "window count")
    _require_equal(metadata.get("window_policy"), feature_contract["window_policy"], "window policy")
    _require_equal(metadata.get("preprocessing"), feature_contract["preprocessing"], "preprocessing")
    _require_equal(list(METADATA_COLUMNS), feature_contract["metadata_schema"], "feature metadata schema")
    row_counts = {
        split: int((table["split"].astype(str) == split).sum())
        for split in ("train", "val")
    }
    _require_equal(row_counts, feature_contract["expected_rows"], "feature row counts")
    signatures = metadata.get("dataset_signatures", {})
    for split in ("train", "val"):
        _require_equal(
            signatures.get(split, {}).get("manifest_sha256"),
            protocol["manifest"]["sha256"],
            f"{split} feature manifest hash",
        )
        _require_equal(
            signatures.get(split, {}).get("sequence_count"),
            protocol["manifest"]["splits"][split]["sequences"],
            f"{split} feature sequence count",
        )
        _require_equal(
            signatures.get(split, {}).get("inventory_sha256"),
            protocol["manifest"]["sampled_inventory_sha256"],
            f"{split} sampled inventory hash",
        )
    _require_equal(
        metadata.get("device"), feature_contract["reference_device"], "feature device"
    )

    probe_json_path = Path(probe_json_path).resolve()
    probe_hash = checkpoint_sha256(probe_json_path)
    probe = json.loads(probe_json_path.read_text())
    probe_contract = protocol["probes"]
    for key in (
        "seed",
        "max_iter",
        "identity_validation_fraction",
        "retrieval_enrollment_sequences",
    ):
        _require_equal(probe.get(key), probe_contract[key], f"probe {key}")
    _require_equal(probe.get("feature_table_sha256"), metadata["feature_table_sha256"], "probe feature hash")
    _require_equal(probe.get("feature_metadata_sha256"), sidecar_hash, "probe feature metadata hash")
    _require_equal(probe.get("checkpoint_sha256"), checkpoint["sha256"], "probe checkpoint hash")
    _require_equal(probe.get("feature_source"), FEATURE_SOURCE, "probe feature source")
    _require_equal(probe.get("feature_formula"), FEATURE_FORMULA, "probe feature formula")
    results = probe.get("results")
    if not isinstance(results, list):
        raise ValueError("probe results must be a list")
    tasks = [result.get("task") for result in results]
    _require_equal(tasks, probe_contract["tasks"], "probe task order")
    reproduced = evaluate_all_probes(
        table,
        feature_source=FEATURE_SOURCE,
        validation_fraction=probe_contract["identity_validation_fraction"],
        enrollment_sequences=probe_contract["retrieval_enrollment_sequences"],
        max_iter=probe_contract["max_iter"],
        seed=probe_contract["seed"],
    )
    _require_equal(results, reproduced, "recomputed probe results")
    require_unchanged_hash(checkpoint_path, checkpoint["sha256"], "checkpoint")
    require_unchanged_hash(feature_path, feature_hash, "feature table")
    require_unchanged_hash(sidecar_path, sidecar_hash, "feature metadata")
    require_unchanged_hash(probe_json_path, probe_hash, "probe report")
    metrics = {
        result["task"]: {
            key: result[key]
            for key in ("majority_baseline", "accuracy", "balanced_accuracy", "macro_f1")
        }
        for result in results
    }
    record = {
        "checkpoint": checkpoint,
        "features": {
            "path": str(feature_path),
            "sha256": feature_hash,
            "metadata_sha256": sidecar_hash,
            "rows": row_counts,
            "feature_dim": metadata["feature_dim"],
            "device": metadata.get("device"),
        },
        "probes": {
            "json_path": str(probe_json_path),
            "json_sha256": probe_hash,
            "metrics": metrics,
        },
    }
    expected_artifacts = protocol["candidate_checkpoints"][candidate_name].get("artifacts")
    if expected_artifacts is not None:
        actual_artifacts = {
            "feature_sha256": record["features"]["sha256"],
            "feature_metadata_sha256": record["features"]["metadata_sha256"],
            "probe_json_sha256": record["probes"]["json_sha256"],
        }
        _require_equal(actual_artifacts, expected_artifacts, f"{candidate_name} locked artifacts")
    return record


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(text)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_baseline_report(repo_root, artifact_dir, report_path, protocol_path=PROTOCOL_PATH):
    repo_root = Path(repo_root).resolve()
    protocol = load_protocol(repo_root, protocol_path)
    manifest = validate_manifest(protocol, repo_root)
    baseline_dir = _resolved(Path(protocol["read_only_baseline_directory"]), repo_root)
    validate_completed_run(baseline_dir / "latest.pt")
    artifact_dir = guard_research_path(artifact_dir, repo_root, write=True)
    candidates = {}
    for filename in protocol["candidate_checkpoints"]:
        label = Path(filename).stem
        candidate_dir = artifact_dir / label
        record = validate_candidate_artifacts(
            baseline_dir / filename,
            candidate_dir / "features.npz",
            candidate_dir / "probes" / "probe_metrics.json",
            protocol,
            filename,
        )
        for section in ("features", "probes"):
            for key in ("path", "json_path"):
                if key in record[section]:
                    record[section][key] = _relative(Path(record[section][key]), repo_root)
        candidates[filename] = record
    canonical_name = protocol["canonical_checkpoint"]["filename"]
    payload = {
        "schema_version": 1,
        "status": "locked",
        "baseline_job_id": protocol["baseline_job_id"],
        "canonical_checkpoint": protocol["canonical_checkpoint"],
        "manifest": {
            **manifest,
            "metadata_schema": protocol["manifest"]["metadata_schema"],
            "sampled_inventory_sha256": protocol["manifest"]["sampled_inventory_sha256"],
            "inventory_hash_scope": protocol["manifest"]["inventory_hash_scope"],
        },
        "feature_export": protocol["feature_export"],
        "probe_protocol": protocol["probes"],
        "candidates": candidates,
        "reproducibility": {
            "protocol_path": _relative(_resolved(Path(protocol_path), repo_root), repo_root),
            "protocol_sha256": checkpoint_sha256(_resolved(Path(protocol_path), repo_root)),
            "uv_lock_sha256": checkpoint_sha256(repo_root / "uv.lock"),
            "code_sha256": {
                path: checkpoint_sha256(repo_root / path)
                for path in REPRODUCIBILITY_CODE_PATHS
            },
        },
    }
    if candidates[canonical_name]["checkpoint"]["identifier"] != protocol["canonical_checkpoint"]["identifier"]:
        raise ValueError("canonical checkpoint identifier drift")
    report_path = guard_research_path(report_path, repo_root, write=True)
    if report_path.suffix.casefold() != ".md":
        raise ValueError("baseline report path must end in .md")
    json_path = report_path.with_suffix(".json")
    write_text_atomic(json_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Phase 0 baseline report",
        "",
        "**Status:** Locked. Job 91108 and `outputs/jepa-v4/` remain read-only evidence.",
        "",
        f"**Canonical checkpoint:** `{canonical_name}` (`{protocol['canonical_checkpoint']['identifier']}`).",
        "It is locked by the training-time subject-balanced validation-loss rule, independent of probe scores.",
        "",
        "## Frozen evaluation protocol",
        "",
        f"- Manifest SHA-256: `{manifest['sha256']}`",
        f"- Manifest schema: `{', '.join(protocol['manifest']['metadata_schema'])}`",
        f"- Split counts: train {manifest['splits']['train']['sequences']} sequences / {manifest['splits']['train']['subjects']} subjects; val {manifest['splits']['val']['sequences']} sequences / {manifest['splits']['val']['subjects']} subjects",
        f"- Feature formula: `{protocol['feature_export']['formula']}`",
        f"- Reference export runtime: device `{protocol['feature_export']['reference_device']}`, batch size {protocol['feature_export']['reference_batch_size']}, loader workers {protocol['feature_export']['reference_num_workers']}",
        f"- Feature rows: train {protocol['feature_export']['expected_rows']['train']}; val {protocol['feature_export']['expected_rows']['val']} ({protocol['feature_export']['windows_per_sequence']} deterministic windows per sequence)",
        f"- Probe seed: {protocol['probes']['seed']}; max iterations: {protocol['probes']['max_iter']}; closed-set validation fraction: {protocol['probes']['identity_validation_fraction']}; retrieval enrollment sequences: {protocol['probes']['retrieval_enrollment_sequences']}",
        f"- Sampled inventory SHA-256: `{protocol['manifest']['sampled_inventory_sha256']}`. Scope: {protocol['manifest']['inventory_hash_scope']}",
        "",
        "## Checkpoint comparison",
        "",
        "| Checkpoint | Epoch | Checkpoint SHA-256 | Feature SHA-256 | Effective rank | Rank ratio | Wrong-context gap | Closed-set identity | Held-out retrieval | Gait balanced accuracy |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for filename, record in candidates.items():
        metrics = record["probes"]["metrics"]
        validation = record["checkpoint"]["validation"]
        required_validation = {
            "effective_rank",
            "effective_rank_ratio",
            "subject_balanced_context_shuffle_loss_gap",
        }
        missing_validation = sorted(required_validation - validation.keys())
        if missing_validation:
            raise ValueError(
                f"checkpoint {filename} is missing validation metrics: "
                f"{', '.join(missing_validation)}"
            )
        marker = " (canonical)" if filename == canonical_name else ""
        lines.append(
            f"| `{filename}`{marker} | {record['checkpoint']['completed_epochs']} | "
            f"`{record['checkpoint']['sha256']}` | `{record['features']['sha256']}` | "
            f"{float(validation['effective_rank']):.2f} | "
            f"{float(validation['effective_rank_ratio']):.4f} | "
            f"{float(validation['subject_balanced_context_shuffle_loss_gap']):.6f} | "
            f"{metrics['identity_closed_set']['accuracy']:.4f} | "
            f"{metrics['identity_heldout_retrieval']['accuracy']:.4f} | "
            f"{metrics['gait_system']['balanced_accuracy']:.4f} |"
        )
    lines.extend([
        "",
        "Probe metrics are computed per exported clip window. Gait balanced accuracy is class-balanced, not subject-balanced. All three probes were rerun from distinct clean feature exports under current code. Full metrics, artifact paths, hashes, and the machine-readable frozen contract are in the adjacent JSON report.",
        "",
    ])
    write_text_atomic(report_path, "\n".join(lines))
    return {"markdown": report_path, "json": json_path, "payload": payload}


__all__ = [
    "PROTOCOL_PATH",
    "REPRODUCIBILITY_CODE_PATHS",
    "build_baseline_report",
    "checkpoint_record",
    "guard_research_path",
    "load_protocol",
    "prepare_empty_directory",
    "portable_path",
    "require_unchanged_hash",
    "validate_candidate_artifacts",
    "validate_completed_run",
    "validate_manifest",
    "write_text_atomic",
]
