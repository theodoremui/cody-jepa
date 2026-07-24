"""Phase-0 reproducibility contract, artifact validation, and compact reports."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import csv
import json
import math
import os
import uuid

from .probes import (
    FEATURE_FORMULA,
    FEATURE_SOURCE,
    METADATA_COLUMNS,
    checkpoint_sha256,
    evaluate_all_probes,
    read_feature_table,
    validate_feature_metadata,
)
from .single_stream_jepa import (
    CHECKPOINT_SCHEMA,
    LEGACY_CHECKPOINT_SCHEMA,
    MODEL_ARCHITECTURE,
    checkpoint_model_state_sha256,
    load_checkpoint,
    representation_health,
)


PROTOCOL_PATH = Path("protocols/phase0-baseline.json")
READ_ONLY_BASELINE_PATH = Path("outputs/jepa-v4")
RETIRED_BASELINE_PATH = Path("outputs/jepa-v3")
RETAINED_JOB_NOTEBOOK_PATH = Path("haic-results/job_91108.ipynb")
REQUIRED_READ_ONLY_EVIDENCE_PATHS = frozenset(
    {
        RETAINED_JOB_NOTEBOOK_PATH.as_posix(),
        (READ_ONLY_BASELINE_PATH / "probe_metrics.csv").as_posix(),
        (READ_ONLY_BASELINE_PATH / "probe_metrics.json").as_posix(),
        (READ_ONLY_BASELINE_PATH / "best_loss.pt").as_posix(),
        (READ_ONLY_BASELINE_PATH / "latest.pt").as_posix(),
    }
)
REPRODUCIBILITY_CODE_PATHS = (
    "notebooks/single-stream-jepa.ipynb",
    "pyproject.toml",
    "scripts/export_single_stream_features.py",
    "scripts/eval_probes.py",
    "scripts/run_phase0_pipeline.py",
    "slurm/train-single-stream-jepa.sbatch",
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
LOCKED_LEGACY_CHECKPOINT_SHA256 = {
    "best_loss.pt": "ab1e24043b2ba453e03fa427b0e845b74b2771682220732267d966be360097a5",
    "latest.pt": "5571a59c045dab3d4fd87d57e0baa296ad13f28992ab6d32f425f9340a848dad",
}
PHASE0_CANDIDATE_FILENAMES = frozenset({"best_loss.pt", "latest.pt"})


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
    """Reject retired paths and writes into immutable retained evidence."""
    repo_root = Path(repo_root).resolve()
    requested = Path(path).expanduser()
    lexical = Path(
        os.path.abspath(requested if requested.is_absolute() else repo_root / requested)
    )
    resolved = _resolved(requested, repo_root)
    retired = repo_root / RETIRED_BASELINE_PATH
    baseline = repo_root / READ_ONLY_BASELINE_PATH
    retained_notebook = repo_root / RETAINED_JOB_NOTEBOOK_PATH
    if _is_within(resolved, retired):
        raise ValueError("outputs/jepa-v3 is retired and must not be recreated or used")
    if write and _is_within(resolved, baseline):
        raise ValueError("outputs/jepa-v4 is read-only baseline evidence")
    if write and (
        lexical == retained_notebook or resolved == retained_notebook.resolve()
    ):
        raise ValueError("retained HAIC job notebook is read-only evidence")
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
    if protocol.get("schema_version") != 2:
        raise ValueError("unsupported Phase-0 protocol schema")
    if protocol.get("read_only_baseline_directory") != READ_ONLY_BASELINE_PATH.as_posix():
        raise ValueError("Phase-0 protocol read-only baseline directory drift")
    if protocol.get("retired_baseline_directories") != [
        RETIRED_BASELINE_PATH.as_posix()
    ]:
        raise ValueError("Phase-0 protocol retired baseline declaration drift")
    evidence = protocol.get("read_only_evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("Phase-0 protocol requires a read-only evidence contract")
    if any(not isinstance(path, str) for path in evidence):
        raise ValueError("read-only evidence paths must be strings")
    evidence_paths = set(evidence)
    missing = sorted(REQUIRED_READ_ONLY_EVIDENCE_PATHS - evidence_paths)
    unexpected = sorted(evidence_paths - REQUIRED_READ_ONLY_EVIDENCE_PATHS)
    if missing or unexpected:
        raise ValueError(
            "Phase-0 protocol read-only evidence path mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    candidates = protocol.get("candidate_checkpoints")
    if not isinstance(candidates, Mapping) or set(candidates) != PHASE0_CANDIDATE_FILENAMES:
        raise ValueError(
            "Phase-0 protocol candidate checkpoints must be exactly "
            f"{sorted(PHASE0_CANDIDATE_FILENAMES)}"
        )
    for filename in candidates:
        candidate = Path(filename)
        if candidate.is_absolute() or len(candidate.parts) != 1 or candidate.name != filename:
            raise ValueError(
                f"Phase-0 candidate checkpoint must be a direct filename: {filename!r}"
            )
    canonical = protocol.get("canonical_checkpoint")
    if not isinstance(canonical, Mapping) or canonical.get("filename") != "best_loss.pt":
        raise ValueError("Phase-0 canonical checkpoint must be best_loss.pt")
    return protocol


def validate_read_only_evidence(protocol: Mapping, repo_root) -> dict:
    """Verify immutable local evidence by canonical path and locked digest."""
    repo_root = Path(repo_root).resolve()
    retired_path = repo_root / RETIRED_BASELINE_PATH
    if os.path.lexists(retired_path):
        raise ValueError(
            "outputs/jepa-v3 is retired and must remain lexically absent, including symlinks"
        )
    evidence = protocol.get("read_only_evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("Phase-0 protocol requires a read-only evidence contract")
    if any(not isinstance(path, str) for path in evidence):
        raise ValueError("read-only evidence paths must be strings")
    evidence_paths = set(evidence)
    missing = sorted(REQUIRED_READ_ONLY_EVIDENCE_PATHS - evidence_paths)
    unexpected = sorted(evidence_paths - REQUIRED_READ_ONLY_EVIDENCE_PATHS)
    if missing or unexpected:
        raise ValueError(
            "Phase-0 protocol read-only evidence path mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )

    verified = {}
    for raw_path, contract in evidence.items():
        if not isinstance(raw_path, str):
            raise ValueError("read-only evidence paths must be strings")
        relative_path = Path(raw_path)
        if (
            relative_path.is_absolute()
            or raw_path != relative_path.as_posix()
            or any(part in {"", ".", ".."} for part in relative_path.parts)
        ):
            raise ValueError(
                f"read-only evidence path must be canonical and repository-relative: {raw_path!r}"
            )
        if not isinstance(contract, Mapping):
            raise ValueError(f"read-only evidence contract must be a mapping: {raw_path}")
        expected_hash = contract.get("sha256")
        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or expected_hash != expected_hash.casefold()
            or any(character not in "0123456789abcdef" for character in expected_hash)
        ):
            raise ValueError(f"invalid read-only evidence SHA-256: {raw_path}")
        role = contract.get("role")
        if not isinstance(role, str) or not role.strip():
            raise ValueError(f"read-only evidence role is required: {raw_path}")

        lexical_path = repo_root / relative_path
        if not lexical_path.is_file():
            raise ValueError(f"read-only evidence file is missing: {raw_path}")
        resolved_path = lexical_path.resolve(strict=True)
        if resolved_path != lexical_path:
            raise ValueError(f"read-only evidence path must not use symlinks: {raw_path}")
        actual_hash = checkpoint_sha256(resolved_path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"read-only evidence hash drift for {raw_path}: "
                f"expected={expected_hash}, actual={actual_hash}"
            )
        verified[raw_path] = {
            "sha256": actual_hash,
            "role": role,
        }

    for filename, checkpoint_contract in protocol.get("candidate_checkpoints", {}).items():
        evidence_path = (READ_ONLY_BASELINE_PATH / filename).as_posix()
        if evidence_path in evidence:
            _require_equal(
                evidence[evidence_path]["sha256"],
                checkpoint_contract.get("sha256"),
                f"{filename} evidence and candidate hashes",
            )
    return verified


def _baseline_checkpoint_path(baseline_dir, filename) -> Path:
    """Return a direct, non-symlink child of the retained baseline directory."""
    baseline_dir = Path(baseline_dir).resolve(strict=True)
    filename_path = Path(filename)
    if (
        filename_path.is_absolute()
        or len(filename_path.parts) != 1
        or filename_path.name != filename
        or filename not in PHASE0_CANDIDATE_FILENAMES
    ):
        raise ValueError(f"invalid Phase-0 candidate checkpoint path: {filename!r}")
    lexical = baseline_dir / filename
    resolved = lexical.resolve(strict=True)
    if resolved != lexical or resolved.parent != baseline_dir:
        raise ValueError(
            f"Phase-0 candidate checkpoint must be a direct non-symlink child: {filename}"
        )
    return resolved


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


def _validate_model_commitments(checkpoint: Mapping, model_state_sha256: str) -> None:
    for epoch_key, commitment_key in (
        ("best_epoch", "best_loss_model_state_sha256"),
        ("best_healthy_epoch", "best_healthy_model_state_sha256"),
    ):
        selected_epoch = checkpoint.get(epoch_key)
        commitment = checkpoint.get(commitment_key)
        if selected_epoch is None:
            if commitment is not None:
                raise ValueError(f"checkpoint {commitment_key} is unselected")
            continue
        if (
            not isinstance(commitment, str)
            or len(commitment) != 64
            or any(character not in "0123456789abcdef" for character in commitment)
        ):
            raise ValueError(f"checkpoint {commitment_key} is invalid")
        if (
            selected_epoch == checkpoint.get("completed_epochs")
            and commitment != model_state_sha256
        ):
            raise ValueError(f"checkpoint {commitment_key} does not commit current state")


def checkpoint_record(path, protocol: Mapping | None = None, expected_name=None) -> dict:
    path = Path(path).resolve()
    digest = checkpoint_sha256(path)
    checkpoint = load_checkpoint(path)
    require_unchanged_hash(path, digest, "checkpoint")
    expected_schema = (
        protocol.get("checkpoint_schema") if protocol is not None else CHECKPOINT_SCHEMA
    )
    if checkpoint.get("schema") != expected_schema:
        if protocol is not None:
            raise ValueError(
                "frozen protocol checkpoint schema does not match the checkpoint"
            )
        raise ValueError(f"unsupported checkpoint schema in {path}")
    if expected_schema == LEGACY_CHECKPOINT_SCHEMA and protocol is None:
        raise ValueError("legacy checkpoints require the locked Phase-0 protocol")
    if expected_schema not in {LEGACY_CHECKPOINT_SCHEMA, CHECKPOINT_SCHEMA}:
        raise ValueError(f"unsupported checkpoint schema in {path}")
    if checkpoint.get("architecture") != MODEL_ARCHITECTURE:
        raise ValueError(f"unsupported checkpoint architecture in {path}")
    if protocol is not None:
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
        if expected_schema == LEGACY_CHECKPOINT_SCHEMA:
            _require_equal(
                digest,
                LOCKED_LEGACY_CHECKPOINT_SHA256.get(name),
                f"locked legacy checkpoint hash for {name}",
            )
        if train.get("manifest_sha256") != protocol["manifest"]["sha256"]:
            raise ValueError(f"checkpoint {name} does not use the frozen manifest")
        for split, dataset in (("train", train), ("val", val)):
            expected_count = protocol["manifest"]["splits"][split]["sequences"]
            if dataset.get("sequence_count") != expected_count:
                raise ValueError(f"checkpoint {name} {split} sequence count drift")
    model_state_sha256 = checkpoint.get("model_state_sha256")
    if expected_schema == CHECKPOINT_SCHEMA:
        actual_model_state_sha256 = checkpoint_model_state_sha256(checkpoint)
        _require_equal(
            model_state_sha256,
            actual_model_state_sha256,
            "checkpoint model-state fingerprint",
        )
        _validate_model_commitments(checkpoint, actual_model_state_sha256)
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
    record = {
        "filename": expected_name or path.name,
        "schema": expected_schema,
        "sha256": digest,
        "identifier": f"sha256:{digest}",
        "completed_epochs": completed_epochs,
        "global_step": global_step,
        "best_epoch": checkpoint.get("best_epoch"),
        "best_val_loss": checkpoint.get("best_val_loss"),
        "best_healthy_epoch": checkpoint.get("best_healthy_epoch"),
        "best_loss_model_state_sha256": checkpoint.get(
            "best_loss_model_state_sha256"
        ),
        "best_healthy_model_state_sha256": checkpoint.get(
            "best_healthy_model_state_sha256"
        ),
        "manifest_sha256": train.get("manifest_sha256"),
        "model_state_sha256": model_state_sha256,
        "validation": validation,
    }
    require_unchanged_hash(path, digest, "checkpoint")
    return record


def _require_positive_int(value, description):
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{description} must be a positive integer")
    return value


def _validate_history_and_selection(checkpoint: Mapping, *, require_completion: bool):
    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint has no training configuration")
    num_epochs = _require_positive_int(config.get("num_epochs"), "num_epochs")
    configured_steps = _require_positive_int(config.get("steps"), "steps")
    completed_epochs = _require_positive_int(
        checkpoint.get("completed_epochs"), "completed_epochs"
    )
    global_step = _require_positive_int(checkpoint.get("global_step"), "global_step")
    if completed_epochs > num_epochs:
        raise ValueError("checkpoint completed_epochs exceeds configured num_epochs")
    if global_step > configured_steps:
        raise ValueError("checkpoint global_step exceeds configured steps")

    history = checkpoint.get("history")
    if not isinstance(history, list) or len(history) != completed_epochs:
        raise ValueError("checkpoint history length does not match completed_epochs")
    updates_per_epoch = None
    previous_step = 0
    for epoch, row in enumerate(history, start=1):
        if not isinstance(row, Mapping) or row.get("epoch") != epoch:
            raise ValueError("checkpoint history does not map one-to-one to epochs")
        step = _require_positive_int(row.get("step"), f"history epoch {epoch} step")
        if updates_per_epoch is None:
            updates_per_epoch = step
        epoch_updates = step - previous_step
        if epoch_updates != updates_per_epoch:
            raise ValueError("checkpoint history steps are not epoch-boundary exact")
        previous_step = step
    if history[-1]["step"] != global_step:
        raise ValueError("checkpoint history final step does not match global_step")
    if configured_steps % updates_per_epoch:
        raise ValueError("configured steps are not epoch-boundary exact")
    if require_completion and not (
        completed_epochs == num_epochs or global_step == configured_steps
    ):
        raise ValueError("checkpoint does not represent a completed run")

    selection_metric = config.get("selection_metric", "subject_balanced_loss")
    if selection_metric not in {"loss", "subject_balanced_loss"}:
        raise ValueError("checkpoint selection metric is not a supported loss metric")
    best = None
    best_healthy = None
    for row in history:
        validation = row.get("val")
        if validation is None:
            continue
        if not isinstance(validation, Mapping):
            raise ValueError("checkpoint validation history must be mappings or null")
        value = validation.get(selection_metric)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise ValueError("checkpoint selection metric must be finite")
        try:
            health = representation_health(validation, config)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("checkpoint validation health metrics are incomplete") from error
        _require_equal(
            validation.get("representations_healthy"),
            health["representations_healthy"],
            f"epoch {row['epoch']} representation health",
        )
        _require_equal(
            validation.get("health_issues"),
            health["health_issues"],
            f"epoch {row['epoch']} health issues",
        )
        candidate = (float(value), row["epoch"])
        if best is None or candidate[0] < best[0]:
            best = candidate
        if health["representations_healthy"] and (
            best_healthy is None or candidate[0] < best_healthy[0]
        ):
            best_healthy = candidate
    if best is None:
        raise ValueError("checkpoint history contains no validation selection metric")
    _require_equal(checkpoint.get("best_val_loss"), best[0], "best validation loss")
    _require_equal(checkpoint.get("best_epoch"), best[1], "best loss epoch")
    if best_healthy is None:
        _require_equal(
            checkpoint.get("best_healthy_epoch"), None, "best healthy epoch"
        )
        healthy_loss = checkpoint.get("best_healthy_val_loss")
        if not isinstance(healthy_loss, (int, float)) or not math.isinf(healthy_loss):
            raise ValueError("best healthy validation loss must be infinity when unselected")
    else:
        _require_equal(
            checkpoint.get("best_healthy_val_loss"),
            best_healthy[0],
            "best healthy validation loss",
        )
        _require_equal(
            checkpoint.get("best_healthy_epoch"),
            best_healthy[1],
            "best healthy epoch",
        )
    return {
        "best": best,
        "best_healthy": best_healthy,
        "updates_per_epoch": updates_per_epoch,
    }


def validate_completed_run(latest_path, protocol: Mapping | None = None) -> dict:
    """Prove training reached its declared epoch or optimizer-step boundary."""
    requested_name = Path(latest_path).name
    latest_path = Path(latest_path).resolve()
    if requested_name != "latest.pt" or latest_path.name != "latest.pt":
        raise ValueError("completed-run checkpoint must be named latest.pt")
    digest = checkpoint_sha256(latest_path)
    checkpoint = load_checkpoint(latest_path)
    record = checkpoint_record(
        latest_path, protocol, "latest.pt" if protocol is not None else None
    )
    if record["sha256"] != digest:
        raise RuntimeError("latest checkpoint changed during completed-run validation")
    require_unchanged_hash(latest_path, digest, "latest checkpoint")
    _validate_history_and_selection(checkpoint, require_completion=True)
    if checkpoint.get("schema") == CHECKPOINT_SCHEMA:
        best_commitment = checkpoint.get("best_loss_model_state_sha256")
        if (
            not isinstance(best_commitment, str)
            or len(best_commitment) != 64
            or any(character not in "0123456789abcdef" for character in best_commitment)
        ):
            raise ValueError("latest checkpoint lacks a best-model commitment")
        healthy_commitment = checkpoint.get("best_healthy_model_state_sha256")
        if checkpoint.get("best_healthy_epoch") is None:
            if healthy_commitment is not None:
                raise ValueError("latest checkpoint has an unselected healthy commitment")
        elif (
            not isinstance(healthy_commitment, str)
            or len(healthy_commitment) != 64
            or any(character not in "0123456789abcdef" for character in healthy_commitment)
        ):
            raise ValueError("latest checkpoint lacks a best-healthy-model commitment")
    require_unchanged_hash(latest_path, digest, "latest checkpoint")
    return record


def validate_checkpoint_from_completed_run(
    selected_path, latest_path, protocol: Mapping | None = None
) -> dict:
    """Prove a selected checkpoint was produced by the declared completed run.

    The terminal checkpoint is the authority for run completion and checkpoint
    selection. Exact contract and history-prefix equality prevent an unrelated
    checkpoint from being paired with a convenient ``latest.pt``.
    """
    selected_name = Path(selected_path).name
    latest_name = Path(latest_path).name
    selected_path = Path(selected_path).resolve()
    latest_path = Path(latest_path).resolve()
    allowed_names = {"best_loss.pt", "best_healthy.pt", "latest.pt"}
    if selected_name not in allowed_names:
        raise ValueError("selected checkpoint has an unsupported filename")
    if latest_name != "latest.pt" or latest_path.name != "latest.pt":
        raise ValueError("completed-run checkpoint must be named latest.pt")
    if selected_name == "latest.pt" and selected_path != latest_path:
        raise ValueError("selected latest.pt must be the completed-run checkpoint path")
    if selected_name != "latest.pt" and selected_path == latest_path:
        raise ValueError("selected best checkpoint must be distinct from latest.pt")
    initial_hashes = {
        selected_path: checkpoint_sha256(selected_path),
        latest_path: checkpoint_sha256(latest_path),
    }
    selected_state = load_checkpoint(selected_path)
    latest_state = (
        selected_state if selected_path == latest_path else load_checkpoint(latest_path)
    )
    selected = checkpoint_record(
        selected_path, protocol, selected_name if protocol is not None else None
    )
    completed_run = validate_completed_run(latest_path, protocol)

    if (
        not isinstance(selected_state.get("mask_groups"), list)
        or not selected_state["mask_groups"]
    ):
        raise ValueError("selected checkpoint has no mask-group contract")
    if (
        not isinstance(latest_state.get("mask_groups"), list)
        or not latest_state["mask_groups"]
    ):
        raise ValueError("completed-run checkpoint has no mask-group contract")

    for key, description in (
        ("schema", "checkpoint schema"),
        ("architecture", "checkpoint architecture"),
        ("config", "checkpoint configuration"),
        ("mask_groups", "checkpoint mask groups"),
        ("data_contract", "checkpoint data contract"),
    ):
        _require_equal(selected_state.get(key), latest_state.get(key), description)

    selected_epochs = selected["completed_epochs"]
    selected_history = selected_state.get("history")
    terminal_history = latest_state.get("history")
    if not isinstance(selected_history, list) or not isinstance(terminal_history, list):
        raise ValueError("selected and terminal checkpoints require training history")
    if selected_epochs > len(terminal_history):
        raise ValueError("selected checkpoint epoch is beyond the completed run")
    if selected_history != terminal_history[:selected_epochs]:
        raise ValueError("selected checkpoint history is not a prefix of the completed run")
    _validate_history_and_selection(selected_state, require_completion=False)

    same_checkpoint = initial_hashes[selected_path] == initial_hashes[latest_path]
    if selected_name == "latest.pt":
        if selected_path != latest_path or not same_checkpoint:
            raise ValueError("selected latest.pt does not match the completed-run checkpoint")
        selection_role = "latest"
    elif selected_name == "best_healthy.pt":
        selection_role = "best_healthy"
        _require_equal(
            selected_epochs,
            latest_state.get("best_healthy_epoch"),
            "terminal best healthy epoch",
        )
        _require_equal(
            selected_state.get("best_healthy_epoch"),
            selected_epochs,
            "selected best healthy epoch",
        )
        _require_equal(
            selected_state.get("best_healthy_val_loss"),
            latest_state.get("best_healthy_val_loss"),
            "best healthy validation loss",
        )
        final_val = selected_history[-1].get("val") if selected_history else None
        if (
            not isinstance(final_val, Mapping)
            or final_val.get("representations_healthy") is not True
        ):
            raise ValueError("best_healthy.pt was not selected from a healthy validation epoch")
    else:
        selection_role = "best_loss"
        _require_equal(
            selected_epochs, latest_state.get("best_epoch"), "terminal best loss epoch"
        )
        _require_equal(
            selected_state.get("best_epoch"), selected_epochs, "selected best loss epoch"
        )
        _require_equal(
            selected_state.get("best_val_loss"),
            latest_state.get("best_val_loss"),
            "best validation loss",
        )

    if selected_state.get("schema") == CHECKPOINT_SCHEMA:
        selected_fingerprint = selected_state["model_state_sha256"]
        commitment_key = {
            "best_loss": "best_loss_model_state_sha256",
            "best_healthy": "best_healthy_model_state_sha256",
            "latest": "model_state_sha256",
        }[selection_role]
        _require_equal(
            selected_fingerprint,
            latest_state.get(commitment_key),
            f"terminal {selection_role} model-state commitment",
        )

    if selection_role != "latest":
        selection_metric = str(
            latest_state["config"].get(
                "selection_metric", "subject_balanced_loss"
            )
        )
        final_val = selected_history[-1].get("val") if selected_history else None
        if not isinstance(final_val, Mapping) or selection_metric not in final_val:
            raise ValueError(
                f"selected checkpoint history lacks selection metric {selection_metric!r}"
            )
        expected_loss_key = (
            "best_healthy_val_loss" if selection_role == "best_healthy" else "best_val_loss"
        )
        _require_equal(
            final_val[selection_metric],
            latest_state.get(expected_loss_key),
            f"selected {selection_role} metric",
        )

    for path, digest in initial_hashes.items():
        require_unchanged_hash(path, digest, path.name)
    return {
        "selected_checkpoint": {**selected, "selection_role": selection_role},
        "completed_run_checkpoint": completed_run,
    }


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


def _open_directory_no_symlinks(directory: Path, *, create: bool) -> int:
    """Open an absolute directory through no-follow directory descriptors."""
    directory = Path(os.path.abspath(directory))
    if not directory.is_absolute():  # pragma: no cover - abspath guarantees this
        raise ValueError("atomic-write directory must be absolute")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(directory.anchor, flags)
    try:
        for part in directory.parts[1:]:
            try:
                next_descriptor = os.open(
                    part, flags | nofollow, dir_fd=descriptor
                )
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(part, mode=0o755, dir_fd=descriptor)
                except FileExistsError:
                    pass
                next_descriptor = os.open(
                    part, flags | nofollow, dir_fd=descriptor
                )
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _write_staged_text(parent_descriptor: int, temporary_name: str, text: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary_name, flags, 0o600, dir_fd=parent_descriptor)
    try:
        data = text.encode("utf-8")
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_texts_atomic(
    contents: Mapping[Path, str], *, validate_after=None, require_absent: bool = False
) -> None:
    """Publish same-directory text files with exception-safe pair rollback.

    The parent is held by a no-follow directory descriptor, so replacing an
    ancestor with a symlink after path validation cannot redirect publication.
    """
    if not contents:
        return
    normalized = {
        Path(os.path.abspath(Path(path))): text for path, text in contents.items()
    }
    parents = {path.parent for path in normalized}
    if len(parents) != 1:
        raise ValueError("atomic text publication requires one parent directory")
    if len({path.name for path in normalized}) != len(normalized):
        raise ValueError("atomic text publication requires distinct filenames")
    parent_descriptor = _open_directory_no_symlinks(parents.pop(), create=True)
    token = uuid.uuid4().hex
    staged = {
        path.name: f".{path.name}.{token}.tmp" for path in normalized
    }
    backups = {
        path.name: f".{path.name}.{token}.backup" for path in normalized
    }
    existed = {}
    backed_up = set()
    published = set()
    staged_inodes = {}
    try:
        for path, text in normalized.items():
            _write_staged_text(parent_descriptor, staged[path.name], text)
            staged_stat = os.stat(
                staged[path.name],
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            staged_inodes[path.name] = (staged_stat.st_dev, staged_stat.st_ino)
        try:
            if require_absent:
                for path in normalized:
                    try:
                        os.link(
                            staged[path.name],
                            path.name,
                            src_dir_fd=parent_descriptor,
                            dst_dir_fd=parent_descriptor,
                            follow_symlinks=False,
                        )
                    except FileExistsError as error:
                        raise FileExistsError(
                            f"atomic publication destination must be fresh: {path}"
                        ) from error
                    published.add(path.name)
            else:
                for path in normalized:
                    try:
                        os.stat(
                            path.name,
                            dir_fd=parent_descriptor,
                            follow_symlinks=False,
                        )
                        os.replace(
                            path.name,
                            backups[path.name],
                            src_dir_fd=parent_descriptor,
                            dst_dir_fd=parent_descriptor,
                        )
                        existed[path.name] = True
                        backed_up.add(path.name)
                    except FileNotFoundError:
                        existed[path.name] = False
                for path in normalized:
                    os.replace(
                        staged[path.name],
                        path.name,
                        src_dir_fd=parent_descriptor,
                        dst_dir_fd=parent_descriptor,
                    )
                    published.add(path.name)
            if validate_after is not None:
                validate_after()
        except BaseException:
            for path in reversed(tuple(normalized)):
                if path.name in published:
                    owns_destination = True
                    if require_absent:
                        try:
                            destination_stat = os.stat(
                                path.name,
                                dir_fd=parent_descriptor,
                                follow_symlinks=False,
                            )
                            owns_destination = (
                                destination_stat.st_dev,
                                destination_stat.st_ino,
                            ) == staged_inodes[path.name]
                        except FileNotFoundError:
                            owns_destination = False
                    if owns_destination:
                        try:
                            os.unlink(path.name, dir_fd=parent_descriptor)
                        except FileNotFoundError:
                            pass
                if path.name in backed_up:
                    os.replace(
                        backups[path.name],
                        path.name,
                        src_dir_fd=parent_descriptor,
                        dst_dir_fd=parent_descriptor,
                    )
            raise
        if not require_absent:
            for path in normalized:
                if existed[path.name]:
                    os.unlink(backups[path.name], dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
    finally:
        for temporary_name in (*staged.values(), *backups.values()):
            try:
                os.unlink(temporary_name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
        os.close(parent_descriptor)


def write_text_atomic(path: Path, text: str) -> None:
    write_texts_atomic({Path(path): text})


def build_baseline_report(repo_root, artifact_dir, report_path, protocol_path=PROTOCOL_PATH):
    repo_root = Path(repo_root).resolve()
    protocol_path = _resolved(Path(protocol_path), repo_root)
    protocol_hash = checkpoint_sha256(protocol_path)
    protocol = load_protocol(repo_root, protocol_path)
    require_unchanged_hash(protocol_path, protocol_hash, "Phase-0 protocol")
    read_only_evidence = validate_read_only_evidence(protocol, repo_root)
    manifest = validate_manifest(protocol, repo_root)
    baseline_dir = _resolved(Path(protocol["read_only_baseline_directory"]), repo_root)
    latest_path = _baseline_checkpoint_path(baseline_dir, "latest.pt")
    validate_completed_run(latest_path, protocol)
    artifact_dir = guard_research_path(artifact_dir, repo_root, write=True)
    candidates = {}
    for filename in protocol["candidate_checkpoints"]:
        checkpoint_path = _baseline_checkpoint_path(baseline_dir, filename)
        validate_checkpoint_from_completed_run(
            checkpoint_path, latest_path, protocol
        )
        label = Path(filename).stem
        candidate_dir = artifact_dir / label
        record = validate_candidate_artifacts(
            checkpoint_path,
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
    uv_lock_path = repo_root / "uv.lock"
    uv_lock_hash = checkpoint_sha256(uv_lock_path)
    code_hashes = {
        path: checkpoint_sha256(repo_root / path)
        for path in REPRODUCIBILITY_CODE_PATHS
    }
    payload = {
        "schema_version": 1,
        "status": "locked",
        "baseline_job_id": protocol["baseline_job_id"],
        "read_only_evidence": read_only_evidence,
        "retired_baseline_directories": protocol["retired_baseline_directories"],
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
            "protocol_path": _relative(protocol_path, repo_root),
            "protocol_sha256": protocol_hash,
            "protocol_payload": protocol,
            "uv_lock_sha256": uv_lock_hash,
            "code_sha256": code_hashes,
        },
    }
    if candidates[canonical_name]["checkpoint"]["identifier"] != protocol["canonical_checkpoint"]["identifier"]:
        raise ValueError("canonical checkpoint identifier drift")
    report_path = guard_research_path(report_path, repo_root, write=True)
    if report_path.suffix.casefold() != ".md":
        raise ValueError("baseline report path must end in .md")
    json_path = report_path.with_suffix(".json")

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
    markdown = "\n".join(lines)

    # Everything in the payload is a snapshot. Recheck every source both
    # immediately before and immediately after pair publication. The latter
    # runs while backups are still held, so drift rolls the pair back.
    def recheck_sources():
        require_unchanged_hash(protocol_path, protocol_hash, "Phase-0 protocol")
        require_unchanged_hash(uv_lock_path, uv_lock_hash, "uv.lock")
        require_unchanged_hash(
            repo_root / protocol["manifest"]["path"], manifest["sha256"], "manifest"
        )
        for path, digest in code_hashes.items():
            require_unchanged_hash(repo_root / path, digest, f"attested code {path}")
        verified_after = validate_read_only_evidence(protocol, repo_root)
        _require_equal(
            verified_after, read_only_evidence, "read-only evidence snapshot"
        )
        for filename, record in candidates.items():
            candidate_dir = artifact_dir / Path(filename).stem
            require_unchanged_hash(
                candidate_dir / "features.npz",
                record["features"]["sha256"],
                f"{filename} feature table",
            )
            require_unchanged_hash(
                candidate_dir / "features.npz.metadata.json",
                record["features"]["metadata_sha256"],
                f"{filename} feature metadata",
            )
            require_unchanged_hash(
                candidate_dir / "probes" / "probe_metrics.json",
                record["probes"]["json_sha256"],
                f"{filename} probe report",
            )

    recheck_sources()
    write_texts_atomic(
        {
            json_path: json.dumps(payload, indent=2, sort_keys=True) + "\n",
            report_path: markdown,
        },
        validate_after=recheck_sources,
    )
    return {"markdown": report_path, "json": json_path, "payload": payload}


__all__ = [
    "PROTOCOL_PATH",
    "READ_ONLY_BASELINE_PATH",
    "REPRODUCIBILITY_CODE_PATHS",
    "REQUIRED_READ_ONLY_EVIDENCE_PATHS",
    "RETAINED_JOB_NOTEBOOK_PATH",
    "RETIRED_BASELINE_PATH",
    "build_baseline_report",
    "checkpoint_record",
    "guard_research_path",
    "load_protocol",
    "prepare_empty_directory",
    "portable_path",
    "require_unchanged_hash",
    "validate_candidate_artifacts",
    "validate_completed_run",
    "validate_checkpoint_from_completed_run",
    "validate_manifest",
    "validate_read_only_evidence",
    "write_text_atomic",
    "write_texts_atomic",
]
