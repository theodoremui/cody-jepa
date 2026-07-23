#!/usr/bin/env python3
"""Run or submit the reproducible train-to-report single-stream pipeline."""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import uuid

from cody_jepa.phase0 import (
    build_baseline_report,
    checkpoint_record,
    guard_research_path,
    load_protocol,
    prepare_empty_directory,
    validate_completed_run,
    validate_manifest,
    write_text_atomic,
)
from cody_jepa.probes import (
    FEATURE_FORMULA,
    FEATURE_SOURCE,
    checkpoint_sha256,
    evaluate_all_probes,
    read_feature_table,
    validate_feature_metadata,
)


def _run(command, *, cwd, env=None):
    print("+", " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _relative_or_absolute(path, repo_root):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _baseline_destinations(args, repo_root):
    """Return explicit destinations or a unique ignored regeneration directory."""
    if (args.artifact_dir is None) != (args.report is None):
        raise ValueError("baseline --artifact-dir and --report must be provided together")
    if args.artifact_dir is not None:
        return Path(args.artifact_dir), Path(args.report)
    regeneration = (
        repo_root
        / "outputs"
        / "phase0"
        / "regenerations"
        / uuid.uuid4().hex
    )
    return regeneration / "artifacts", regeneration / "phase0-baseline.md"


def _evaluate_checkpoint(
    *,
    checkpoint,
    artifact_dir,
    repo_root,
    device,
    batch_size,
    num_workers,
    windows_per_sequence,
    seed,
    max_iter,
    identity_validation_fraction,
    retrieval_enrollment_sequences,
):
    checkpoint = guard_research_path(checkpoint, repo_root, write=False)
    artifact_dir = prepare_empty_directory(artifact_dir, repo_root)
    features = artifact_dir / "features.npz"
    probes = artifact_dir / "probes"
    _run(
        [
            sys.executable,
            str(repo_root / "scripts" / "export_single_stream_features.py"),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(features),
            "--repo-root",
            str(repo_root),
            "--device",
            device,
            "--batch-size",
            str(batch_size),
            "--num-workers",
            str(num_workers),
            "--windows-per-sequence",
            str(windows_per_sequence),
            "--image-verify-mode",
            "none",
        ],
        cwd=repo_root,
    )
    _run(
        [
            sys.executable,
            str(repo_root / "scripts" / "eval_probes.py"),
            "--features",
            str(features),
            "--output-dir",
            str(probes),
            "--seed",
            str(seed),
            "--max-iter",
            str(max_iter),
            "--identity-validation-fraction",
            str(identity_validation_fraction),
            "--retrieval-enrollment-sequences",
            str(retrieval_enrollment_sequences),
        ],
        cwd=repo_root,
    )
    return {"features": features, "probe_json": probes / "probe_metrics.json"}


def _write_generic_report(
    checkpoint, artifacts, report_path, repo_root, success_criterion=None
):
    report_path = guard_research_path(report_path, repo_root, write=True)
    if report_path.suffix.casefold() != ".md":
        raise ValueError("report path must end in .md")
    if report_path.exists() or report_path.with_suffix(".json").exists():
        raise FileExistsError(f"report destination must be fresh: {report_path}")
    checkpoint = checkpoint_record(checkpoint)
    table, feature_metadata = read_feature_table(artifacts["features"])
    validate_feature_metadata(table, artifacts["features"], feature_metadata)
    if feature_metadata["checkpoint_sha256"] != checkpoint["sha256"]:
        raise ValueError("feature table was not exported from the declared checkpoint")
    probes = json.loads(artifacts["probe_json"].read_text())
    sidecar = artifacts["features"].with_suffix(".npz.metadata.json")
    expected_probe_metadata = {
        "feature_table_sha256": checkpoint_sha256(artifacts["features"]),
        "feature_metadata_sha256": checkpoint_sha256(sidecar),
        "checkpoint_sha256": checkpoint["sha256"],
        "feature_source": FEATURE_SOURCE,
        "feature_formula": FEATURE_FORMULA,
    }
    for key, expected in expected_probe_metadata.items():
        if probes.get(key) != expected:
            raise ValueError(
                f"probe provenance {key!r} mismatch: "
                f"expected={expected!r}, actual={probes.get(key)!r}"
            )
    reproduced = evaluate_all_probes(
        table,
        feature_source=FEATURE_SOURCE,
        validation_fraction=float(probes["identity_validation_fraction"]),
        enrollment_sequences=int(probes["retrieval_enrollment_sequences"]),
        max_iter=int(probes["max_iter"]),
        seed=int(probes["seed"]),
    )
    if probes.get("results") != reproduced:
        raise ValueError("probe metrics do not reproduce from the feature table")
    payload = {
        "schema_version": 1,
        "success_criterion": success_criterion,
        "checkpoint": checkpoint,
        "feature_table": {
            "path": _relative_or_absolute(artifacts["features"], repo_root),
            "sha256": checkpoint_sha256(artifacts["features"]),
            "metadata_sha256": checkpoint_sha256(
                artifacts["features"].with_suffix(".npz.metadata.json")
            ),
        },
        "probe_results": probes,
    }
    write_text_atomic(report_path.with_suffix(".json"), json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Single-stream checkpoint report",
        "",
        f"Checkpoint: `{checkpoint['identifier']}` (epoch {checkpoint['completed_epochs']}).",
        f"Predeclared success criterion: {success_criterion or 'evaluation-only; no training claim'}.",
        "",
        "| Probe | Accuracy | Balanced accuracy | Macro F1 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for result in probes["results"]:
        lines.append(
            f"| {result['task']} | {result['accuracy']:.4f} | "
            f"{result['balanced_accuracy']:.4f} | {result['macro_f1']:.4f} |"
        )
    lines.append("")
    write_text_atomic(report_path, "\n".join(lines))
    return {"markdown": str(report_path), "json": str(report_path.with_suffix('.json'))}


def command_baseline(args):
    repo_root = args.repo_root.resolve()
    artifact_dir, report_path = _baseline_destinations(args, repo_root)
    protocol = load_protocol(repo_root, args.protocol)
    validate_manifest(protocol, repo_root)
    frozen = protocol["probes"]
    export = protocol["feature_export"]
    if args.device != export["reference_device"]:
        raise ValueError(
            f"baseline reference device is frozen to {export['reference_device']!r}; "
            f"got {args.device!r}"
        )
    if args.batch_size != export["reference_batch_size"]:
        raise ValueError("baseline batch size differs from the frozen reference")
    if args.num_workers != export["reference_num_workers"]:
        raise ValueError("baseline worker count differs from the frozen reference")
    artifact_root = guard_research_path(artifact_dir, repo_root, write=True)
    if artifact_root.exists() and any(artifact_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite baseline artifacts: {artifact_root}")
    baseline_dir = guard_research_path(
        protocol["read_only_baseline_directory"], repo_root, write=False
    )
    validate_completed_run(baseline_dir / "latest.pt")
    for filename in protocol["candidate_checkpoints"]:
        checkpoint_record(baseline_dir / filename, protocol, filename)
        _evaluate_checkpoint(
            checkpoint=baseline_dir / filename,
            artifact_dir=artifact_root / Path(filename).stem,
            repo_root=repo_root,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            windows_per_sequence=export["windows_per_sequence"],
            seed=frozen["seed"],
            max_iter=frozen["max_iter"],
            identity_validation_fraction=frozen["identity_validation_fraction"],
            retrieval_enrollment_sequences=frozen["retrieval_enrollment_sequences"],
        )
    result = build_baseline_report(
        repo_root, artifact_root, report_path, args.protocol
    )
    print(json.dumps({key: str(value) for key, value in result.items() if key != "payload"}, indent=2))


def command_baseline_report(args):
    result = build_baseline_report(
        args.repo_root.resolve(), args.artifact_dir, args.report, args.protocol
    )
    print(json.dumps({key: str(value) for key, value in result.items() if key != "payload"}, indent=2))


def command_evaluate(args):
    repo_root = args.repo_root.resolve()
    artifacts = _evaluate_checkpoint(
        checkpoint=args.checkpoint,
        artifact_dir=args.artifact_dir,
        repo_root=repo_root,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        windows_per_sequence=args.windows_per_sequence,
        seed=args.seed,
        max_iter=args.max_iter,
        identity_validation_fraction=args.identity_validation_fraction,
        retrieval_enrollment_sequences=args.retrieval_enrollment_sequences,
    )
    result = _write_generic_report(args.checkpoint, artifacts, args.report, repo_root)
    print(json.dumps(result, indent=2))


def command_run(args):
    repo_root = args.repo_root.resolve()
    if not os.environ.get("SLURM_JOB_ID") and not args.allow_local_run:
        raise RuntimeError(
            "run requires a Slurm allocation; pass --allow-local-run only on an intentional local GPU worker"
        )
    run_dir = guard_research_path(args.run_dir, repo_root, write=True)
    artifact_dir = guard_research_path(args.artifact_dir, repo_root, write=True)
    report_path = guard_research_path(args.report, repo_root, write=True)
    resolved_destinations = {run_dir, artifact_dir, report_path}
    if len(resolved_destinations) != 3:
        raise ValueError("run, artifact, and report destinations must be distinct")
    if report_path.exists() or report_path.with_suffix(".json").exists():
        raise FileExistsError(f"report destination must be fresh: {report_path}")
    if run_dir.exists():
        raise FileExistsError(f"training run directory must be fresh: {run_dir}")
    run_dir.mkdir(parents=True)
    (run_dir / ".pipeline-claim").write_text(f"pid={os.getpid()}\n")
    notebook_dir = guard_research_path(args.notebook_dir, repo_root, write=True)
    notebook_dir.mkdir(parents=True, exist_ok=True)
    run_label = os.environ.get("SLURM_JOB_ID", "local")
    env = dict(os.environ)
    env.update(
        {
            "CODY_JEPA_RUN_FULL_TRAINING": "1",
            "CODY_JEPA_RUN_DATA_AUDIT": "0",
            "CODY_JEPA_RUN_EXHAUSTIVE_DATA_AUDIT": "0",
            "CODY_JEPA_OUTPUT_DIR": _relative_or_absolute(run_dir, repo_root),
            "MPLCONFIGDIR": env.get("MPLCONFIGDIR", "/tmp/mpl"),
        }
    )
    env.pop("CODY_JEPA_RESUME_CHECKPOINT", None)
    _run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            str(repo_root / "notebooks" / "single-stream-jepa.ipynb"),
            "--output-dir",
            str(notebook_dir),
            "--output",
            f"single-stream-jepa-{run_label}.executed.ipynb",
            "--ExecutePreprocessor.timeout=-1",
        ],
        cwd=repo_root,
        env=env,
    )
    validate_completed_run(run_dir / "latest.pt")
    selected = run_dir / args.checkpoint_name
    if not selected.is_file():
        raise FileNotFoundError(
            f"declared checkpoint was not produced: {selected}; no fallback is permitted"
        )
    artifacts = _evaluate_checkpoint(
        checkpoint=selected,
        artifact_dir=artifact_dir,
        repo_root=repo_root,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        windows_per_sequence=args.windows_per_sequence,
        seed=args.seed,
        max_iter=args.max_iter,
        identity_validation_fraction=args.identity_validation_fraction,
        retrieval_enrollment_sequences=args.retrieval_enrollment_sequences,
    )
    result = _write_generic_report(
        selected, artifacts, report_path, repo_root, args.success_criterion
    )
    print(json.dumps(result, indent=2))


def command_submit(args):
    if os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("submit must run outside an existing Slurm allocation")
    repo_root = args.repo_root.resolve()
    run_dir = guard_research_path(args.run_dir, repo_root, write=True)
    artifact_dir = guard_research_path(args.artifact_dir, repo_root, write=True)
    report = guard_research_path(args.report, repo_root, write=True)
    if len({run_dir, artifact_dir, report}) != 3:
        raise ValueError("run, artifact, and report destinations must be distinct")
    if run_dir.exists() or artifact_dir.exists() or report.exists() or report.with_suffix(".json").exists():
        raise FileExistsError("run, artifact, and report destinations must be fresh")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    claim = run_dir.parent / f".{run_dir.name}.pipeline-claim"
    descriptor = os.open(claim, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(f"pid={os.getpid()}\n")
    (repo_root / "logs").mkdir(exist_ok=True)
    (repo_root / "notebook-runs").mkdir(exist_ok=True)
    exported = {
        "CODY_JEPA_OUTPUT_DIR": _relative_or_absolute(run_dir, repo_root),
        "CODY_JEPA_ARTIFACT_DIR": _relative_or_absolute(artifact_dir, repo_root),
        "CODY_JEPA_REPORT_PATH": _relative_or_absolute(report, repo_root),
        "CODY_JEPA_CHECKPOINT_NAME": args.checkpoint_name,
        "CODY_JEPA_SUCCESS_CRITERION": args.success_criterion,
    }
    for value in exported.values():
        if "," in value or "\n" in value:
            raise ValueError("Slurm-exported values must not contain commas or newlines")
    export_arg = "ALL," + ",".join(exported)
    command = [
        "sbatch",
        "--parsable",
        f"--export={export_arg}",
        str(repo_root / "slurm" / "train-single-stream-jepa.sbatch"),
    ]
    print("+", " ".join(command), flush=True)
    submit_env = dict(os.environ)
    submit_env.update(exported)
    try:
        result = subprocess.run(
            command,
            cwd=repo_root,
            env=submit_env,
            check=True,
            text=True,
            capture_output=True,
        )
    except Exception:
        claim.unlink(missing_ok=True)
        raise
    print(json.dumps({"submitted_job_id": result.stdout.strip(), **exported}, indent=2))


def _add_shared_evaluation(parser):
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--windows-per-sequence", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--identity-validation-fraction", type=float, default=0.25)
    parser.add_argument("--retrieval-enrollment-sequences", type=int, default=1)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline = subparsers.add_parser("baseline", help="cleanly rerun both job-91108 candidates")
    baseline.add_argument("--protocol", type=Path, default=Path("protocols/phase0-baseline.json"))
    baseline.add_argument(
        "--artifact-dir",
        type=Path,
        help="fresh artifact directory; requires --report (default: unique ignored regeneration)",
    )
    baseline.add_argument(
        "--report",
        type=Path,
        help="fresh report path; requires --artifact-dir (default: unique ignored regeneration)",
    )
    baseline.add_argument("--device", default="mps")
    baseline.add_argument("--batch-size", type=int, default=4)
    baseline.add_argument("--num-workers", type=int, default=4)
    baseline.set_defaults(func=command_baseline)

    report = subparsers.add_parser("baseline-report", help="verify artifacts and regenerate report")
    report.add_argument("--protocol", type=Path, default=Path("protocols/phase0-baseline.json"))
    report.add_argument("--artifact-dir", type=Path, default=Path("outputs/phase0/job-91108"))
    report.add_argument("--report", type=Path, default=Path("reports/phase0-baseline.md"))
    report.set_defaults(func=command_baseline_report)

    evaluate = subparsers.add_parser("evaluate", help="evaluate one completed checkpoint")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--artifact-dir", type=Path, required=True)
    evaluate.add_argument("--report", type=Path, required=True)
    _add_shared_evaluation(evaluate)
    evaluate.set_defaults(func=command_evaluate)

    run = subparsers.add_parser("run", help="train in this allocation, then evaluate and report")
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--artifact-dir", type=Path, required=True)
    run.add_argument("--report", type=Path, required=True)
    run.add_argument("--checkpoint-name", choices=("best_loss.pt", "best_healthy.pt", "latest.pt"), required=True)
    run.add_argument("--notebook-dir", type=Path, default=Path("notebook-runs"))
    run.add_argument("--success-criterion", required=True)
    run.add_argument("--allow-local-run", action="store_true")
    _add_shared_evaluation(run)
    run.set_defaults(func=command_run)

    submit = subparsers.add_parser("submit", help="submit the full pipeline to HAIC Slurm")
    submit.add_argument("--run-dir", type=Path, required=True)
    submit.add_argument("--artifact-dir", type=Path, required=True)
    submit.add_argument("--report", type=Path, required=True)
    submit.add_argument("--checkpoint-name", choices=("best_loss.pt", "best_healthy.pt", "latest.pt"), required=True)
    submit.add_argument("--success-criterion", required=True)
    submit.set_defaults(func=command_submit)
    return parser.parse_args()


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
