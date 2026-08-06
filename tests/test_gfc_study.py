import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from cody_jepa.evaluation.gfc.roles import ROLE_MAP_VERSION
from cody_jepa.evaluation.gfc.study import (
    ANALYSIS_FREEZE_TAG,
    ANALYSIS_IDS,
    CHECKPOINT_METADATA_VERSION,
    GALLERY,
    LADDER_CONTRAST_COLUMNS,
    PRIMARY_ANALYSIS_ID,
    PROTOCOL,
    REGISTRY_COLUMNS,
    RUN_TABLE_COLUMNS,
    PrimaryRun,
    infer_five_ladder_study,
    preflight_study,
    run_study,
    summarize_study,
    validate_study_registry,
)


LADDERS = tuple(f"ladder-{index}" for index in range(5))
RUNGS = ("small", "medium", "large", "full")
SIZES = (2_500, 25_000, 250_000, 900_000)


def registry_table() -> pd.DataFrame:
    rows = []
    for ladder_index, ladder in enumerate(LADDERS):
        for rung, size in zip(RUNGS, SIZES):
            label = f"{ladder}-{rung}"
            rows.append(
                {
                    "model_label": label,
                    "ladder": ladder,
                    "rung": rung,
                    "checkpoint_id": f"{label}-final",
                    "checkpoint_path": f"/private/checkpoints/{label}.pt",
                    "feature_path": f"/private/features/{label}.npz",
                    "pool_seed": ladder_index,
                    "optimization_seed": ladder_index,
                    "unique_sequences": size,
                    "training_exposure": 8_192_000,
                }
            )
    return pd.DataFrame(rows, columns=REGISTRY_COLUMNS)


def cohort_roles() -> dict[str, object]:
    return {
        "version": ROLE_MAP_VERSION,
        "fit_role": "development",
        "evaluation_role": "locked_outcome",
        "assigned_counts": {"development": 80, "locked_outcome": 318},
        "complete_counts": {"development": 76, "locked_outcome": 308},
        "excluded_counts": {"development": 4, "locked_outcome": 10},
    }


def primary_summary(label: str, ladder: str, rung: str) -> dict[str, object]:
    ladder_index = LADDERS.index(ladder)
    rung_index = RUNGS.index(rung)
    return {
        "protocol": PROTOCOL,
        "gallery": GALLERY,
        "queries_per_participant": 16,
        "analysis_id": PRIMARY_ANALYSIS_ID,
        "ridge_alpha": 1.0,
        "normalization": "raw_retain_all",
        "method_settings": {"frozen": "settings-v1"},
        "model": {
            "label": label,
            "ladder": ladder,
            "rung": rung,
            "checkpoint_id": f"{label}-final",
            "pool_seed": ladder_index,
            "optimization_seed": ladder_index,
            "unique_sequences": SIZES[rung_index],
            "training_exposure": 8_192_000,
        },
        "revision": {
            "code_commit": "abc123",
            "analysis_freeze_tag": ANALYSIS_FREEZE_TAG,
        },
        "cohort_roles": cohort_roles(),
        "learned": {"top1": 0.5, "mrr": 0.65},
        "shortcut": {"top1": 0.25, "mrr": 0.4},
        "independent_factor_controls": {
            "hard": {"top1": 0.4, "mrr": 0.55},
            "soft": {
                "top1": 0.4,
                "mrr": 0.57,
                "target_probability": 0.3,
                "target_nll": 1.3,
            },
            "temperature": {"fitted_temperature": 1.2},
        },
    }


def inference_runs(endpoint_contrasts=None, participant_offsets=None):
    endpoint_contrasts = endpoint_contrasts or [0.10] * 5
    if participant_offsets is None:
        participant_offsets = np.linspace(-0.02, 0.03, 308)
    runs = []
    for ladder, contrast in zip(LADDERS, endpoint_contrasts):
        for rung_index, rung in enumerate(RUNGS):
            fraction = rung_index / 3.0
            values = {
                f"private-{index}": 0.4 + offset + contrast * fraction
                for index, offset in enumerate(participant_offsets)
            }
            label = f"{ladder}-{rung}"
            runs.append(
                PrimaryRun(
                    model_label=label,
                    ladder=ladder,
                    rung=rung,
                    participant_top1=values,
                    summary=primary_summary(label, ladder, rung),
                )
            )
    return runs


class RegistryTest(unittest.TestCase):
    def test_exact_five_by_four_registry_is_canonicalized(self):
        shuffled = registry_table().sample(frac=1.0, random_state=4)
        result = validate_study_registry(shuffled)
        self.assertEqual(len(result), 20)
        self.assertEqual(result.groupby("ladder")["rung"].apply(tuple).iloc[0], RUNGS)

    def test_incomplete_duplicate_and_exposure_mismatch_fail(self):
        with self.assertRaisesRegex(ValueError, "exactly 20"):
            validate_study_registry(registry_table().iloc[:-1])
        duplicate = registry_table()
        duplicate.loc[1, "rung"] = "small"
        with self.assertRaisesRegex(ValueError, "one of each rung"):
            validate_study_registry(duplicate)
        exposure = registry_table()
        exposure.loc[0, "training_exposure"] = 1
        with self.assertRaisesRegex(ValueError, "equal training exposure"):
            validate_study_registry(exposure)

    def test_registry_rejects_unsafe_labels_and_mismatched_full_counts(self):
        unsafe = registry_table()
        unsafe.loc[0, "model_label"] = "../outside"
        with self.assertRaisesRegex(ValueError, "safe single path components"):
            validate_study_registry(unsafe)
        inconsistent_full = registry_table()
        full_index = inconsistent_full.index[inconsistent_full["rung"] == "full"][0]
        inconsistent_full.loc[full_index, "unique_sequences"] += 1
        with self.assertRaisesRegex(ValueError, "same full-data sequence count"):
            validate_study_registry(inconsistent_full)

    def test_registry_rejects_canonical_path_aliases(self):
        aliased_features = registry_table()
        aliased_features.loc[0, "feature_path"] = "/private/features/shared.npz"
        aliased_features.loc[1, "feature_path"] = (
            "/private/features/subdirectory/../shared.npz"
        )
        with self.assertRaisesRegex(ValueError, "feature_path values must resolve"):
            validate_study_registry(aliased_features)

        aliased_checkpoints = registry_table()
        aliased_checkpoints.loc[0, "checkpoint_path"] = "/private/checkpoints/shared.pt"
        aliased_checkpoints.loc[1, "checkpoint_path"] = (
            "/private/checkpoints/subdirectory/../shared.pt"
        )
        with self.assertRaisesRegex(ValueError, "checkpoint_path values must resolve"):
            validate_study_registry(aliased_checkpoints)

    def test_run_orchestration_loads_each_archive_once_and_stops_on_failure(self):
        loaded = []
        evaluated = []

        def load(path):
            loaded.append(path)
            return path

        def evaluate(archive, row, analysis_id):
            evaluated.append((archive, analysis_id))
            return row["model_label"], analysis_id

        outputs = run_study(registry_table(), load_archive=load, evaluate_analysis=evaluate)
        self.assertEqual(len(loaded), 20)
        self.assertEqual(len(outputs), 100)
        self.assertEqual([item[1] for item in evaluated[:5]], list(ANALYSIS_IDS))


class PreflightTest(unittest.TestCase):
    def _private_inputs(self, root: Path):
        registry = registry_table().copy()
        checkpoint_states = {}
        for row in registry.itertuples(index=False):
            checkpoint = root / "checkpoints" / f"{row.model_label}.pt"
            feature = root / "features" / f"{row.model_label}.npz"
            checkpoint.parent.mkdir(exist_ok=True)
            feature.parent.mkdir(exist_ok=True)
            checkpoint.touch()
            feature.touch()
            registry.loc[registry["model_label"] == row.model_label, "checkpoint_path"] = str(
                checkpoint
            )
            registry.loc[registry["model_label"] == row.model_label, "feature_path"] = str(
                feature
            )
            checkpoint_states[checkpoint.resolve()] = {
                "config": {"steps": 100},
                "global_step": 100,
                "study_metadata": {
                    "version": CHECKPOINT_METADATA_VERSION,
                    "training_dataset": "GaitLU-1M",
                    "checkpoint_kind": "final_step",
                    "model_label": row.model_label,
                    "checkpoint_id": row.checkpoint_id,
                    "pool_seed": row.pool_seed,
                    "optimization_seed": row.optimization_seed,
                    "unique_sequences": row.unique_sequences,
                    "training_exposure": row.training_exposure,
                },
            }
        role_rows = [
            *(
                {"subject_id": f"D{index:03d}", "role": "development"}
                for index in range(80)
            ),
            *(
                {"subject_id": f"O{index:03d}", "role": "locked_outcome"}
                for index in range(318)
            ),
        ]
        role_map = root / "roles.csv"
        pd.DataFrame(role_rows).to_csv(role_map, index=False)
        subjects = {row["subject_id"] for row in role_rows}
        complete = {
            *(f"D{index:03d}" for index in range(76)),
            *(f"O{index:03d}" for index in range(308)),
        }
        return registry, role_map, subjects, complete, checkpoint_states

    def test_preflight_checks_all_models_metadata_and_frozen_cohort_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry, role_map, subjects, complete, states = self._private_inputs(root)
            with mock.patch(
                "cody_jepa.evaluation.gfc.study.load_checkpoint",
                side_effect=lambda path: states[Path(path).resolve()],
            ), mock.patch(
                "cody_jepa.evaluation.gfc.study._feature_participants",
                return_value=(subjects, complete),
            ):
                result = preflight_study(
                    registry,
                    role_map=role_map,
                    output_root=root / "private-output",
                    aggregate_output=root / "aggregate-output",
                    require_frozen_revision=False,
                )
            self.assertEqual(result["eligible_models"], 20)
            self.assertEqual(
                result["complete_counts"],
                {"development": 76, "locked_outcome": 308},
            )

    def test_preflight_requires_checkpoint_provenance_in_existing_sidecar(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry, role_map, subjects, complete, states = self._private_inputs(root)
            feature = Path(registry.iloc[0]["feature_path"])
            feature.with_suffix(feature.suffix + ".metadata.json").write_text(
                json.dumps({"schema": "feature-sidecar-v1"}), encoding="utf-8"
            )
            with mock.patch(
                "cody_jepa.evaluation.gfc.study.load_checkpoint",
                side_effect=lambda path: states[Path(path).resolve()],
            ), mock.patch(
                "cody_jepa.evaluation.gfc.study._feature_participants",
                return_value=(subjects, complete),
            ), self.assertRaisesRegex(ValueError, "lacks checkpoint provenance"):
                preflight_study(
                    registry,
                    role_map=role_map,
                    output_root=root / "private-output",
                    aggregate_output=root / "aggregate-output",
                    require_frozen_revision=False,
                )

    def test_preflight_rejects_non_gaitlu_metadata_and_wrong_complete_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry, role_map, subjects, complete, states = self._private_inputs(root)
            first = next(iter(states.values()))
            first["study_metadata"]["training_dataset"] = "Health&Gait"
            with mock.patch(
                "cody_jepa.evaluation.gfc.study.load_checkpoint",
                side_effect=lambda path: states[Path(path).resolve()],
            ), mock.patch(
                "cody_jepa.evaluation.gfc.study._feature_participants",
                return_value=(subjects, complete),
            ), self.assertRaisesRegex(ValueError, "study metadata disagrees"):
                preflight_study(
                    registry,
                    role_map=role_map,
                    output_root=root / "private-output",
                    aggregate_output=root / "aggregate-output",
                    require_frozen_revision=False,
                )

            first["study_metadata"]["training_dataset"] = "GaitLU-1M"
            incomplete = set(complete)
            incomplete.remove("O000")
            with mock.patch(
                "cody_jepa.evaluation.gfc.study.load_checkpoint",
                side_effect=lambda path: states[Path(path).resolve()],
            ), mock.patch(
                "cody_jepa.evaluation.gfc.study._feature_participants",
                return_value=(subjects, incomplete),
            ), self.assertRaisesRegex(ValueError, "complete role counts"):
                preflight_study(
                    registry,
                    role_map=role_map,
                    output_root=root / "private-output",
                    aggregate_output=root / "aggregate-output",
                    require_frozen_revision=False,
                )


class FiveLadderInferenceTest(unittest.TestCase):
    def test_known_contrasts_t_df_pairing_and_determinism(self):
        contrasts = [0.02, 0.04, 0.06, 0.08, 0.10]
        first = infer_five_ladder_study(
            inference_runs(contrasts), participant_resamples=300, crossed_resamples=500, seed=9
        )
        second = infer_five_ladder_study(
            inference_runs(contrasts), participant_resamples=300, crossed_resamples=500, seed=9
        )
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["mean_full_minus_small"], 0.06)
        self.assertEqual(first["t_interval_95"]["degrees_of_freedom"], 4)
        self.assertEqual(
            [round(item["full_minus_small"], 8) for item in first["ladder_contrasts"]],
            contrasts,
        )
        # A shared participant draw preserves this synthetic constant endpoint delta.
        for item in first["participant_bootstraps"]:
            interval = item["participant_bootstrap"]["endpoint_interval_95"]
            self.assertAlmostEqual(interval["lower"], item["full_minus_small"])
            self.assertAlmostEqual(interval["upper"], item["full_minus_small"])

    def test_decision_categories_and_overlap_flags(self):
        meaningful = infer_five_ladder_study(
            inference_runs([0.08] * 5), participant_resamples=20, crossed_resamples=20
        )
        self.assertEqual(meaningful["decision"], "Meaningful positive")
        self.assertTrue(meaningful["superiority"])

        small = infer_five_ladder_study(
            inference_runs([0.03] * 5), participant_resamples=20, crossed_resamples=20
        )
        self.assertEqual(small["decision"], "Positive but small")
        self.assertTrue(small["equivalence"])

        equivalent = infer_five_ladder_study(
            inference_runs([0.0] * 5), participant_resamples=20, crossed_resamples=20
        )
        self.assertEqual(equivalent["decision"], "Equivalent at the 6.25-point resolution")

        inconclusive = infer_five_ladder_study(
            inference_runs([-0.2, -0.1, 0.0, 0.1, 0.2]),
            participant_resamples=20,
            crossed_resamples=20,
        )
        self.assertEqual(inconclusive["decision"], "Inconclusive")
        self.assertFalse(inconclusive["superiority"])
        self.assertFalse(inconclusive["equivalence"])

    def test_missing_cells_participants_and_mixed_revision_fail(self):
        runs = inference_runs()
        with self.assertRaisesRegex(ValueError, "exactly 20"):
            infer_five_ladder_study(runs[:-1], participant_resamples=2, crossed_resamples=2)
        changed = list(runs)
        run = changed[0]
        changed[0] = PrimaryRun(
            run.model_label,
            run.ladder,
            run.rung,
            {"different": 0.5},
            run.summary,
        )
        with self.assertRaisesRegex(ValueError, "same complete outcome"):
            infer_five_ladder_study(changed, participant_resamples=2, crossed_resamples=2)
        changed = list(runs)
        summary = dict(changed[0].summary)
        summary["revision"] = {**summary["revision"], "code_commit": "different"}
        changed[0] = PrimaryRun(
            changed[0].model_label,
            changed[0].ladder,
            changed[0].rung,
            changed[0].participant_top1,
            summary,
        )
        with self.assertRaisesRegex(ValueError, "mixed code commit"):
            infer_five_ladder_study(changed, participant_resamples=2, crossed_resamples=2)

        changed = list(runs)
        summary = dict(changed[0].summary)
        summary["method_settings"] = {"frozen": "different"}
        changed[0] = PrimaryRun(
            changed[0].model_label,
            changed[0].ladder,
            changed[0].rung,
            changed[0].participant_top1,
            summary,
        )
        with self.assertRaisesRegex(ValueError, "mixed method_settings"):
            infer_five_ladder_study(changed, participant_resamples=2, crossed_resamples=2)


class AggregateSummarizerTest(unittest.TestCase):
    def _write_outputs(self, root: Path) -> Path:
        registry = registry_table()
        registry_path = root / "registry.csv"
        registry.to_csv(registry_path, index=False)
        outputs = root / "private-results"
        for row in registry.itertuples(index=False):
            for analysis_id in ANALYSIS_IDS:
                directory = outputs / row.model_label / analysis_id
                directory.mkdir(parents=True)
                summary = primary_summary(row.model_label, row.ladder, row.rung)
                normalization, alpha = {
                    "raw_retain_all-alpha-1": ("raw_retain_all", 1.0),
                    "raw_retain_all-alpha-0.1": ("raw_retain_all", 0.1),
                    "raw_retain_all-alpha-10": ("raw_retain_all", 10.0),
                    "raw_effective_rank-alpha-1": ("raw_effective_rank", 1.0),
                    "pca_effective_rank-alpha-1": ("pca_effective_rank", 1.0),
                }[analysis_id]
                summary.update(
                    analysis_id=analysis_id,
                    normalization=normalization,
                    ridge_alpha=alpha,
                )
                rung_index = RUNGS.index(row.rung)
                summary["learned"]["top1"] = 0.4 + rung_index * 0.02
                (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
                if analysis_id == PRIMARY_ANALYSIS_ID:
                    pd.DataFrame(
                        {
                            "participant": [
                                f"private-{index:03d}" for index in range(308)
                            ],
                            "learned_top1": np.linspace(0.35, 0.45, 308)
                            + rung_index * 0.02,
                        }
                    ).to_csv(directory / "participants.csv", index=False)
        return registry_path

    def test_writes_only_aggregate_stable_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry = self._write_outputs(root)
            aggregate = root / "aggregate"
            result = summarize_study(
                registry,
                root / "private-results",
                aggregate,
                participant_resamples=30,
                crossed_resamples=40,
                seed=3,
            )
            self.assertEqual(
                sorted(path.name for path in aggregate.iterdir()),
                ["ladder_contrasts.csv", "outcome_summary.json", "run_table.csv"],
            )
            runs = pd.read_csv(aggregate / "run_table.csv")
            ladders = pd.read_csv(aggregate / "ladder_contrasts.csv")
            self.assertEqual(runs.columns.tolist(), list(RUN_TABLE_COLUMNS))
            self.assertEqual(ladders.columns.tolist(), list(LADDER_CONTRAST_COLUMNS))
            self.assertEqual((len(runs), len(ladders)), (20, 5))
            serialized = json.dumps(result) + runs.to_csv(index=False) + ladders.to_csv(index=False)
            for private in (
                "private-000",
                "/private/features",
                "feature_path",
                "checkpoint_path",
            ):
                self.assertNotIn(private, serialized)

    def test_missing_outcome_participant_cannot_be_summarized(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry = self._write_outputs(root)
            for participant_file in (root / "private-results").glob(
                f"*/{PRIMARY_ANALYSIS_ID}/participants.csv"
            ):
                rows = pd.read_csv(participant_file).iloc[:-1]
                rows.to_csv(participant_file, index=False)
            with self.assertRaisesRegex(ValueError, "all 308 complete outcome"):
                summarize_study(
                    registry,
                    root / "private-results",
                    root / "aggregate",
                    participant_resamples=2,
                    crossed_resamples=2,
                )

    def test_registry_metadata_change_cannot_relabel_completed_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry = self._write_outputs(root)
            rows = pd.read_csv(registry)
            rows.loc[0, "unique_sequences"] += 1
            rows.to_csv(registry, index=False)
            with self.assertRaisesRegex(ValueError, "model metadata"):
                summarize_study(
                    registry,
                    root / "private-results",
                    root / "aggregate",
                    participant_resamples=2,
                    crossed_resamples=2,
                )

    def test_partial_model_output_cannot_be_summarized(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry = self._write_outputs(root)
            missing = (
                root
                / "private-results"
                / "ladder-0-small"
                / "raw_effective_rank-alpha-1"
                / "summary.json"
            )
            missing.unlink()
            with self.assertRaisesRegex(ValueError, "missing analysis summary"):
                summarize_study(
                    registry,
                    root / "private-results",
                    root / "aggregate",
                    participant_resamples=2,
                    crossed_resamples=2,
                )


if __name__ == "__main__":
    unittest.main()
