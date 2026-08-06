import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from cody_jepa.cli.make_gfc_study_results import (
    LADDER_TABLE_COLUMNS,
    OUTPUT_FILENAMES,
    RUN_TABLE_COLUMNS,
    make_gfc_study_results,
)


LADDERS = [f"ladder-{index}" for index in range(1, 6)]
RUNGS = ["small", "medium", "large", "full"]


def _write_aggregate_inputs(root: Path) -> None:
    root.mkdir()
    run_rows = []
    ladder_rows = []
    contrasts = []
    for ladder_index, ladder in enumerate(LADDERS):
        rung_values = [0.40 + 0.01 * ladder_index + 0.02 * rung for rung in range(4)]
        for rung_index, (rung, learned) in enumerate(zip(RUNGS, rung_values)):
            run_rows.append(
                {
                    "model_label": f"{ladder}-{rung}",
                    "ladder": ladder,
                    "rung": rung,
                    "checkpoint_id": f"checkpoint-{ladder_index}-{rung_index}",
                    "pool_seed": ladder_index,
                    "optimization_seed": rung_index,
                    "unique_sequences": 2_500 * (10**rung_index),
                    "training_exposure": 1_000_000,
                    "primary_analysis_id": "raw_retain_all-alpha-1",
                    "primary_ridge_alpha": 1.0,
                    "primary_normalization": "raw_retain_all",
                    "learned_top1": learned,
                    "learned_mrr": learned + 0.1,
                    "shortcut_top1": 0.25,
                    "shortcut_mrr": 0.4,
                    "hard_control_top1": learned - 0.01,
                    "hard_control_mrr": learned + 0.05,
                    "soft_control_top1": learned - 0.01,
                    "soft_control_mrr": learned + 0.06,
                    "soft_control_target_probability": 0.3,
                    "soft_control_target_nll": 1.2,
                    "soft_temperature": 0.8,
                    "alpha_0_1_learned_top1": learned - 0.01,
                    "alpha_10_learned_top1": learned + 0.01,
                    "raw_effective_rank_learned_top1": learned - 0.02,
                    "pca_effective_rank_learned_top1": learned + 0.02,
                }
            )
        contrast = rung_values[-1] - rung_values[0]
        contrasts.append({"ladder": ladder, "full_minus_small": contrast})
        ladder_rows.append(
            {
                "ladder": ladder,
                "participant_count": 308,
                "small_top1": rung_values[0],
                "medium_top1": rung_values[1],
                "large_top1": rung_values[2],
                "full_top1": rung_values[3],
                "full_minus_small": contrast,
                "participant_bootstrap_95_lower": contrast - 0.02,
                "participant_bootstrap_95_upper": contrast + 0.02,
            }
        )
    pd.DataFrame(reversed(run_rows), columns=RUN_TABLE_COLUMNS).to_csv(
        root / "run_table.csv", index=False
    )
    pd.DataFrame(reversed(ladder_rows), columns=LADDER_TABLE_COLUMNS).to_csv(
        root / "ladder_contrasts.csv", index=False
    )
    summary = {
        "schema_version": "gfc-v2-study-aggregate-v1",
        "protocol": "gfc_v2",
        "gallery": "retain_all_8",
        "queries_per_participant": 16,
        "analysis_id": "raw_retain_all-alpha-1",
        "ridge_alpha": 1.0,
        "normalization": "raw_retain_all",
        "revision": {
            "code_commit": "a" * 40,
            "analysis_freeze_tag": "gfc-v2-analysis-freeze-v1",
        },
        "cohort_roles": {
            "version": "healthgait-gfc-v2-roles-v1",
            "fit_role": "development",
            "evaluation_role": "locked_outcome",
            "assigned_counts": {"development": 80, "locked_outcome": 318},
            "complete_counts": {"development": 76, "locked_outcome": 308},
            "excluded_counts": {"development": 4, "locked_outcome": 10},
        },
        "inference": {
            "ladder_count": 5,
            "participant_count": 308,
            "rung_order": RUNGS,
            "resolution": 1 / 16,
            "mean_full_minus_small": sum(row["full_minus_small"] for row in contrasts)
            / 5,
            "ladder_contrasts": contrasts,
            "t_interval_95": {"confidence_level": 0.95, "lower": 0.03, "upper": 0.09},
            "t_interval_90": {"confidence_level": 0.90, "lower": 0.04, "upper": 0.08},
            "crossed_bootstrap_95": {
                "confidence_level": 0.95,
                "lower": 0.02,
                "upper": 0.10,
                "resamples": 10_000,
                "seed": 7,
            },
            "superiority": True,
            "equivalence": False,
            "decision": "Positive but small",
        },
        "artifacts": {
            "run_table": "run_table.csv",
            "run_table_rows": 20,
            "ladder_contrasts": "ladder_contrasts.csv",
            "ladder_contrast_rows": 5,
        },
    }
    (root / "outcome_summary.json").write_text(json.dumps(summary), encoding="utf-8")


class GfcStudyResultRendererTest(unittest.TestCase):
    def test_renders_exact_tables_and_both_figure_formats_in_stable_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            aggregate = root / "aggregate"
            output = root / "paper"
            _write_aggregate_inputs(aggregate)
            paths = make_gfc_study_results(aggregate, output)

            self.assertEqual([path.name for path in paths], list(OUTPUT_FILENAMES))
            run_table = pd.read_csv(output / "gfc_study_run_table.csv")
            ladder_table = pd.read_csv(output / "gfc_study_ladder_contrasts.csv")
            self.assertEqual(len(run_table), 20)
            self.assertEqual(len(ladder_table), 5)
            self.assertEqual(tuple(run_table.columns), RUN_TABLE_COLUMNS)
            self.assertEqual(tuple(ladder_table.columns), LADDER_TABLE_COLUMNS)
            self.assertEqual(list(run_table["ladder"][:4]), [LADDERS[0]] * 4)
            self.assertEqual(list(run_table["rung"][:4]), RUNGS)
            self.assertEqual(list(ladder_table["ladder"]), LADDERS)
            self.assertGreater((output / "gfc_study_scaling.png").stat().st_size, 1_000)
            self.assertTrue(
                (output / "gfc_study_scaling.pdf").read_bytes().startswith(b"%PDF")
            )

    def test_rejects_legacy_private_and_malformed_inputs_and_removes_stale_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            aggregate = root / "aggregate"
            output = root / "paper"
            _write_aggregate_inputs(aggregate)
            output.mkdir()
            for name in OUTPUT_FILENAMES:
                (output / name).write_text("stale", encoding="utf-8")

            summary_path = aggregate / "outcome_summary.json"
            summary = json.loads(summary_path.read_text())
            summary["protocol"] = "legacy_donor_excluded_v1"
            summary_path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "legacy and mixed"):
                make_gfc_study_results(aggregate, output)
            self.assertFalse(any((output / name).exists() for name in OUTPUT_FILENAMES))

            summary["protocol"] = "gfc_v2"
            summary["cohort_roles"]["role_map_path"] = "/private/roles.csv"
            summary_path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "private field"):
                make_gfc_study_results(aggregate, output)

            summary["cohort_roles"].pop("role_map_path")
            summary["participants"] = [{"participant": "private-000"}]
            summary_path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "private field"):
                make_gfc_study_results(aggregate, output)

            summary.pop("participants")
            summary["inference"]["participant_count"] = 307
            summary_path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "all 308 complete outcome"):
                make_gfc_study_results(aggregate, output)

            summary["inference"]["participant_count"] = 308
            summary["queries_per_participant"] = 16.9
            summary_path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "queries per participant"):
                make_gfc_study_results(aggregate, output)

            summary["queries_per_participant"] = 16
            summary_path.write_text(json.dumps(summary))
            table = pd.read_csv(aggregate / "run_table.csv")
            table["participant_id"] = "forbidden"
            table.to_csv(aggregate / "run_table.csv", index=False)
            with self.assertRaisesRegex(ValueError, "private columns"):
                make_gfc_study_results(aggregate, output)


if __name__ == "__main__":
    unittest.main()
