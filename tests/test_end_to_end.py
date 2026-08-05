import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from cody_jepa.evaluation.features import (
    INTEGER_METADATA_COLUMNS,
    METADATA_COLUMNS,
    NUMERIC_METADATA_COLUMNS,
)
from cody_jepa.gfc import CANONICAL_CELLS
from scripts.run_gfc import run_gfc


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def synthetic_rows(splits=(("train", ("train-a", "train-b")), ("val", ("dev-a", "dev-b", "dev-c")))):
    rows = []
    for split, subjects in splits:
        for subject_index, subject_id in enumerate(subjects):
            for cell_index, cell in enumerate(CANONICAL_CELLS):
                recording_id = f"{subject_id}/{cell.key}"
                learned = [
                    float(cell.speed == "UGS"),
                    float(cell.speed == "FGS"),
                    float(cell.clothing == "WoJ"),
                    float(cell.clothing == "WJ"),
                    float(cell.direction == "R2L"),
                    float(cell.direction == "L2R"),
                    float(subject_index) / 10.0,
                    float(cell_index) / 100.0,
                ]
                for window_start in (0, 8, 16):
                    row = {
                        "subject_id": subject_id,
                        "recording_id": recording_id,
                        "source_video_id": f"{subject_id}/{cell.speed}/{cell.clothing}",
                        "direction_clip_id": recording_id,
                        "split": split,
                        "speed": cell.speed,
                        "clothing": cell.clothing,
                        "direction": cell.direction,
                        "window_start": window_start,
                        "num_frames": 32,
                        "fps": 32.0,
                        "shortcut_log_frame_count": math.log(32),
                        "shortcut_duration_seconds": 1.0,
                        "shortcut_horizontal_centroid_drift_signed": 0.0,
                        "shortcut_horizontal_centroid_drift_absolute": 0.0,
                        "shortcut_foreground_area_mean": 0.5,
                        "shortcut_foreground_area_std": 0.0,
                        "shortcut_foreground_area_q25": 0.5,
                        "shortcut_foreground_area_median": 0.5,
                        "shortcut_foreground_area_q75": 0.5,
                    }
                    row.update(
                        {f"feature_{index}": value for index, value in enumerate(learned)}
                    )
                    rows.append(row)
    return rows


def corrupt_source_pair(table):
    mask = (
        (table["subject_id"] == "dev-a")
        & (table["speed"] == "UGS")
        & (table["clothing"] == "WoJ")
        & (table["direction"] == "L2R")
    )
    table.loc[mask, "source_video_id"] = "mismatched-source"


def corrupt_duration(table):
    table["shortcut_duration_seconds"] = 2.0


def corrupt_quantiles(table):
    table["shortcut_foreground_area_q25"] = 0.75


def write_runner_npz(rows, path):
    table = pd.DataFrame(rows).copy()
    table["sequence_id"] = table["recording_id"]
    table["gait_system"] = table["speed"]
    table["trial"] = table["direction"]
    feature_columns = sorted(
        (column for column in table if column.startswith("feature_")),
        key=lambda column: int(column.removeprefix("feature_")),
    )
    arrays = {
        column: table[column].to_numpy(
            dtype=(
                np.int64
                if column in INTEGER_METADATA_COLUMNS
                else np.float64
                if column in NUMERIC_METADATA_COLUMNS
                else str
            )
        )
        for column in METADATA_COLUMNS
    }
    arrays["features"] = table[feature_columns].to_numpy(dtype=np.float32)
    np.savez_compressed(path, **arrays)


class EndToEndResearchTest(unittest.TestCase):
    def test_feature_table_to_gfc_summary(self):
        rows = synthetic_rows()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = root / "features.csv"
            output = root / "gfc"
            pd.DataFrame(rows).to_csv(features, index=False)
            summary = run_gfc(
                features,
                PROJECT_ROOT / "configs" / "eval" / "gfc_healthgait.json",
                "development",
                output,
                model_label="synthetic-model",
            )

            self.assertEqual(summary["evaluation"]["participant_count"], 3)
            self.assertEqual(summary["evaluation"]["excluded_participant_count"], 0)
            self.assertEqual(summary["protocol"], "gfc_v2")
            self.assertEqual(summary["gallery"], "retain_all_8")
            self.assertEqual(summary["queries_per_participant"], 16)
            self.assertEqual(summary["factor_heads"], "three_matched_ridge_heads")
            self.assertTrue(summary["source_independence_verified"])
            self.assertAlmostEqual(summary["learned"]["top1"], 1.0)
            self.assertAlmostEqual(summary["shortcut"]["top1"], 1.0 / 8.0)
            self.assertEqual(
                {value["input_dimension"] for key, value in summary["adapter_diagnostics"].items() if key.startswith("shortcut_")},
                {9},
            )
            self.assertEqual(summary["learned_minus_shortcut"]["resamples"], 10_000)
            self.assertTrue((output / "participants.csv").is_file())
            self.assertTrue((output / "summary.csv").is_file())
            scalar = pd.read_csv(output / "summary.csv").iloc[0]
            self.assertEqual(scalar["protocol"], "gfc_v2")
            self.assertEqual(scalar["gallery"], "retain_all_8")
            self.assertEqual(scalar["queries_per_participant"], 16)
            self.assertEqual(scalar["factor_heads"], "three_matched_ridge_heads")
            self.assertTrue(scalar["source_independence_verified"])
            self.assertEqual(scalar["model_label"], "synthetic-model")
            saved = json.loads((output / "summary.json").read_text())
            self.assertEqual(saved["normalization"], "raw_retain_all")

    def test_runner_rejects_config_remapping_and_ignored_fields(self):
        base = json.loads(
            (PROJECT_ROOT / "configs" / "eval" / "gfc_healthgait.json").read_text()
        )
        mutations = (
            (lambda value: value["split_map"].update({"confirmation": "val"}), "split map"),
            (lambda value: value["adapter"].update({"fit_split": "development"}), "fit split"),
            (lambda value: value["bootstrap"].update({"resamples": 10.5}), "integer"),
            (lambda value: value["power"].update({"effect": "NaN"}), "finite number"),
            (lambda value: value["bootstrap"].update({"unused": True}), "fields"),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = root / "features.csv"
            pd.DataFrame(synthetic_rows()).to_csv(features, index=False)
            for index, (mutate, message) in enumerate(mutations):
                with self.subTest(message=message):
                    config = json.loads(json.dumps(base))
                    mutate(config)
                    config_path = root / f"config-{index}.json"
                    config_path.write_text(json.dumps(config))
                    with self.assertRaisesRegex(ValueError, message):
                        run_gfc(
                            features,
                            config_path,
                            "development",
                            root / f"out-{index}",
                            model_label="synthetic-model",
                        )

    def test_runner_rejects_unknown_splits_and_nonaggregate_output_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            table = pd.DataFrame(synthetic_rows())
            table.loc[table["subject_id"] == "dev-c", "split"] = "vla"
            features = root / "features.csv"
            table.to_csv(features, index=False)
            config = PROJECT_ROOT / "configs" / "eval" / "gfc_healthgait.json"

            with self.assertRaisesRegex(ValueError, "unsupported split labels.*vla"):
                run_gfc(
                    features,
                    config,
                    "development",
                    root / "detail-unknown",
                    model_label="synthetic-model",
                )

            with self.assertRaisesRegex(ValueError, "must not overlap"):
                run_gfc(
                    features,
                    config,
                    "development",
                    root / "same",
                    model_label="synthetic-model",
                    aggregate_output_dir=root / "same",
                )

            aggregate = root / "aggregate"
            aggregate.mkdir()
            (aggregate / "participants.csv").write_text("participant\n")
            with self.assertRaisesRegex(ValueError, "non-summary files"):
                run_gfc(
                    features,
                    config,
                    "development",
                    root / "detail-reused",
                    model_label="synthetic-model",
                    aggregate_output_dir=aggregate,
                )

    def test_runner_validates_recording_hierarchy_and_shortcuts(self):
        mutations = (
            (corrupt_source_pair, "share one source video"),
            (corrupt_duration, "duration_seconds"),
            (corrupt_quantiles, "quantiles"),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, (mutate, message) in enumerate(mutations):
                with self.subTest(message=message):
                    table = pd.DataFrame(synthetic_rows())
                    mutate(table)
                    features = root / f"features-{index}.csv"
                    table.to_csv(features, index=False)
                    with self.assertRaisesRegex(ValueError, message):
                        run_gfc(
                            features,
                            PROJECT_ROOT / "configs" / "eval" / "gfc_healthgait.json",
                            "development",
                            root / f"out-{index}",
                            model_label="synthetic-model",
                        )

    def test_heldout_rows_do_not_change_development_fit_and_outputs_are_private(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_features = root / "base.csv"
            extended_features = root / "extended.csv"
            pd.DataFrame(synthetic_rows()).to_csv(base_features, index=False)
            confirmation = synthetic_rows((("test", ("heldout-a", "heldout-b")),))
            for row in confirmation:
                for key in tuple(row):
                    if key.startswith("feature_"):
                        row[key] = 1_000_000.0 + float(row[key])
                if (
                    row["subject_id"] == "heldout-a"
                    and row["speed"] == "UGS"
                    and row["clothing"] == "WoJ"
                    and row["direction"] == "L2R"
                ):
                    row["source_video_id"] = "mismatched-heldout-source"
            incomplete = synthetic_rows((("val", ("dev-incomplete",)),))
            omitted_recording = incomplete[0]["recording_id"]
            incomplete = [
                row for row in incomplete if row["recording_id"] != omitted_recording
            ]
            pd.DataFrame([*synthetic_rows(), *confirmation, *incomplete]).to_csv(
                extended_features, index=False
            )
            first = run_gfc(
                base_features,
                PROJECT_ROOT / "configs" / "eval" / "gfc_healthgait.json",
                "development",
                root / "first",
                model_label="synthetic-model",
                write_queries=True,
            )
            second = run_gfc(
                extended_features,
                PROJECT_ROOT / "configs" / "eval" / "gfc_healthgait.json",
                "development",
                root / "second",
                model_label="synthetic-model",
                write_queries=True,
                aggregate_output_dir=root / "aggregate",
            )
            self.assertEqual(first["learned"], second["learned"])
            self.assertEqual(first["shortcut"], second["shortcut"])
            self.assertEqual(first["adapter_diagnostics"], second["adapter_diagnostics"])
            self.assertEqual(
                first["normalizer_diagnostics"], second["normalizer_diagnostics"]
            )
            serialized = "\n".join(
                path.read_text()
                for path in (root / "second").iterdir()
                if path.suffix in {".csv", ".json", ".jsonl"}
            )
            for private_id in ("dev-a", "dev-b", "dev-c", "dev-incomplete"):
                self.assertNotIn(private_id, serialized)
            self.assertEqual(
                {path.name for path in (root / "aggregate").iterdir()},
                {"summary.csv", "summary.json"},
            )
            aggregate = (root / "aggregate" / "summary.json").read_text()
            self.assertNotIn('"participant":', aggregate)
            self.assertNotIn("participant_000", aggregate)
            self.assertIn('"participant_count": 1', aggregate)
            self.assertTrue((root / "second" / "queries.jsonl").is_file())
            first_queries = (root / "first" / "queries.jsonl").read_text().splitlines()
            second_queries = (root / "second" / "queries.jsonl").read_text().splitlines()
            self.assertEqual(first_queries, second_queries)
            self.assertEqual(len(second_queries), 3 * 2 * 16)
            for line in second_queries:
                query = json.loads(line)
                self.assertEqual(query["protocol"], "gfc_v2")
                self.assertIn(query["focal_factor"], {"speed", "clothing"})
                self.assertEqual(len(query["gallery"]), 8)
                self.assertTrue(query["source_independence_verified"])
                self.assertNotIn("source_video_id", line)
            run_gfc(
                extended_features,
                PROJECT_ROOT / "configs" / "eval" / "gfc_healthgait.json",
                "development",
                root / "second",
                model_label="synthetic-model",
                write_queries=False,
            )
            self.assertFalse((root / "second" / "queries.jsonl").exists())

    def test_npz_heldout_rows_are_isolated_before_semantic_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = synthetic_rows(
                (
                    ("train", ("train-a", "train-b")),
                    ("val", ("dev-a", "dev-b", "dev-c")),
                    ("test", ("heldout-a",)),
                )
            )
            for row in rows:
                if (
                    row["subject_id"] == "heldout-a"
                    and row["direction"] == "L2R"
                ):
                    row["source_video_id"] = "invalid-heldout-source"
            features = root / "features.npz"
            write_runner_npz(rows, features)
            summary = run_gfc(
                features,
                PROJECT_ROOT / "configs" / "eval" / "gfc_healthgait.json",
                "development",
                root / "output",
                model_label="synthetic-model",
            )
            self.assertEqual(summary["evaluation"]["participant_count"], 3)


if __name__ == "__main__":
    unittest.main()
