import ast
import csv
import json
import math
from pathlib import Path
import tempfile
import unittest

import pandas as pd
import torch

from cody_jepa.cli.export_histories import export_histories


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HISTORY_PATH = PROJECT_ROOT / "results" / "checkpoint_histories.csv"
METADATA_PATH = PROJECT_ROOT / "results" / "checkpoint_histories.json"
HASH_METADATA_TERMS = ("sha256", "checksum", "digest", "hash")
HASH_CALLS = {
    "hash",
    "md5",
    "sha1",
    "sha224",
    "sha256",
    "sha384",
    "sha512",
    "blake2b",
    "blake2s",
}


def _hash_metadata_keys(value, location="root"):
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{location}.{key}"
            if any(term in str(key).casefold() for term in HASH_METADATA_TERMS):
                found.append(child)
            found.extend(_hash_metadata_keys(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_hash_metadata_keys(item, f"{location}[{index}]"))
    return found


def _implements_hashing(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(
            item.name == "hashlib" for item in node.names
        ):
            return True
        if isinstance(node, ast.ImportFrom) and node.module == "hashlib":
            return True
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else None
            attribute = node.func.attr if isinstance(node.func, ast.Attribute) else None
            if name in HASH_CALLS or attribute in HASH_CALLS:
                return True
    return False


class CheckpointHistoryExportTest(unittest.TestCase):
    def test_runtime_hashing_is_limited_to_gaitlu_data_integrity(self):
        implementations = {
            path.relative_to(PROJECT_ROOT).as_posix()
            for path in (PROJECT_ROOT / "src").rglob("*.py")
            if _implements_hashing(path)
        }
        self.assertEqual(
            implementations,
            {
                "src/cody_jepa/data/gaitlu.py",
                "src/cody_jepa/data/gaitlu_prepare.py",
            },
        )

    def test_tracked_result_metadata_contains_no_hash_fields(self):
        results_root = PROJECT_ROOT / "results"
        violations = []
        for path in results_root.rglob("*.json"):
            value = json.loads(path.read_text(encoding="utf-8"))
            violations.extend(
                f"{path.relative_to(PROJECT_ROOT)}:{key}"
                for key in _hash_metadata_keys(value)
            )
        for path in results_root.rglob("*.csv"):
            with path.open(encoding="utf-8", newline="") as handle:
                columns = csv.DictReader(handle).fieldnames or []
            violations.extend(
                f"{path.relative_to(PROJECT_ROOT)}:{column}"
                for column in columns
                if any(term in column.casefold() for term in HASH_METADATA_TERMS)
            )
        self.assertEqual(violations, [])

    def test_export_is_flat_deterministic_and_json_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "latest.pt"
            state = {
                "schema": 4,
                "architecture": "synthetic",
                "torch_version": torch.__version__,
                "config": {"num_epochs": 2},
                "mask_groups": [{"label": "test"}],
                "data_contract": {"dataset": "synthetic"},
                "history": [
                    {
                        "epoch": 1,
                        "step": 2,
                        "train_loss": 1.0,
                        "val": None,
                        "train_eval": None,
                    },
                    {
                        "epoch": 2,
                        "step": 4,
                        "train_loss": 0.5,
                        "val": {
                            "loss": 0.75,
                            "representations_healthy": True,
                            "health_issues": [],
                        },
                        "train_eval": None,
                    },
                ],
                "completed_epochs": 2,
                "global_step": 4,
                "best_val_loss": 0.75,
                "best_epoch": 2,
                "best_healthy_val_loss": math.inf,
                "best_healthy_epoch": None,
            }
            torch.save(state, checkpoint)
            csv_path = root / "history.csv"
            metadata_path = root / "history.json"
            export_histories(
                [("synthetic-run", "test", checkpoint)],
                csv_path,
                metadata_path,
            )

            with csv_path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["val_loss"], "")
            self.assertEqual(rows[1]["val_loss"], "0.75")
            self.assertEqual(rows[1]["val_representations_healthy"], "true")
            self.assertEqual(rows[1]["val_health_issues"], "[]")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertIsNone(metadata["runs"][0]["best_healthy_val_loss"])
            self.assertEqual(metadata["runs"][0]["evaluation_epochs"], [2])
            self.assertFalse(_hash_metadata_keys(metadata))

    def test_committed_histories_are_complete_and_match_summaries(self):
        histories = pd.read_csv(HISTORY_PATH)
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        expected_counts = {
            "phase0-job-91108": 100,
            **{f"a{index:02d}": 40 for index in range(8)},
            **{f"b{index:02d}": 100 for index in range(3)},
        }
        observed_counts = {
            run_id.split("-", 1)[0] if run_id != "phase0-job-91108" else run_id: count
            for run_id, count in histories.groupby("run_id").size().items()
        }
        self.assertEqual(observed_counts, expected_counts)
        self.assertEqual(len(histories), 720)
        self.assertFalse(histories.duplicated(["run_id", "epoch"]).any())
        self.assertEqual(metadata["row_count"], len(histories))
        self.assertEqual(metadata["columns"], histories.columns.tolist())
        self.assertFalse(_hash_metadata_keys(metadata))

        phase0 = json.loads(
            (PROJECT_ROOT / "results" / "phase0_summary.json").read_text(encoding="utf-8")
        )
        phase0_history = histories.loc[histories["run_id"].eq("phase0-job-91108")]
        for checkpoint in phase0["checkpoints"]:
            row = phase0_history.loc[phase0_history["epoch"].eq(checkpoint["epoch"])].iloc[0]
            self.assertAlmostEqual(row["val_loss"], checkpoint["validation_loss"])
            self.assertAlmostEqual(row["val_effective_rank"], checkpoint["effective_rank"])
            self.assertAlmostEqual(
                row["val_subject_balanced_context_shuffle_loss_gap"],
                checkpoint["wrong_context_gap"],
            )

        phase1 = pd.read_csv(PROJECT_ROOT / "results" / "phase1_summary.csv")
        for summary in phase1.itertuples(index=False):
            row = histories.loc[
                histories["run_id"].eq(summary.run_id)
                & histories["epoch"].eq(summary.selected_epoch)
            ].iloc[0]
            self.assertAlmostEqual(row["val_loss"], summary.validation_loss)
            self.assertAlmostEqual(row["val_effective_rank"], summary.effective_rank)
            self.assertAlmostEqual(
                row["val_subject_balanced_context_shuffle_loss_gap"],
                summary.wrong_context_gap,
            )

    def test_notebook_sources_use_portable_result_inputs(self):
        for notebook_path in sorted((PROJECT_ROOT / "notebooks").glob("*.ipynb")):
            notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
            source = "\n".join(
                "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
            )
            self.assertNotIn("reports/", source, notebook_path.name)
            self.assertNotIn("outputs/", source, notebook_path.name)
            self.assertNotIn("load_checkpoint", source, notebook_path.name)
            self.assertIn("results", source, notebook_path.name)


if __name__ == "__main__":
    unittest.main()
