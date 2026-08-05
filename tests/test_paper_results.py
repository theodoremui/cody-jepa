import json
from pathlib import Path
import shutil
import tempfile
import unittest

import pandas as pd

from scripts.make_paper_results import (
    CONTEXT_LABELS,
    GFC_NORMALIZATIONS,
    LEGACY_GFC_PROTOCOL,
    RANK_LABELS,
    make_paper_results,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def gfc_summary(normalization, model_label="synthetic-model"):
    return {
        "protocol": LEGACY_GFC_PROTOCOL,
        "split": "development",
        "normalization": normalization,
        "model_label": model_label,
        "seed": 7,
        "feature_dimension": 8,
        "method_settings": {"adapter": {"alpha": 1.0}, "distance": {"metric": "cosine"}},
        "training": {"participant_count": 2, "recording_count": 16},
        "evaluation": {
            "participant_count": 3,
            "excluded_participant_count": 0,
            "recording_count": 24,
        },
        "learned": {"top1": 0.7, "mrr": 0.8, "donor_attraction": 0.6},
        "shortcut": {"top1": 0.2, "mrr": 0.4, "donor_attraction": 0.5},
        "learned_minus_shortcut": {
            "point_estimate": 0.5,
            "confidence_level": 0.95,
            "confidence_interval": {"lower": 0.55, "upper": 0.6},
            "resamples": 100,
        },
    }


class PaperResultTest(unittest.TestCase):
    def _copy_compact_inputs(self, destination):
        destination.mkdir()
        for name in ("phase0_summary.json", "phase1_summary.csv", "context_diagnosis.json"):
            shutil.copy2(PROJECT_ROOT / "results" / name, destination / name)

    def test_gfc_outputs_require_all_analyses_and_remove_stale_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            generated = root / "generated"
            self._copy_compact_inputs(results)
            first = results / "gfc-primary"
            first.mkdir()
            (first / "summary.json").write_text(
                json.dumps(gfc_summary("raw_retain_all"))
            )
            with self.assertRaisesRegex(ValueError, "both sensitivities"):
                make_paper_results(results, generated)

            without_protocol = gfc_summary("raw_retain_all")
            without_protocol.pop("protocol")
            (first / "summary.json").write_text(json.dumps(without_protocol))
            with self.assertRaisesRegex(ValueError, "must declare protocol"):
                make_paper_results(results, generated)
            v2_summary = gfc_summary("raw_retain_all")
            v2_summary["protocol"] = "gfc_v2"
            (first / "summary.json").write_text(json.dumps(v2_summary))
            with self.assertRaisesRegex(ValueError, "v2 and mixed summaries"):
                make_paper_results(results, generated)
            (first / "summary.json").write_text(
                json.dumps(gfc_summary("raw_retain_all"))
            )

            for normalization in GFC_NORMALIZATIONS - {"raw_retain_all"}:
                target = results / f"gfc-{normalization}"
                target.mkdir()
                (target / "summary.json").write_text(
                    json.dumps(gfc_summary(normalization))
                )
            incompatible_path = results / "gfc-raw_effective_rank" / "summary.json"
            incompatible = json.loads(incompatible_path.read_text())
            incompatible["method_settings"]["adapter"]["alpha"] = 2.0
            incompatible_path.write_text(json.dumps(incompatible))
            with self.assertRaisesRegex(ValueError, "incompatible"):
                make_paper_results(results, generated)
            incompatible_path.write_text(json.dumps(gfc_summary("raw_effective_rank")))
            paths = make_paper_results(results, generated)
            self.assertIn(generated / "legacy_gfc_table.csv", paths)
            self.assertTrue((generated / "legacy_gfc_comparison.png").is_file())

            for normalization in GFC_NORMALIZATIONS:
                target = results / f"gfc-second-{normalization}"
                target.mkdir()
                (target / "summary.json").write_text(
                    json.dumps(gfc_summary(normalization, "second-model"))
                )
                confirmation = gfc_summary(normalization)
                confirmation["split"] = "confirmation"
                confirmation_target = results / f"gfc-confirmation-{normalization}"
                confirmation_target.mkdir()
                (confirmation_target / "summary.json").write_text(
                    json.dumps(confirmation)
                )
            make_paper_results(results, generated)
            table = pd.read_csv(generated / "legacy_gfc_table.csv")
            self.assertEqual(len(table), 9)
            self.assertEqual(
                list(
                    zip(
                        table["model_label"],
                        table["split"],
                        table["normalization"],
                    )
                ),
                [
                    ("second-model", "development", "raw_retain_all"),
                    ("second-model", "development", "raw_effective_rank"),
                    ("second-model", "development", "pca_effective_rank"),
                    ("synthetic-model", "confirmation", "raw_retain_all"),
                    ("synthetic-model", "confirmation", "raw_effective_rank"),
                    ("synthetic-model", "confirmation", "pca_effective_rank"),
                    ("synthetic-model", "development", "raw_retain_all"),
                    ("synthetic-model", "development", "raw_effective_rank"),
                    ("synthetic-model", "development", "pca_effective_rank"),
                ],
            )

            for path in results.glob("gfc-*"):
                shutil.rmtree(path)
            make_paper_results(results, generated)
            self.assertFalse((generated / "legacy_gfc_table.csv").exists())
            self.assertFalse((generated / "legacy_gfc_comparison.png").exists())

    def test_context_figure_labels_cover_every_compact_result_value(self):
        value = json.loads((PROJECT_ROOT / "results" / "context_diagnosis.json").read_text())
        context = {
            row["condition"]
            for row in value["context_substitution"]
            if row["condition"] != "self"
        }
        ranks = {row["representation"] for row in value["representation_rank"]}
        self.assertEqual(context, set(CONTEXT_LABELS))
        self.assertEqual(ranks, set(RANK_LABELS))


if __name__ == "__main__":
    unittest.main()
