import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "linear-probe-results.ipynb"


class ProbeNotebookTest(unittest.TestCase):
    def test_notebook_is_valid_clean_and_covers_both_probe_artifacts(self):
        notebook = json.loads(NOTEBOOK_PATH.read_text())
        self.assertEqual(notebook["nbformat"], 4)
        sources = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        self.assertIn("probe_metrics.csv", sources)
        self.assertIn("probe_metrics.json", sources)
        self.assertIn("confusion_matrix", sources)
        self.assertIn("CODY_JEPA_PROBE_DIR", sources)
        cell_ids = [cell.get("id") for cell in notebook["cells"]]
        self.assertNotIn(None, cell_ids)
        self.assertEqual(len(cell_ids), len(set(cell_ids)))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell.get("execution_count"))
                self.assertEqual(cell.get("outputs"), [])


if __name__ == "__main__":
    unittest.main()
