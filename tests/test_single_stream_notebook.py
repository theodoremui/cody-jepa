import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "single-stream-jepa.ipynb"


def code_cells():
    notebook = json.loads(NOTEBOOK_PATH.read_text())
    return [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]


class SingleStreamNotebookTest(unittest.TestCase):
    def test_full_training_does_not_trigger_exhaustive_data_audit(self):
        source = "\n".join(code_cells())

        self.assertIn("CODY_JEPA_RUN_FULL_TRAINING", source)
        self.assertIn(
            "RUN_DATA_AUDIT = env_flag('CODY_JEPA_RUN_DATA_AUDIT', "
            "not RUN_FULL_TRAINING)",
            source,
        )
        self.assertIn("CODY_JEPA_RUN_EXHAUSTIVE_DATA_AUDIT", source)
        self.assertIn(
            "audit_healthgait_clip_quality((train_ds, val_ds)) "
            "if RUN_DATA_AUDIT else None",
            source,
        )
        self.assertNotIn(
            "image_verify_mode='all' if RUN_FULL_TRAINING else 'sample'", source
        )
        self.assertNotIn(
            "inventory_hash_mode='full' if RUN_FULL_TRAINING else 'sample'", source
        )
        self.assertNotIn(
            "full training requires full frame decode verification and hashing", source
        )

    def test_training_path_has_bounded_real_batch_cuda_preflight(self):
        source = "\n".join(code_cells())

        self.assertIn("cuda_training_preflight", source)
        self.assertIn("preflight_batch", source)
        self.assertIn("torch.cuda.synchronize", source)
        self.assertIn("peak_gpu_memory_mib", source)
        self.assertIn("CODY_JEPA_RESUME_CHECKPOINT", source)
        self.assertIn("CODY_JEPA_OUTPUT_DIR", source)


if __name__ == "__main__":
    unittest.main()
