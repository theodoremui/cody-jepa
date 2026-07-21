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

    def test_healthy_checkpoint_is_reported_only_when_it_was_selected_and_written(self):
        source = "\n".join(code_cells())

        self.assertIn(
            "best_healthy_path = healthy_checkpoint_path(",
            source,
        )
        self.assertIn(
            "str(best_healthy_path) if best_healthy_path else None",
            source,
        )
        self.assertIn("OUTPUT_DIR, result['best_healthy_epoch']", source)
        self.assertNotIn(
            "'best_healthy': str(OUTPUT_DIR / 'best_healthy.pt')",
            source,
        )

    def test_plots_name_corrected_online_and_subject_balanced_diagnostics(self):
        source = "\n".join(code_cells())

        self.assertIn("subject_balanced_context_shuffle_loss_gap", source)
        self.assertIn("Online full-view effective rank", source)
        self.assertIn("Subject-balanced wrong-context loss gap", source)
        self.assertNotIn("shortcut_diagnostic_batches", source)


if __name__ == "__main__":
    unittest.main()
