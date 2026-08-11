import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs" / "train"


def _read_config(name):
    return json.loads((CONFIG_ROOT / name).read_text())


class GaitLUHierarchyConfigTests(unittest.TestCase):
    def test_production_exposure_tiers_have_exact_clip_totals(self):
        expected = {
            "gaitlu_hierarchy_full.json": {
                "loader_epoch_examples": 65_536,
                "num_epochs": 125,
                "steps": 128_000,
                "warmup_steps": 2_000,
                "training_exposure": 8_192_000,
            },
            "gaitlu_hierarchy_half.json": {
                "loader_epoch_examples": 64_000,
                "num_epochs": 64,
                "steps": 64_000,
                "warmup_steps": 1_000,
                "training_exposure": 4_096_000,
            },
        }
        for filename, tier in expected.items():
            with self.subTest(filename=filename):
                config = _read_config(filename)
                effective_batch = config["batch_size"] * config["accumulation_steps"]
                self.assertEqual(effective_batch, 64)
                self.assertEqual(config["loader_epoch_examples"], tier["loader_epoch_examples"])
                self.assertEqual(config["num_epochs"], tier["num_epochs"])
                self.assertEqual(config["steps"], tier["steps"])
                self.assertEqual(config["warmup_steps"], tier["warmup_steps"])
                self.assertEqual(config["steps"] * effective_batch, tier["training_exposure"])
                self.assertEqual(
                    config["loader_epoch_examples"] * config["num_epochs"],
                    tier["training_exposure"],
                )

    def test_production_tiers_only_change_exposure_schedule(self):
        template = _read_config("gaitlu_scaling.json")
        allowed_changes = {
            "run_id",
            "loader_epoch_examples",
            "steps",
            "num_epochs",
            "warmup_steps",
        }
        for filename in ("gaitlu_hierarchy_full.json", "gaitlu_hierarchy_half.json"):
            with self.subTest(filename=filename):
                config = _read_config(filename)
                changed = {
                    key
                    for key in template
                    if config.get(key) != template[key]
                }
                self.assertEqual(set(config), set(template))
                self.assertLessEqual(changed, allowed_changes)
                self.assertEqual(config["train_horizontal_flip_prob"], 0.0)

    def test_smoke_is_one_real_cpu_update_and_visibly_nonproduction(self):
        config = _read_config("gaitlu_hierarchy_smoke.json")
        self.assertEqual(config["batch_size"] * config["accumulation_steps"], 64)
        self.assertEqual(config["loader_epoch_examples"], 64)
        self.assertEqual(config["steps"], 1)
        self.assertEqual(config["num_epochs"], 1)
        self.assertEqual(config["required_device"], "cpu")
        self.assertIsNone(config["amp_dtype"])
        self.assertFalse(config["compile"])
        self.assertFalse(config["tf32"])
        self.assertIn("smoke", config["run_id"])
        self.assertNotEqual(config["embed_dim"], 384)


if __name__ == "__main__":
    unittest.main()
