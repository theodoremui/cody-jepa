import copy
from pathlib import Path
import unittest

from cody_jepa.config.gfc import load_gfc_config, validate_gfc_config
from cody_jepa.evaluation.gfc.oracle import compile_healthgait_gfc_v2_protocol


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "eval" / "gfc_healthgait.json"


class GFCConfigTest(unittest.TestCase):
    def setUp(self):
        self.config = load_gfc_config(CONFIG_PATH)

    def test_checked_in_config_matches_compiled_v2_protocol(self):
        protocol = compile_healthgait_gfc_v2_protocol()
        self.assertEqual(self.config["protocol"]["name"], protocol.name)
        self.assertEqual(
            self.config["protocol"]["focal_factors"], list(protocol.focal_factors)
        )
        self.assertEqual(self.config["protocol"]["gallery_policy"], "retain_all")
        self.assertEqual(len(protocol.design.cells), 8)
        self.assertEqual(len(protocol.queries), 16)
        self.assertTrue(all(len(query.gallery) == 8 for query in protocol.queries))
        self.assertEqual(self.config["power"]["effect"], 1.0 / len(protocol.queries))

    def test_legacy_protocol_sections_and_redundant_counts_are_rejected(self):
        for section in ("query", "gallery"):
            with self.subTest(section=section):
                value = copy.deepcopy(self.config)
                value[section] = {}
                with self.assertRaisesRegex(ValueError, "top-level config fields"):
                    validate_gfc_config(value)
        mutations = (
            ("protocol", "queries_per_participant", 16),
            ("protocol", "gallery_size", 8),
            ("complete_case", "required_cells", 8),
        )
        for section, field, value in mutations:
            with self.subTest(field=field):
                config = copy.deepcopy(self.config)
                config[section][field] = value
                with self.assertRaisesRegex(ValueError, "fields"):
                    validate_gfc_config(config)

    def test_protocol_settings_are_strict(self):
        mutations = {
            "name": "legacy_gfc",
            "donor_rule": "single_donor",
            "focal_factors": ["speed", "clothing", "direction"],
            "gallery_policy": "exclude_donors",
            "require_target_source_independence": False,
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                config = copy.deepcopy(self.config)
                config["protocol"][field] = value
                with self.assertRaisesRegex(ValueError, "unsupported"):
                    validate_gfc_config(config)
        config = copy.deepcopy(self.config)
        config["protocol"]["require_target_source_independence"] = 1
        with self.assertRaisesRegex(ValueError, "unsupported"):
            validate_gfc_config(config)

    def test_factor_heads_distance_and_shortcut_schema_are_strict(self):
        self.assertEqual(
            self.config["adapter"]["factor_names"],
            ["speed", "clothing", "direction"],
        )
        self.assertEqual(len(self.config["shortcut"]["columns"]), 9)
        self.assertEqual(self.config["distance"]["factor_aggregation"], "equal_mean")
        mutations = (
            ("adapter", "condition_labels", ["WoJ", "WJ"]),
            ("shortcut", "condition_columns", []),
            ("distance", "condition_weight", 0.5),
        )
        for section, field, value in mutations:
            with self.subTest(section=section, field=field):
                config = copy.deepcopy(self.config)
                config[section][field] = value
                with self.assertRaisesRegex(ValueError, "fields"):
                    validate_gfc_config(config)


if __name__ == "__main__":
    unittest.main()
