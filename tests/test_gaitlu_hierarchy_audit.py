import csv
from pathlib import Path
import tempfile
import unittest

from cody_jepa.data.gaitlu_hierarchy_audit import (
    anchor_count,
    audit_hierarchical_support,
)


def write_inventory(path: Path, *, count: int, frames: int):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("sequence_id", "source_group", "num_frames", "eligible"),
        )
        writer.writeheader()
        for index in range(count):
            writer.writerow(
                {
                    "sequence_id": f"sequence-{index}",
                    "source_group": f"group-{index}",
                    "num_frames": frames,
                    "eligible": "true",
                }
            )


class TestGaitLUHierarchyAudit(unittest.TestCase):
    def test_anchor_count(self):
        self.assertEqual(anchor_count(15), 0)
        self.assertEqual(anchor_count(16), 1)
        self.assertEqual(anchor_count(24), 2)
        self.assertEqual(anchor_count(40), 4)

    def test_gate_passes_with_four_anchors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inventory.csv"
            write_inventory(path, count=30, frames=40)
            report = audit_hierarchical_support(
                path,
                draws=100_000,
                holdout_size=2,
                pool_seeds=(0, 1),
                low_target=4,
                high_target=12,
            )
        self.assertTrue(report["gate_pass"])
        self.assertEqual(len(report["pool_audits"]), 4)
        self.assertTrue(
            all(row["median_8_frame_anchors"] == 4 for row in report["pool_audits"])
        )
        self.assertTrue(
            all(
                row["mean_distinct_window_overlap_fraction"] == 0.25
                for row in report["pool_audits"]
            )
        )

    def test_gate_fails_when_median_has_only_two_anchors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inventory.csv"
            write_inventory(path, count=30, frames=24)
            report = audit_hierarchical_support(
                path,
                draws=100_000,
                holdout_size=2,
                pool_seeds=(0, 1),
                low_target=4,
                high_target=12,
            )
        self.assertFalse(report["gate_pass"])
        self.assertTrue(report["failure_reasons"])


if __name__ == "__main__":
    unittest.main()
