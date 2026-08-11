import csv
from pathlib import Path
import tempfile
import unittest

from cody_jepa.data.gaitlu import GAITLU_MANIFEST_COLUMNS
from cody_jepa.data.gaitlu_hierarchy import (
    HIERARCHY_REGISTRY_COLUMNS,
    finalize_gaitlu_hierarchy,
    read_hierarchy_registry,
)
from cody_jepa.data.gaitlu_prepare import FINAL_INVENTORY_COLUMNS


def read_csv(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def inventory_row(index: int, *, frames=24, eligible="true"):
    return {
        "sequence_id": f"sequence-{index:03d}",
        "source_shard": "synthetic.tar.gz",
        "source_member": f"sequence-{index:03d}.pkl",
        "shard_path": "shards/synthetic.tar",
        "record_offset": 512 + index * 64,
        "record_size": 64,
        "num_frames": frames,
        "height": 8,
        "width": 8,
        "content_sha256": f"{index:064x}",
        "foreground_fraction": 0.25,
        "empty_frame_fraction": 0.0,
        "intermediate_fraction": 0.0,
        "valid": "true",
        "exclusion_reason": "",
        "source_group": f"group-{index // 2:03d}",
        "duplicate_of": "",
        "eligible": eligible,
        "cohort": "training" if eligible == "true" else "excluded",
    }


def write_inventory(root: Path, *, eligible_count=44):
    rows = [inventory_row(index) for index in range(eligible_count)]
    rows.extend(
        [
            inventory_row(eligible_count, frames=16),
            inventory_row(eligible_count + 1, frames=23),
            inventory_row(eligible_count + 2, frames=40, eligible="false"),
        ]
    )
    write_csv(root / "inventory.csv", FINAL_INVENTORY_COLUMNS, rows)


def finalize(root: Path):
    return finalize_gaitlu_hierarchy(
        root,
        training_exposure=4_096,
        holdout_target=4,
        holdout_seed=99,
        pool_seeds=tuple(range(8)),
        low_target=4,
        high_target=12,
    )


class GaitLUHierarchyFinalizerTest(unittest.TestCase):
    def test_finalizer_filters_temporal_support_and_builds_nested_factorial_registry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_inventory(root)
            summary = finalize(root)
            hierarchy = root / "hierarchy"
            registry = read_hierarchy_registry(hierarchy / "training_registry.csv")

            self.assertEqual(summary["eligible_after_temporal_rule"], 44)
            self.assertEqual(summary["excluded_by_temporal_rule"], 2)
            self.assertEqual(summary["holdout_sequences"], 4)
            self.assertEqual(summary["training_manifests"], 16)
            self.assertEqual(summary["registry_rows"], 32)
            self.assertEqual(len(list((hierarchy / "manifests").glob("replicate-*.csv"))), 16)
            self.assertEqual(len(registry), 32)

            holdout = read_csv(hierarchy / "manifests" / "common-holdout.csv")
            holdout_groups = {row["source_group"] for row in holdout}
            self.assertEqual(len(holdout_groups), 2)
            self.assertTrue(
                all(sum(row["source_group"] == group for row in holdout) == 2 for group in holdout_groups)
            )
            for replicate in range(8):
                block = [row for row in registry if row["replicate"] == replicate]
                self.assertEqual(
                    {(row["sequence_support"], row["window_policy"]) for row in block},
                    {
                        ("low", "frozen_random"),
                        ("low", "resampled_anchor"),
                        ("high", "frozen_random"),
                        ("high", "resampled_anchor"),
                    },
                )
                self.assertEqual(len({row["optimization_seed"] for row in block}), 1)
                self.assertEqual(len({row["replicate_seed"] for row in block}), 1)
                for support, expected_count in (("low", 4), ("high", 12)):
                    pair = [row for row in block if row["sequence_support"] == support]
                    self.assertEqual(len({row["train_manifest"] for row in pair}), 1)
                    self.assertEqual({row["unique_sequences"] for row in pair}, {expected_count})

                low = read_csv(hierarchy / f"manifests/replicate-{replicate}-low.csv")
                high = read_csv(hierarchy / f"manifests/replicate-{replicate}-high.csv")
                self.assertLess(
                    {row["sequence_id"] for row in low},
                    {row["sequence_id"] for row in high},
                )
                self.assertLess(
                    {row["source_group"] for row in low},
                    {row["source_group"] for row in high},
                )
                self.assertFalse({row["source_group"] for row in high} & holdout_groups)
            self.assertEqual({row["training_exposure"] for row in registry}, {4_096})

    def test_same_seeds_produce_byte_identical_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            roots = [Path(directory) / name for name in ("first", "second")]
            for root in roots:
                root.mkdir()
                write_inventory(root)
                finalize(root)
            first_files = {
                path.relative_to(roots[0] / "hierarchy"): path.read_bytes()
                for path in (roots[0] / "hierarchy").rglob("*.csv")
            }
            second_files = {
                path.relative_to(roots[1] / "hierarchy"): path.read_bytes()
                for path in (roots[1] / "hierarchy").rglob("*.csv")
            }
            self.assertEqual(first_files, second_files)

    def test_finalizer_rejects_insufficient_support_and_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_inventory(root, eligible_count=14)
            with self.assertRaisesRegex(ValueError, "only .* high target"):
                finalize(root)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_inventory(root)
            finalize(root)
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                finalize(root)


class HierarchyRegistryValidationTest(unittest.TestCase):
    def _prepared_registry(self, root: Path):
        write_inventory(root)
        finalize(root)
        return root / "hierarchy" / "training_registry.csv"

    def test_rejects_missing_and_duplicate_cells(self):
        for mutation, message in (("missing", "exactly 32"), ("duplicate", "duplicate cell")):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                registry_path = self._prepared_registry(Path(directory))
                rows = read_csv(registry_path)
                if mutation == "missing":
                    rows.pop()
                else:
                    rows[-1].update(
                        {
                            "replicate": rows[0]["replicate"],
                            "sequence_support": rows[0]["sequence_support"],
                            "window_policy": rows[0]["window_policy"],
                        }
                    )
                write_csv(registry_path, HIERARCHY_REGISTRY_COLUMNS, rows)
                with self.assertRaisesRegex(ValueError, message):
                    read_hierarchy_registry(registry_path)

    def test_rejects_non_nested_and_duplicated_manifest_rows(self):
        for mutation, message in (
            ("non_nested", "whole source groups"),
            ("duplicate_sequence", "duplicate or empty sequence_id"),
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                registry_path = self._prepared_registry(root)
                low_path = root / "hierarchy" / "manifests" / "replicate-0-low.csv"
                rows = read_csv(low_path)
                if mutation == "non_nested":
                    rows[0]["sequence_id"] = "outside-high"
                    rows[0]["source_group"] = "outside-high"
                else:
                    rows[1]["sequence_id"] = rows[0]["sequence_id"]
                write_csv(low_path, GAITLU_MANIFEST_COLUMNS, rows)
                with self.assertRaisesRegex(ValueError, message):
                    read_hierarchy_registry(registry_path)

    def test_rejects_unsafe_model_labels_and_negative_seeds(self):
        for column, value, message in (
            ("model_label", "../escape", "safe path component"),
            ("optimization_seed", "-1", "must be nonnegative"),
        ):
            with self.subTest(column=column), tempfile.TemporaryDirectory() as directory:
                registry_path = self._prepared_registry(Path(directory))
                rows = read_csv(registry_path)
                rows[0][column] = value
                write_csv(registry_path, HIERARCHY_REGISTRY_COLUMNS, rows)
                with self.assertRaisesRegex(ValueError, message):
                    read_hierarchy_registry(registry_path)

    def test_rejects_partial_groups_and_noncanonical_manifest_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry_path = self._prepared_registry(root)
            low_path = root / "hierarchy" / "manifests" / "replicate-0-low.csv"
            low_rows = read_csv(low_path)
            removed_group = low_rows[0]["source_group"]
            low_rows = [
                row
                for index, row in enumerate(low_rows)
                if not (row["source_group"] == removed_group and index == 0)
            ]
            write_csv(low_path, GAITLU_MANIFEST_COLUMNS, low_rows)
            registry_rows = read_csv(registry_path)
            for row in registry_rows:
                if row["replicate"] == "0" and row["sequence_support"] == "low":
                    row["unique_sequences"] = str(len(low_rows))
            write_csv(registry_path, HIERARCHY_REGISTRY_COLUMNS, registry_rows)
            with self.assertRaisesRegex(ValueError, "whole source groups"):
                read_hierarchy_registry(registry_path)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry_path = self._prepared_registry(root)
            low_path = root / "hierarchy" / "manifests" / "replicate-0-low.csv"
            high_path = root / "hierarchy" / "manifests" / "replicate-0-high.csv"
            low_groups = {row["source_group"] for row in read_csv(low_path)}
            high_rows = read_csv(high_path)
            high_only_group = next(
                row["source_group"]
                for row in high_rows
                if row["source_group"] not in low_groups
            )
            removed = False
            partial_high = []
            for row in high_rows:
                if row["source_group"] == high_only_group and not removed:
                    removed = True
                    continue
                partial_high.append(row)
            write_csv(high_path, GAITLU_MANIFEST_COLUMNS, partial_high)
            registry_rows = read_csv(registry_path)
            for row in registry_rows:
                if row["replicate"] == "0" and row["sequence_support"] == "high":
                    row["unique_sequences"] = str(len(partial_high))
            write_csv(registry_path, HIERARCHY_REGISTRY_COLUMNS, registry_rows)
            with self.assertRaisesRegex(ValueError, "whole source groups"):
                read_hierarchy_registry(registry_path)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry_path = self._prepared_registry(root)
            registry_rows = read_csv(registry_path)
            for row in registry_rows:
                if row["replicate"] == "1":
                    support = row["sequence_support"]
                    row["train_manifest"] = (
                        f"manifests/./replicate-0-{support}.csv"
                    )
            write_csv(registry_path, HIERARCHY_REGISTRY_COLUMNS, registry_rows)
            with self.assertRaisesRegex(ValueError, "canonical safe relative path"):
                read_hierarchy_registry(registry_path)


if __name__ == "__main__":
    unittest.main()
