import csv
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from cody_jepa.evaluation.gfc.roles import (
    DEVELOPMENT_ROLE,
    EXPECTED_ASSIGNED_COUNTS,
    LOCKED_OUTCOME_ROLE,
    build_role_map,
    load_role_map,
    validate_role_map,
)


def role_rows():
    return [
        *(
            {"subject_id": f"D{index:03d}", "role": DEVELOPMENT_ROLE}
            for index in range(EXPECTED_ASSIGNED_COUNTS[DEVELOPMENT_ROLE])
        ),
        *(
            {"subject_id": f"O{index:03d}", "role": LOCKED_OUTCOME_ROLE}
            for index in range(EXPECTED_ASSIGNED_COUNTS[LOCKED_OUTCOME_ROLE])
        ),
    ]


class RoleMapTest(unittest.TestCase):
    def test_builder_maps_historical_splits_and_sorts_deterministically(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            rows = [
                *(
                    {"subject_id": f"O{index:03d}", "split": "train", "recording": "a"}
                    for index in reversed(range(318))
                ),
                *(
                    {"subject_id": f"D{index:03d}", "split": "val", "recording": "a"}
                    for index in reversed(range(80))
                ),
            ]
            with manifest.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            output = root / "private" / "roles.csv"
            result = build_role_map(manifest, output)
            self.assertEqual(list(result.columns), ["subject_id", "role"])
            self.assertEqual(result.iloc[0].to_dict(), {"subject_id": "D000", "role": "development"})
            self.assertEqual(result.iloc[-1].to_dict(), {"subject_id": "O317", "role": "locked_outcome"})
            self.assertEqual(output.read_text().splitlines()[0], "subject_id,role")
            pd.testing.assert_frame_equal(result, load_role_map(output))
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                build_role_map(manifest, output)

    def test_role_map_rejects_schema_counts_duplicates_and_bad_text(self):
        valid = pd.DataFrame(role_rows())
        mutations = []
        wrong_schema = valid.rename(columns={"role": "cohort"})
        mutations.append((wrong_schema, "columns"))
        missing = valid.iloc[:-1].copy()
        mutations.append((missing, "assigned counts"))
        duplicate = valid.copy()
        duplicate.loc[1, "subject_id"] = duplicate.loc[0, "subject_id"]
        mutations.append((duplicate, "duplicate"))
        case_collision = valid.copy()
        case_collision.loc[1, "subject_id"] = case_collision.loc[0, "subject_id"].lower()
        mutations.append((case_collision, "case-colliding"))
        whitespace = valid.copy()
        whitespace.loc[0, "subject_id"] += " "
        mutations.append((whitespace, "outer whitespace"))
        unknown = valid.copy()
        unknown.loc[0, "role"] = "test"
        mutations.append((unknown, "unknown roles"))
        for table, message in mutations:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                validate_role_map(table)

    def test_exact_feature_coverage_rejects_missing_extra_and_case_collisions(self):
        valid = pd.DataFrame(role_rows())
        subjects = list(valid["subject_id"])
        with self.assertRaisesRegex(ValueError, "missing=.*extra="):
            validate_role_map(valid, expected_subject_ids=[*subjects[:-1], "X999"])
        case_changed = list(subjects)
        case_changed[0] = case_changed[0].lower()
        with self.assertRaisesRegex(ValueError, "missing=.*extra="):
            validate_role_map(valid, expected_subject_ids=case_changed)

    def test_builder_rejects_unknown_split_and_cross_split_participant(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for rows, message in (
                ([{"subject_id": "P1", "split": "test"}], "unsupported historical split"),
                (
                    [
                        {"subject_id": "P1", "split": "train"},
                        {"subject_id": "P1", "split": "val"},
                    ],
                    "spans historical splits",
                ),
            ):
                manifest = root / "manifest.csv"
                pd.DataFrame(rows).to_csv(manifest, index=False)
                with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                    build_role_map(manifest, root / "roles.csv")


if __name__ == "__main__":
    unittest.main()
