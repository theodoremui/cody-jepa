import math
import unittest

import numpy as np

from cody_jepa.gfc import (
    CANONICAL_CELLS,
    EXPECTED_GALLERY_SIZE,
    EXPECTED_QUERIES,
    Cell,
    Recording,
    aggregate_windows,
    build_queries,
    cosine_distance,
    evaluate_cohort,
    evaluate_participant,
    rank_target,
)


def _one_hot(index, width):
    result = np.zeros(width, dtype=np.float64)
    result[index] = 1.0
    return result


def participant_recordings(subject_id="P001", representation="exact"):
    recordings = []
    for cell in CANONICAL_CELLS:
        if representation == "exact":
            condition = _one_hot(
                2 * ("WoJ", "WJ").index(cell.clothing)
                + ("R2L", "L2R").index(cell.direction),
                4,
            )
            gait = _one_hot(("UGS", "FGS").index(cell.speed), 2)
        elif representation == "constant":
            condition = np.zeros(4)
            gait = np.zeros(2)
        else:
            raise ValueError(representation)
        recordings.append(
            Recording.from_windows(
                subject_id=subject_id,
                recording_id=f"{subject_id}:{cell.key}",
                cell=cell,
                condition_windows=[condition] * 3,
                gait_windows=[gait] * 3,
                window_ids=[f"{subject_id}:{cell.key}:w{index}" for index in range(3)],
            )
        )
    return recordings


class WindowAndFactorTest(unittest.TestCase):
    def test_window_aggregation_precedes_recording_construction(self):
        windows = np.asarray([[1.0, 2.0], [2.0, 4.0], [6.0, 9.0]], dtype=np.float32)
        result = aggregate_windows(windows)
        self.assertEqual(result.dtype, np.float64)
        np.testing.assert_array_equal(result, [3.0, 5.0])
        with self.assertRaisesRegex(ValueError, "exactly 3"):
            aggregate_windows(windows[:2])
        with self.assertRaisesRegex(FloatingPointError, "non-finite"):
            aggregate_windows([[0.0], [math.nan], [1.0]])

    def test_factor_values_are_explicit(self):
        self.assertEqual(len(CANONICAL_CELLS), 8)
        self.assertEqual(CANONICAL_CELLS[0], Cell("UGS", "WoJ", "R2L"))
        with self.assertRaisesRegex(ValueError, "speed"):
            Cell("unknown", "WoJ", "R2L")


class QueryConstructionTest(unittest.TestCase):
    def test_eight_cells_produce_24_donor_excluded_queries(self):
        recordings = participant_recordings()
        cells = {item.recording_id: item.cell for item in recordings}
        queries = build_queries(list(reversed(recordings)))
        self.assertEqual(len(queries), EXPECTED_QUERIES)
        for query in queries:
            self.assertEqual(len(query.gallery_ids), EXPECTED_GALLERY_SIZE)
            self.assertIn(query.target_id, query.gallery_ids)
            self.assertNotIn(query.condition_donor_id, query.gallery_ids)
            self.assertNotIn(query.gait_donor_id, query.gallery_ids)
            target = cells[query.target_id]
            condition = cells[query.condition_donor_id]
            gait = cells[query.gait_donor_id]
            self.assertNotEqual(condition.speed, target.speed)
            self.assertEqual((condition.clothing, condition.direction), (target.clothing, target.direction))
            self.assertEqual(gait.speed, target.speed)
            self.assertNotEqual((gait.clothing, gait.direction), (target.clothing, target.direction))

    def test_missing_cells_exclude_and_duplicate_cells_fail(self):
        incomplete = participant_recordings()[:-1]
        cohort = evaluate_cohort(incomplete)
        self.assertEqual(cohort.participants, ())
        self.assertEqual(len(cohort.exclusions), 1)
        self.assertEqual(cohort.exclusions[0].missing_cells, (CANONICAL_CELLS[-1],))

        duplicate = participant_recordings()
        duplicate[-1] = Recording(
            subject_id="P001",
            recording_id="replacement",
            cell=duplicate[0].cell,
            condition_block=np.ones(4),
            gait_block=np.ones(2),
            window_ids=("replacement:w0", "replacement:w1", "replacement:w2"),
        )
        with self.assertRaisesRegex(ValueError, "duplicate factorial cell"):
            evaluate_cohort(duplicate)


class ScoringTest(unittest.TestCase):
    def test_exact_factor_features_retrieve_every_target(self):
        result = evaluate_participant(participant_recordings(), representation="learned")
        self.assertEqual(result.top1, 1.0)
        self.assertEqual(result.mrr, 1.0)
        self.assertEqual(len(result.queries), 24)

    def test_constant_blocks_produce_declared_tied_null(self):
        result = evaluate_participant(participant_recordings(representation="constant"))
        self.assertAlmostEqual(result.top1, 1.0 / 6.0)
        self.assertAlmostEqual(result.mrr, 2.0 / 7.0)
        self.assertEqual(result.donor_attraction, 0.5)
        self.assertTrue(all(item.rank.tie_size == 6 for item in result.queries))

    def test_fractional_ties_and_mrr_use_average_occupied_rank(self):
        partial = rank_target({"a": 0.0, "target": 1.0, "b": 1.0, "c": 1.0, "d": 2.0}, "target")
        self.assertEqual(partial.strictly_closer_count, 1)
        self.assertEqual(partial.tie_size, 3)
        self.assertEqual(partial.average_rank, 3.0)
        self.assertEqual(partial.top1, 0.0)
        self.assertEqual(partial.reciprocal_rank, 1.0 / 3.0)
        top_tie = rank_target({"target": 0.0, "a": 0.0, "b": 0.0, "c": 1.0}, "target")
        self.assertEqual(top_tie.top1, 1.0 / 3.0)
        self.assertEqual(top_tie.average_rank, 2.0)

    def test_cosine_zero_and_extreme_finite_values_are_stable(self):
        self.assertEqual(cosine_distance([0.0, 0.0], [1.0, 0.0]), 1.0)
        self.assertAlmostEqual(cosine_distance([1e300, 0.0], [1e300, 0.0]), 0.0)

    def test_participants_receive_equal_cohort_weight(self):
        recordings = participant_recordings("P1", "exact") + participant_recordings("P2", "constant")
        result = evaluate_cohort(recordings)
        self.assertEqual(len(result.participants), 2)
        self.assertAlmostEqual(result.top1, (1.0 + 1.0 / 6.0) / 2.0)


if __name__ == "__main__":
    unittest.main()
