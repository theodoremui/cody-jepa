import unittest

import numpy as np

from cody_jepa.gfc import CANONICAL_CELLS
from cody_jepa.gfc_normalization import (
    SCALE_FLOOR,
    entropy_effective_rank,
    fit_condition_adapter,
    fit_gait_adapter,
    fit_pca_normalizer,
    fit_raw_normalizer,
    retained_dimension,
    rowwise_l2_normalize,
)


def factor_training_rows(subjects=("T1", "T2")):
    rows = []
    subject_ids = []
    cells = []
    for subject_index, subject_id in enumerate(subjects):
        for cell in CANONICAL_CELLS:
            speed = float(("UGS", "FGS").index(cell.speed))
            clothing = float(("WoJ", "WJ").index(cell.clothing))
            direction = float(("R2L", "L2R").index(cell.direction))
            rows.append([speed, clothing, direction, float(subject_index), 7.0])
            subject_ids.append(subject_id)
            cells.append(cell)
    return np.asarray(rows, dtype=np.float64), subject_ids, cells


class EffectiveRankTest(unittest.TestCase):
    def test_entropy_rank_and_half_up_dimension(self):
        self.assertEqual(entropy_effective_rank([0.0, 0.0]), 1.0)
        self.assertAlmostEqual(entropy_effective_rank([1.0, 1.0]), 2.0)
        self.assertEqual(retained_dimension(2.499999, 10), 2)
        self.assertEqual(retained_dimension(2.5, 10), 3)
        self.assertEqual(retained_dimension(100.0, 4), 4)

    def test_l2_normalization_preserves_zero_rows(self):
        result = rowwise_l2_normalize([[3.0, 4.0], [0.0, 0.0]])
        np.testing.assert_allclose(result[0], [0.6, 0.8])
        np.testing.assert_array_equal(result[1], [0.0, 0.0])


class NormalizerTest(unittest.TestCase):
    def test_raw_normalizer_uses_only_rows_passed_to_fit(self):
        training = np.asarray([[0.0, 0.0, 7.0], [2.0, 1.0, 7.0], [4.0, 0.0, 7.0]])
        held_out = np.asarray([[10_000.0, -10_000.0, 3.0]])
        fit = fit_raw_normalizer(training, dimension_policy="effective_rank")
        self.assertEqual(fit.fit_row_count, 3)
        contaminated = fit_raw_normalizer(
            np.concatenate([training, held_out], axis=0), dimension_policy="effective_rank"
        )
        self.assertFalse(np.array_equal(fit.output_mean, contaminated.output_mean))

    def test_retain_all_is_safe_for_small_semantic_adapter_blocks(self):
        rows = []
        for cell in CANONICAL_CELLS:
            rows.append(
                [
                    float(cell.clothing == "WoJ"),
                    float(cell.clothing == "WJ"),
                    float(cell.direction == "R2L"),
                    float(cell.direction == "L2R"),
                ]
            )
        rows = np.asarray(rows * 2, dtype=np.float64)
        rank_fit = fit_raw_normalizer(rows, dimension_policy="effective_rank")
        all_fit = fit_raw_normalizer(rows, dimension_policy="retain_all")
        self.assertEqual(rank_fit.retained_dimension, 2)
        self.assertEqual(rank_fit.selected_indices, (0, 1))
        self.assertEqual(all_fit.retained_dimension, 4)
        self.assertEqual(all_fit.selected_indices, (0, 1, 2, 3))
        # Retaining all coordinates preserves both clothing and direction.
        self.assertEqual(len(np.unique(all_fit.transform(rows[:8]), axis=0)), 4)
        self.assertEqual(len(np.unique(rank_fit.transform(rows[:8]), axis=0)), 2)

    def test_pca_is_deterministic_under_row_permutation_and_tied_spectrum(self):
        rows = np.asarray(
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]
        )
        first = fit_pca_normalizer(rows, dimension_policy="effective_rank")
        second = fit_pca_normalizer(rows[::-1], dimension_policy="effective_rank")
        self.assertEqual(first.retained_dimension, 2)
        np.testing.assert_array_equal(first.components, second.components)
        np.testing.assert_array_equal(first.transform(rows), second.transform(rows))

    def test_nonfinite_and_dimension_mismatch_fail(self):
        with self.assertRaisesRegex(FloatingPointError, "non-finite"):
            fit_raw_normalizer([[0.0], [np.nan]])
        fit = fit_pca_normalizer([[0.0, 1.0], [1.0, 0.0]])
        with self.assertRaisesRegex(ValueError, "width"):
            fit.transform([[0.0, 1.0, 2.0]])


class RidgeAdapterTest(unittest.TestCase):
    def test_shapes_labels_and_closed_form_outputs(self):
        rows, subject_ids, cells = factor_training_rows()
        condition = fit_condition_adapter(rows, subject_ids, cells, alpha=1.0)
        gait = fit_gait_adapter(rows, subject_ids, cells, alpha=1.0)
        self.assertEqual(condition.coefficients.shape, (rows.shape[1], 4))
        self.assertEqual(condition.target_labels, ("WoJ", "WJ", "R2L", "L2R"))
        self.assertEqual(gait.coefficients.shape, (rows.shape[1], 2))
        self.assertEqual(gait.target_labels, ("UGS", "FGS"))
        self.assertEqual(condition.feature_scale[-1], SCALE_FLOOR)
        condition_values = condition.transform(rows)
        gait_values = gait.transform(rows)
        self.assertEqual(condition_values.shape, (len(rows), 4))
        self.assertEqual(gait_values.shape, (len(rows), 2))
        self.assertGreater(condition_values[0, 0], condition_values[0, 1])
        self.assertGreater(condition_values[0, 2], condition_values[0, 3])
        self.assertGreater(gait_values[0, 0], gait_values[0, 1])

    def test_adapter_requires_complete_unique_training_grids(self):
        rows, subject_ids, cells = factor_training_rows()
        with self.assertRaisesRegex(ValueError, "all eight"):
            fit_condition_adapter(rows[:-1], subject_ids[:-1], cells[:-1])
        duplicate_cells = list(cells)
        duplicate_cells[1] = duplicate_cells[0]
        with self.assertRaisesRegex(ValueError, "duplicate subject-cell"):
            fit_gait_adapter(rows, subject_ids, duplicate_cells)

    def test_adapter_fit_is_independent_of_input_row_order(self):
        rows, subject_ids, cells = factor_training_rows()
        first = fit_condition_adapter(rows, subject_ids, cells)
        order = np.arange(len(rows))[::-1]
        second = fit_condition_adapter(
            rows[order],
            [subject_ids[index] for index in order],
            [cells[index] for index in order],
        )
        np.testing.assert_array_equal(first.coefficients, second.coefficients)
        np.testing.assert_array_equal(first.intercept, second.intercept)


if __name__ == "__main__":
    unittest.main()
