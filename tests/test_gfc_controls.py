from dataclasses import replace
import itertools
import unittest

import numpy as np

from cody_jepa.evaluation.gfc.controls import (
    _rank_log_masses,
    evaluate_independent_factor_controls,
    fit_shared_temperature,
)
from cody_jepa.evaluation.gfc.core import CANONICAL_CELLS, Cell, FactorBlocks, Recording


FACTOR_VALUES = {
    "speed": ("UGS", "FGS"),
    "clothing": ("WoJ", "WJ"),
    "direction": ("R2L", "L2R"),
}


def control_recordings(recovered=("speed", "clothing", "direction")):
    recovered = set(recovered)
    rows = []
    for cell in CANONICAL_CELLS:
        blocks = {}
        for factor_name, labels in FACTOR_VALUES.items():
            values = np.zeros(2, dtype=np.float64)
            if factor_name in recovered:
                values[labels.index(getattr(cell, factor_name))] = 2.0
            blocks[factor_name] = values
        rows.append(
            Recording(
                subject_id="P1",
                recording_id=f"P1:{cell.key}",
                source_video_id=f"P1:{cell.speed}:{cell.clothing}",
                cell=cell,
                factor_blocks=FactorBlocks(**blocks),
                window_ids=tuple(f"P1:{cell.key}:w{i}" for i in range(3)),
            )
        )
    return rows


class IndependentFactorControlTest(unittest.TestCase):
    def test_hard_control_has_expected_factor_recovery_spectrum(self):
        expected = {0: (1 / 8, 2 / 9), 1: (1 / 4, 2 / 5), 2: (1 / 2, 2 / 3), 3: (1.0, 1.0)}
        factors = tuple(FACTOR_VALUES)
        for count in range(4):
            for recovered in itertools.combinations(factors, count):
                with self.subTest(recovered=recovered):
                    result = evaluate_independent_factor_controls(
                        control_recordings(recovered), temperature=0.7
                    )
                    self.assertAlmostEqual(result.hard_top1, expected[count][0])
                    self.assertAlmostEqual(result.hard_mrr, expected[count][1])
                    self.assertAlmostEqual(result.soft_top1, result.hard_top1)
                    self.assertTrue(result.top1_agreement)

    def test_soft_ranking_is_temperature_invariant_but_probability_is_not(self):
        rows = control_recordings(("speed", "clothing"))
        cold = evaluate_independent_factor_controls(rows, temperature=0.1)
        hot = evaluate_independent_factor_controls(rows, temperature=10.0)
        self.assertEqual(cold.soft_top1, hot.soft_top1)
        self.assertEqual(cold.soft_mrr, hot.soft_mrr)
        self.assertNotEqual(cold.soft_target_probability, hot.soft_target_probability)
        self.assertNotEqual(cold.soft_target_nll, hot.soft_target_nll)

    def test_soft_ranking_tolerance_scales_with_temperature(self):
        rows = []
        for row in control_recordings():
            rows.append(
                replace(
                    row,
                    factor_blocks=FactorBlocks(
                        speed=row.factor_blocks.speed * 1e-12,
                        clothing=row.factor_blocks.clothing * 1e-12,
                        direction=row.factor_blocks.direction * 1e-12,
                    ),
                )
            )
        baseline = evaluate_independent_factor_controls(
            rows, temperature=1.0, tie_tolerance=1e-12
        )
        hot = evaluate_independent_factor_controls(
            rows, temperature=1000.0, tie_tolerance=1e-12
        )
        self.assertEqual(hot.soft_top1, baseline.soft_top1)
        self.assertEqual(hot.soft_mrr, baseline.soft_mrr)

    def test_soft_ranking_does_not_create_underflow_ties(self):
        rank = _rank_log_masses(
            {"best": 0.0, "second": -1000.0, "target": -2000.0, "tie": -2000.0},
            "target",
            tolerance=1e-12,
        )
        self.assertEqual(rank.strictly_closer_count, 2)
        self.assertEqual(rank.tie_size, 2)
        self.assertEqual(rank.average_rank, 3.5)

    def test_control_uses_donors_and_not_query_target_blocks(self):
        from cody_jepa.evaluation.gfc.controls import _composed_scores
        from cody_jepa.evaluation.gfc.core import build_queries

        rows = control_recordings()
        query = build_queries(rows)[0]
        by_id = {item.recording_id: item for item in rows}
        before = _composed_scores(query, by_id)
        target = by_id[query.target_id]
        by_id[query.target_id] = replace(
            target,
            factor_blocks=FactorBlocks(
                speed=np.asarray([100.0, -100.0]),
                clothing=np.asarray([100.0, -100.0]),
                direction=np.asarray([100.0, -100.0]),
            ),
        )
        after = _composed_scores(query, by_id)
        for factor_name in FACTOR_VALUES:
            np.testing.assert_array_equal(before[factor_name], after[factor_name])

    def test_shared_temperature_is_deterministic_and_rejects_boundary_optima(self):
        cells = list(CANONICAL_CELLS)
        scores = {}
        for factor_name, labels in FACTOR_VALUES.items():
            values = []
            for index, cell in enumerate(cells):
                target = labels.index(getattr(cell, factor_name))
                margin = -0.75 if index == 0 else 1.0
                row = np.zeros(2)
                row[target] = margin
                values.append(row)
            scores[factor_name] = np.asarray(values)
        first = fit_shared_temperature(scores, cells)
        second = fit_shared_temperature(scores, cells)
        self.assertEqual(first, second)
        self.assertGreater(first.temperature, 0.001)
        self.assertLess(first.temperature, 1000.0)
        self.assertLess(first.nll_after, first.nll_before)

        perfect = {
            factor_name: np.asarray(
                [
                    [2.0, 0.0] if getattr(cell, factor_name) == labels[0] else [0.0, 2.0]
                    for cell in cells
                ]
            )
            for factor_name, labels in FACTOR_VALUES.items()
        }
        with self.assertRaisesRegex(RuntimeError, "boundary"):
            fit_shared_temperature(perfect, cells)

    def test_temperature_fit_rejects_nonfinite_and_wrong_factor_shape(self):
        cells = [Cell("UGS", "WoJ", "R2L")]
        valid = {name: np.zeros((1, 2)) for name in FACTOR_VALUES}
        bad = dict(valid)
        bad["speed"] = np.asarray([[np.nan, 0.0]])
        with self.assertRaisesRegex(FloatingPointError, "non-finite"):
            fit_shared_temperature(bad, cells)
        with self.assertRaisesRegex(ValueError, "shape"):
            fit_shared_temperature({**valid, "speed": np.zeros((1, 3))}, cells)


if __name__ == "__main__":
    unittest.main()
