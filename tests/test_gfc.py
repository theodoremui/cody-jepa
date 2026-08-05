from dataclasses import replace
import itertools
import math
import unittest

import numpy as np

from cody_jepa.evaluation.gfc.core import _compose_query_blocks, _donor_attraction
from cody_jepa.evaluation.gfc.oracle import compile_healthgait_gfc_v2_protocol
from cody_jepa.gfc import (
    CANONICAL_CELLS,
    EXPECTED_GALLERY_SIZE,
    EXPECTED_QUERIES,
    Cell,
    FactorBlocks,
    Recording,
    aggregate_windows,
    build_queries,
    cosine_distance,
    evaluate_cohort,
    evaluate_participant,
    rank_target,
)


FACTOR_VALUES = {
    "speed": ("UGS", "FGS"),
    "clothing": ("WoJ", "WJ"),
    "direction": ("R2L", "L2R"),
}


def _one_hot(index, width=2):
    result = np.zeros(width, dtype=np.float64)
    result[index] = 1.0
    return result


def participant_recordings(subject_id="P001", recovered_factors=("speed", "clothing", "direction")):
    recovered = set(recovered_factors)
    recordings = []
    for cell in CANONICAL_CELLS:
        blocks = {
            factor: (
                _one_hot(FACTOR_VALUES[factor].index(getattr(cell, factor)))
                if factor in recovered
                else np.zeros(2)
            )
            for factor in FACTOR_VALUES
        }
        recordings.append(
            Recording.from_windows(
                subject_id=subject_id,
                recording_id=f"{subject_id}:{cell.key}",
                source_video_id=f"{subject_id}:{cell.speed}:{cell.clothing}",
                cell=cell,
                speed_windows=[blocks["speed"]] * 3,
                clothing_windows=[blocks["clothing"]] * 3,
                direction_windows=[blocks["direction"]] * 3,
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

    def test_factor_blocks_and_source_lineage_are_immutable_and_required(self):
        recording = participant_recordings()[0]
        self.assertEqual(recording.source_video_id, "P001:UGS:WoJ")
        self.assertFalse(recording.factor_blocks.speed.flags.writeable)
        with self.assertRaises(ValueError):
            recording.factor_blocks.speed[0] = 7.0
        with self.assertRaisesRegex(ValueError, "source_video_id"):
            replace(recording, source_video_id="")

    def test_factor_values_are_compiler_defined(self):
        protocol = compile_healthgait_gfc_v2_protocol()
        self.assertEqual(tuple((cell.speed, cell.clothing, cell.direction) for cell in CANONICAL_CELLS), protocol.design.cells)
        with self.assertRaisesRegex(ValueError, "speed"):
            Cell("unknown", "WoJ", "R2L")


class QueryConstructionTest(unittest.TestCase):
    def test_production_queries_exactly_match_every_compiled_oracle_query(self):
        recordings = participant_recordings()
        by_cell = {item.cell: item for item in recordings}
        queries = build_queries(list(reversed(recordings)))
        protocol = compile_healthgait_gfc_v2_protocol()
        self.assertEqual(len(queries), EXPECTED_QUERIES)
        self.assertEqual(EXPECTED_QUERIES, 16)
        self.assertEqual(EXPECTED_GALLERY_SIZE, 8)
        for query, compiled in zip(queries, protocol.queries):
            self.assertEqual(query.query_index, compiled.query_index)
            self.assertEqual(query.protocol, protocol.name)
            self.assertEqual(query.focal_factor, compiled.focal_factor)
            self.assertEqual(query.target_cell, Cell(*compiled.target))
            self.assertEqual((query.donor_u_cell, query.donor_v_cell), tuple(Cell(*cell) for cell in compiled.donors))
            self.assertEqual(query.factor_sources, tuple("donor_u" if source == 0 else "donor_v" for source in compiled.factor_sources))
            self.assertEqual(query.gallery_cells, tuple(Cell(*cell) for cell in compiled.gallery))
            self.assertEqual(query.gallery_ids, tuple(by_cell[Cell(*cell)].recording_id for cell in compiled.gallery))
            self.assertIn(query.donor_u_id, query.gallery_ids)
            self.assertIn(query.donor_v_id, query.gallery_ids)
            self.assertTrue(query.source_independence_verified)

    def test_donor_source_collisions_fail_separately(self):
        protocol = compile_healthgait_gfc_v2_protocol()
        target_cell = Cell(*protocol.queries[0].target)
        donor_u_cell, donor_v_cell = (Cell(*cell) for cell in protocol.queries[0].donors)

        for donor_cell, expected_role in ((donor_u_cell, "donor_u"), (donor_v_cell, "donor_v")):
            recordings = participant_recordings()
            target_source = next(item.source_video_id for item in recordings if item.cell == target_cell)
            collision = next(index for index, item in enumerate(recordings) if item.cell == donor_cell)
            recordings[collision] = replace(recordings[collision], source_video_id=target_source)
            with self.subTest(role=expected_role), self.assertRaisesRegex(ValueError, expected_role):
                build_queries(recordings)

    def test_valid_paired_direction_lineage_is_accepted(self):
        recordings = participant_recordings()
        for speed, clothing in itertools.product(FACTOR_VALUES["speed"], FACTOR_VALUES["clothing"]):
            pair = [item for item in recordings if item.cell.speed == speed and item.cell.clothing == clothing]
            self.assertEqual(len({item.source_video_id for item in pair}), 1)
        self.assertEqual(len(build_queries(recordings)), 16)

    def test_query_composition_uses_only_declared_donors(self):
        recordings = participant_recordings()
        by_id = {item.recording_id: item for item in recordings}
        query = build_queries(recordings)[0]
        donor_u = by_id[query.donor_u_id]
        donor_v = by_id[query.donor_v_id]
        target = by_id[query.target_id]
        composed = _compose_query_blocks(query, donor_u, donor_v)
        for factor, role in zip(FACTOR_VALUES, query.factor_sources):
            expected = (donor_u if role == "donor_u" else donor_v).factor_blocks.for_factor(factor)
            np.testing.assert_array_equal(composed.for_factor(factor), expected)
        changed_target = replace(target, factor_blocks=FactorBlocks(np.full(2, 17.0), np.full(2, 18.0), np.full(2, 19.0)))
        np.testing.assert_array_equal(
            _compose_query_blocks(query, donor_u, donor_v).speed,
            composed.speed,
        )
        self.assertNotEqual(changed_target.factor_blocks.speed[0], composed.speed[0])

    def test_missing_and_duplicate_cells_fail_as_declared(self):
        incomplete = participant_recordings()[:-1]
        cohort = evaluate_cohort(incomplete)
        self.assertEqual(cohort.participants, ())
        self.assertEqual(cohort.exclusions[0].missing_cells, (CANONICAL_CELLS[-1],))

        duplicate = participant_recordings()
        duplicate[-1] = replace(duplicate[-1], recording_id="replacement", cell=duplicate[0].cell)
        with self.assertRaisesRegex(ValueError, "duplicate factorial cell"):
            evaluate_cohort(duplicate)


class ScoringTest(unittest.TestCase):
    def test_every_recovered_factor_subset_matches_v2_oracle_spectrum(self):
        expected = {
            0: (1.0 / 8.0, 2.0 / 9.0),
            1: (1.0 / 4.0, 2.0 / 5.0),
            2: (1.0 / 2.0, 2.0 / 3.0),
            3: (1.0, 1.0),
        }
        factors = tuple(FACTOR_VALUES)
        for size in range(4):
            for recovered in itertools.combinations(factors, size):
                with self.subTest(recovered=recovered):
                    result = evaluate_participant(participant_recordings(recovered_factors=recovered))
                    self.assertAlmostEqual(result.top1, expected[size][0])
                    self.assertAlmostEqual(result.mrr, expected[size][1])
                    self.assertEqual(len(result.queries), 16)
                    self.assertTrue(all(len(item.distances) == 8 for item in result.queries))

    def test_constant_blocks_produce_declared_tied_null_and_attractions(self):
        result = evaluate_participant(participant_recordings(recovered_factors=()))
        self.assertAlmostEqual(result.top1, 1.0 / 8.0)
        self.assertAlmostEqual(result.mrr, 2.0 / 9.0)
        self.assertEqual(result.donor_u_attraction, 0.5)
        self.assertEqual(result.donor_v_attraction, 0.5)
        self.assertTrue(all(item.rank.tie_size == 8 for item in result.queries))

    def test_donor_attraction_win_loss_and_tie_semantics(self):
        self.assertEqual(_donor_attraction(1.0, 0.0, tolerance=1e-12), 1.0)
        self.assertEqual(_donor_attraction(0.0, 1.0, tolerance=1e-12), 0.0)
        self.assertEqual(_donor_attraction(1.0, 1.0, tolerance=1e-12), 0.5)

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
        recordings = participant_recordings("P1") + participant_recordings("P2", ())
        result = evaluate_cohort(recordings)
        self.assertEqual(len(result.participants), 2)
        self.assertAlmostEqual(result.top1, (1.0 + 1.0 / 8.0) / 2.0)


if __name__ == "__main__":
    unittest.main()
