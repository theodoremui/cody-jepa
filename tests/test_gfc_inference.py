import unittest
from dataclasses import replace

import numpy as np

from cody_jepa.gfc import CANONICAL_CELLS, Recording, evaluate_participant
from cody_jepa.gfc_inference import (
    bootstrap_gfc_gain,
    bootstrap_lsg,
    paired_cohort_gfc_gain,
    paired_cohort_lsg,
    paired_participant_gfc_gain,
    paired_participant_lsg,
    plan_prospective_power,
)
from cody_jepa.gfc_normalization import (
    fit_condition_adapter,
    fit_gait_adapter,
    fit_raw_normalizer,
)


def _one_hot(index, width):
    value = np.zeros(width, dtype=np.float64)
    value[index] = 1.0
    return value


def recordings(subject_id, representation):
    result = []
    for cell in CANONICAL_CELLS:
        if representation == "learned":
            condition = _one_hot(
                2 * ("WoJ", "WJ").index(cell.clothing)
                + ("R2L", "L2R").index(cell.direction),
                4,
            )
            gait = _one_hot(("UGS", "FGS").index(cell.speed), 2)
        elif representation == "shortcut":
            condition = np.zeros(4)
            gait = np.zeros(2)
        else:
            raise ValueError(representation)
        result.append(
            Recording.from_windows(
                subject_id=subject_id,
                recording_id=f"{subject_id}:{cell.key}",
                cell=cell,
                condition_windows=[condition] * 3,
                gait_windows=[gait] * 3,
                window_ids=[f"{subject_id}:{cell.key}:w{index}" for index in range(3)],
            )
        )
    return result


def scores(subject_id, representation, *, split="development", seed=7):
    return evaluate_participant(
        recordings(subject_id, representation),
        split=split,
        seed=seed,
        representation=representation,
    )


class PairingTest(unittest.TestCase):
    def test_initial_release_names_remain_compatible_aliases(self):
        self.assertIs(paired_participant_lsg, paired_participant_gfc_gain)
        self.assertIs(paired_cohort_lsg, paired_cohort_gfc_gain)
        self.assertIs(bootstrap_lsg, bootstrap_gfc_gain)

    def test_identical_scientific_queries_are_required(self):
        learned = scores("P1", "learned")
        shortcut = scores("P1", "shortcut")
        contrast = paired_participant_gfc_gain(learned, shortcut)
        self.assertEqual(contrast.learned_top1, 1.0)
        self.assertAlmostEqual(contrast.shortcut_top1, 1.0 / 6.0)
        self.assertAlmostEqual(contrast.difference, 5.0 / 6.0)

        changed = replace(shortcut.queries[0], split="confirmation")
        mismatched = replace(shortcut, queries=(changed, *shortcut.queries[1:]))
        with self.assertRaisesRegex(ValueError, "scientific query keys differ"):
            paired_participant_gfc_gain(learned, mismatched)

    def test_query_order_does_not_change_pairing(self):
        learned = scores("P1", "learned")
        shortcut = scores("P1", "shortcut")
        reordered = replace(shortcut, queries=tuple(reversed(shortcut.queries)))
        result = paired_participant_gfc_gain(learned, reordered)
        self.assertAlmostEqual(result.difference, 5.0 / 6.0)

    def test_participants_are_averaged_before_cohort_inference(self):
        learned = [scores("P1", "learned"), scores("P2", "learned")]
        shortcut = [scores("P1", "shortcut"), scores("P2", "shortcut")]
        cohort = paired_cohort_gfc_gain(learned, shortcut)
        self.assertEqual(len(cohort.participants), 2)
        self.assertAlmostEqual(cohort.mean_difference, 5.0 / 6.0)


class InferenceTest(unittest.TestCase):
    def test_bootstrap_is_deterministic_for_fixed_seed(self):
        learned = [scores(f"P{index}", "learned") for index in range(1, 4)]
        shortcut = [scores(f"P{index}", "shortcut") for index in range(1, 4)]
        cohort = paired_cohort_gfc_gain(learned, shortcut)
        first = bootstrap_gfc_gain(cohort, resamples=2_000, seed=123)
        second = bootstrap_gfc_gain(cohort, resamples=2_000, seed=123)
        self.assertEqual(first, second)
        self.assertEqual(first.participant_count, 3)
        self.assertTrue(first.positive_supported)

    def test_power_calculation_is_optional_and_requires_variation(self):
        learned = [scores("P1", "learned"), scores("P2", "learned")]
        shortcut = [scores("P1", "shortcut"), scores("P2", "shortcut")]
        constant = paired_cohort_gfc_gain(learned, shortcut)
        with self.assertRaisesRegex(ValueError, "variation"):
            plan_prospective_power(constant)
        varied = replace(
            constant,
            participants=(
                constant.participants[0],
                replace(constant.participants[1], difference=0.5),
            ),
            mean_difference=(constant.participants[0].difference + 0.5) / 2.0,
        )
        result = plan_prospective_power(varied)
        self.assertTrue(0.0 <= result.planned_power <= 1.0)


class SyntheticEndToEndTest(unittest.TestCase):
    def test_recording_features_to_paired_summary(self):
        training_rows = []
        training_subjects = []
        training_cells = []
        for subject_index, subject_id in enumerate(("T1", "T2")):
            for cell in CANONICAL_CELLS:
                training_rows.append(
                    [
                        float(cell.speed == "FGS"),
                        float(cell.clothing == "WJ"),
                        float(cell.direction == "L2R"),
                        float(subject_index),
                    ]
                )
                training_subjects.append(subject_id)
                training_cells.append(cell)
        training_rows = np.asarray(training_rows, dtype=np.float64)
        condition_adapter = fit_condition_adapter(
            training_rows, training_subjects, training_cells
        )
        gait_adapter = fit_gait_adapter(training_rows, training_subjects, training_cells)
        training_condition = condition_adapter.transform(training_rows)
        training_gait = gait_adapter.transform(training_rows)
        condition_normalizer = fit_raw_normalizer(
            training_condition, dimension_policy="retain_all"
        )
        gait_normalizer = fit_raw_normalizer(training_gait, dimension_policy="retain_all")
        shortcut_condition_normalizer = fit_raw_normalizer(
            np.zeros((len(training_rows), 4)), dimension_policy="retain_all"
        )
        shortcut_gait_normalizer = fit_raw_normalizer(
            np.zeros((len(training_rows), 2)), dimension_policy="retain_all"
        )

        learned_scores = []
        shortcut_scores = []
        for subject_id in ("E1", "E2"):
            raw = np.asarray(
                [
                    [
                        float(cell.speed == "FGS"),
                        float(cell.clothing == "WJ"),
                        float(cell.direction == "L2R"),
                        0.5,
                    ]
                    for cell in CANONICAL_CELLS
                ]
            )
            learned_condition = condition_normalizer.transform(
                condition_adapter.transform(raw)
            )
            learned_gait = gait_normalizer.transform(gait_adapter.transform(raw))
            shortcut_condition = shortcut_condition_normalizer.transform(
                np.zeros((8, 4))
            )
            shortcut_gait = shortcut_gait_normalizer.transform(np.zeros((8, 2)))

            def make(path, condition_values, gait_values):
                return [
                    Recording.from_windows(
                        subject_id=subject_id,
                        recording_id=f"{subject_id}:{cell.key}",
                        cell=cell,
                        condition_windows=[condition_values[index]] * 3,
                        gait_windows=[gait_values[index]] * 3,
                        window_ids=[
                            f"{subject_id}:{cell.key}:w{window}" for window in range(3)
                        ],
                    )
                    for index, cell in enumerate(CANONICAL_CELLS)
                ]

            learned_scores.append(
                evaluate_participant(
                    make("learned", learned_condition, learned_gait),
                    split="development",
                    seed=11,
                    representation="learned",
                )
            )
            shortcut_scores.append(
                evaluate_participant(
                    make("shortcut", shortcut_condition, shortcut_gait),
                    split="development",
                    seed=11,
                    representation="shortcut",
                )
            )
        cohort = paired_cohort_gfc_gain(learned_scores, shortcut_scores)
        summary = bootstrap_gfc_gain(cohort, resamples=1_000, seed=11).to_dict()
        self.assertEqual(summary["participant_count"], 2)
        self.assertGreater(summary["point_estimate"], 0.0)


if __name__ == "__main__":
    unittest.main()
