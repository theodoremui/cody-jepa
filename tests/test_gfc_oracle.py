from dataclasses import replace
from fractions import Fraction
import json
import unittest

from cody_jepa.evaluation.gfc.oracle import (
    BINARY_COMPLEMENTARY_TWO_DONOR,
    EXCLUDE_DONORS,
    FRACTIONAL_AVERAGE_RANK,
    RETAIN_ALL,
    Factor,
    FactorialDesign,
    compile_binary_complement_protocol,
    compile_healthgait_gfc_v2_protocol,
    enumerate_oracle_spectrum,
    score_oracle_query,
)


def binary_design(factor_count):
    return FactorialDesign(
        tuple(Factor(f"factor_{index}", ("zero", "one")) for index in range(factor_count))
    )


class ExactSpectrumTest(unittest.TestCase):
    def test_every_subset_matches_the_analytic_full_gallery_spectrum(self):
        for factor_count in range(2, 6):
            with self.subTest(factor_count=factor_count):
                design = binary_design(factor_count)
                protocol = compile_binary_complement_protocol(design)
                spectrum = enumerate_oracle_spectrum(protocol)
                self.assertEqual(len(protocol.queries), factor_count * (2**factor_count))
                self.assertEqual(len(spectrum.entries), 2**factor_count)
                for entry in spectrum.entries:
                    tied_cells = 2 ** (factor_count - len(entry.recovered_factors))
                    self.assertEqual(entry.top1, Fraction(1, tied_cells))
                    self.assertEqual(entry.mrr, Fraction(2, tied_cells + 1))
                    self.assertIsInstance(entry.top1, Fraction)
                    self.assertIsInstance(entry.mrr, Fraction)
                    for score in entry.query_scores:
                        self.assertEqual(score.strictly_closer_count, 0)
                        self.assertEqual(score.tie_size, tied_cells)
                        self.assertEqual(score.top1, Fraction(1, tied_cells))
                        self.assertEqual(score.reciprocal_rank, Fraction(2, tied_cells + 1))

    def test_healthgait_protocol_has_exactly_the_declared_queries(self):
        protocol = compile_healthgait_gfc_v2_protocol()
        design = protocol.design
        self.assertEqual(design.factor_names, ("speed", "clothing", "direction"))
        self.assertEqual(protocol.focal_factors, ("speed", "clothing"))
        self.assertEqual(protocol.donor_rule, BINARY_COMPLEMENTARY_TWO_DONOR)
        self.assertEqual(protocol.gallery_policy, RETAIN_ALL)
        self.assertEqual(protocol.tie_policy, FRACTIONAL_AVERAGE_RANK)
        self.assertEqual(len(protocol.queries), 16)
        self.assertEqual(
            [(query.target, query.focal_factor) for query in protocol.queries],
            [(cell, focal) for cell in design.cells for focal in ("speed", "clothing")],
        )

        for query in protocol.queries:
            focal_index = design.factor_index(query.focal_factor)
            self.assertEqual(len(query.donors), 2)
            self.assertEqual(len(set(query.donors)), 2)
            self.assertNotIn(query.target, query.donors)
            self.assertEqual(query.compose(), query.target)
            self.assertEqual(len(query.gallery), 8)
            self.assertIn(query.target, query.gallery)
            self.assertTrue(set(query.donors).issubset(query.gallery))
            for index, factor in enumerate(design.factors):
                opposite = factor.values[1 - factor.values.index(query.target[index])]
                expected_u = query.target[index] if index == focal_index else opposite
                expected_v = opposite if index == focal_index else query.target[index]
                self.assertEqual(query.donors[0][index], expected_u)
                self.assertEqual(query.donors[1][index], expected_v)
                self.assertEqual(query.factor_sources[index], 0 if index == focal_index else 1)

        by_subset = enumerate_oracle_spectrum(protocol).by_recovered_factors()
        expected = {
            (): (Fraction(1, 8), Fraction(2, 9)),
            ("speed",): (Fraction(1, 4), Fraction(2, 5)),
            ("clothing",): (Fraction(1, 4), Fraction(2, 5)),
            ("direction",): (Fraction(1, 4), Fraction(2, 5)),
            ("speed", "clothing"): (Fraction(1, 2), Fraction(2, 3)),
            ("speed", "direction"): (Fraction(1, 2), Fraction(2, 3)),
            ("clothing", "direction"): (Fraction(1, 2), Fraction(2, 3)),
            ("speed", "clothing", "direction"): (Fraction(1, 1), Fraction(1, 1)),
        }
        self.assertEqual(
            {key: (entry.top1, entry.mrr) for key, entry in by_subset.items()},
            expected,
        )

    def test_donor_exclusion_exposes_the_historical_asymmetry(self):
        protocol = compile_healthgait_gfc_v2_protocol(gallery_policy=EXCLUDE_DONORS)
        self.assertTrue(all(len(query.gallery) == 6 for query in protocol.queries))
        self.assertTrue(
            all(not set(query.donors).intersection(query.gallery) for query in protocol.queries)
        )
        by_subset = enumerate_oracle_spectrum(protocol).by_recovered_factors()
        expected = {
            (): (Fraction(1, 6), Fraction(2, 7)),
            ("speed",): (Fraction(1, 3), Fraction(1, 2)),
            ("clothing",): (Fraction(1, 3), Fraction(1, 2)),
            ("direction",): (Fraction(1, 3), Fraction(1, 2)),
            ("speed", "clothing"): (Fraction(1, 2), Fraction(2, 3)),
            ("speed", "direction"): (Fraction(3, 4), Fraction(5, 6)),
            ("clothing", "direction"): (Fraction(3, 4), Fraction(5, 6)),
            ("speed", "clothing", "direction"): (Fraction(1, 1), Fraction(1, 1)),
        }
        self.assertEqual(
            {key: (entry.top1, entry.mrr) for key, entry in by_subset.items()},
            expected,
        )

        clothing_direction = by_subset[("clothing", "direction")]
        per_focal = {"speed": [], "clothing": []}
        for query, score in zip(protocol.queries, clothing_direction.query_scores):
            per_focal[query.focal_factor].append(score.top1)
        self.assertEqual(set(per_focal["speed"]), {Fraction(1, 1)})
        self.assertEqual(set(per_focal["clothing"]), {Fraction(1, 2)})


class DeterminismAndSerializationTest(unittest.TestCase):
    def test_factor_and_focal_order_do_not_change_semantic_scores(self):
        first_design = FactorialDesign(
            (Factor("a", ("0", "1")), Factor("b", ("0", "1")), Factor("c", ("0", "1")))
        )
        second_design = FactorialDesign(tuple(reversed(first_design.factors)))
        first = enumerate_oracle_spectrum(
            compile_binary_complement_protocol(first_design, focal_factors=("a", "b"))
        )
        second = enumerate_oracle_spectrum(
            compile_binary_complement_protocol(second_design, focal_factors=("b", "a"))
        )
        first_scores = {
            frozenset(entry.recovered_factors): (entry.top1, entry.mrr) for entry in first.entries
        }
        second_scores = {
            frozenset(entry.recovered_factors): (entry.top1, entry.mrr) for entry in second.entries
        }
        self.assertEqual(first_scores, second_scores)

    def test_serialization_is_exact_json_safe_and_deterministic(self):
        spectrum = enumerate_oracle_spectrum(
            compile_healthgait_gfc_v2_protocol(gallery_policy=EXCLUDE_DONORS)
        )
        first = json.dumps(spectrum.to_dict(), sort_keys=True, separators=(",", ":"))
        second = json.dumps(spectrum.to_dict(), sort_keys=True, separators=(",", ":"))
        self.assertEqual(first, second)
        payload = json.loads(first)
        self.assertEqual(payload["donor_rule"], BINARY_COMPLEMENTARY_TWO_DONOR)
        self.assertEqual(payload["spectrum"][0]["top1_fraction"], "1/6")
        self.assertEqual(payload["spectrum"][-1]["top1_fraction"], "1")
        pair = next(
            row
            for row in payload["spectrum"]
            if row["recovered_factors"] == ["clothing", "direction"]
        )
        self.assertEqual(
            {item["focal_factor"]: item["top1_fraction"] for item in pair["focal_breakdown"]},
            {"speed": "1", "clothing": "1/2"},
        )


class ValidationTest(unittest.TestCase):
    def test_design_and_compiler_reject_malformed_inputs(self):
        with self.assertRaisesRegex(ValueError, "nonempty"):
            Factor("", ("0", "1"))
        with self.assertRaisesRegex(TypeError, "not text"):
            Factor("a", "01")
        with self.assertRaisesRegex(ValueError, "unique"):
            Factor("a", ("0", "0"))
        with self.assertRaisesRegex(ValueError, "unique"):
            FactorialDesign((Factor("a", ("0", "1")), Factor("a", ("x", "y"))))
        with self.assertRaisesRegex(ValueError, "at least two factors"):
            FactorialDesign((Factor("a", ("0", "1")),))

        nonbinary = FactorialDesign((Factor("a", ("0", "1", "2")), Factor("b", ("0", "1"))))
        with self.assertRaisesRegex(ValueError, "exactly two"):
            compile_binary_complement_protocol(nonbinary)

        design = binary_design(3)
        with self.assertRaisesRegex(TypeError, "not text"):
            compile_binary_complement_protocol(design, focal_factors="factor_0")
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            compile_binary_complement_protocol(design, focal_factors=())
        with self.assertRaisesRegex(ValueError, "unique"):
            compile_binary_complement_protocol(design, focal_factors=("factor_0", "factor_0"))
        with self.assertRaisesRegex(ValueError, "unknown focal"):
            compile_binary_complement_protocol(design, focal_factors=("missing",))
        with self.assertRaisesRegex(ValueError, "gallery_policy"):
            compile_binary_complement_protocol(design, gallery_policy="unknown")

    def test_protocol_validation_rejects_structural_corruption(self):
        protocol = compile_healthgait_gfc_v2_protocol()
        first = protocol.queries[0]

        with self.assertRaisesRegex(ValueError, "every target and focal"):
            enumerate_oracle_spectrum(replace(protocol, queries=protocol.queries[:-1]))
        with self.assertRaisesRegex(ValueError, "unsupported donor"):
            enumerate_oracle_spectrum(replace(protocol, donor_rule="unknown"))
        with self.assertRaisesRegex(ValueError, "unsupported tie"):
            enumerate_oracle_spectrum(replace(protocol, tie_policy="unknown"))
        with self.assertRaisesRegex(TypeError, "gallery_policy must be text"):
            enumerate_oracle_spectrum(replace(protocol, gallery_policy=[]))
        with self.assertRaisesRegex(TypeError, "CompiledQuery"):
            enumerate_oracle_spectrum(replace(protocol, queries=(object(),)))

        floating_index = replace(first, query_index=0.0)
        with self.assertRaisesRegex(TypeError, "indices must be integers"):
            enumerate_oracle_spectrum(
                replace(protocol, queries=(floating_index, *protocol.queries[1:]))
            )
        boolean_index = replace(first, query_index=False)
        with self.assertRaisesRegex(TypeError, "not booleans"):
            enumerate_oracle_spectrum(
                replace(protocol, queries=(boolean_index, *protocol.queries[1:]))
            )

        target_donor = replace(first, donors=(first.target, first.donors[1]))
        with self.assertRaisesRegex(ValueError, "target must not"):
            enumerate_oracle_spectrum(
                replace(protocol, queries=(target_donor, *protocol.queries[1:]))
            )

        missing_target = replace(
            first, gallery=tuple(cell for cell in first.gallery if cell != first.target)
        )
        with self.assertRaisesRegex(ValueError, "target must remain"):
            enumerate_oracle_spectrum(
                replace(protocol, queries=(missing_target, *protocol.queries[1:]))
            )

        duplicate_gallery = replace(first, gallery=(*first.gallery, first.gallery[0]))
        with self.assertRaisesRegex(ValueError, "gallery cells must be unique"):
            enumerate_oracle_spectrum(
                replace(protocol, queries=(duplicate_gallery, *protocol.queries[1:]))
            )

        invalid_source = replace(first, factor_sources=(2, *first.factor_sources[1:]))
        with self.assertRaisesRegex(ValueError, "source index"):
            enumerate_oracle_spectrum(
                replace(protocol, queries=(invalid_source, *protocol.queries[1:]))
            )

        wrong_composition = replace(first, factor_sources=(1, 1, 1))
        with self.assertRaisesRegex(ValueError, "complementary donor rule"):
            enumerate_oracle_spectrum(
                replace(protocol, queries=(wrong_composition, *protocol.queries[1:]))
            )

        target = first.target
        factors = protocol.design.factors
        opposite = tuple(
            factor.values[1 - factor.values.index(value)] for factor, value in zip(factors, target)
        )
        alternative_donors = (
            (target[0], target[1], opposite[2]),
            (opposite[0], opposite[1], target[2]),
        )
        forged_query = replace(
            first,
            donors=alternative_donors,
            factor_sources=(0, 0, 1),
        )
        self.assertEqual(forged_query.compose(), target)
        with self.assertRaisesRegex(ValueError, "complementary donor rule"):
            enumerate_oracle_spectrum(
                replace(protocol, queries=(forged_query, *protocol.queries[1:]))
            )

    def test_recovered_factor_and_query_validation_is_explicit(self):
        protocol = compile_healthgait_gfc_v2_protocol()
        query = protocol.queries[0]
        with self.assertRaisesRegex(TypeError, "not text"):
            score_oracle_query(protocol, query, "speed")
        with self.assertRaisesRegex(ValueError, "unique"):
            score_oracle_query(protocol, query, ("speed", "speed"))
        with self.assertRaisesRegex(ValueError, "unknown recovered"):
            score_oracle_query(protocol, query, ("missing",))
        with self.assertRaisesRegex(ValueError, "not part"):
            score_oracle_query(protocol, replace(query, query_index=99), ("speed",))


if __name__ == "__main__":
    unittest.main()
