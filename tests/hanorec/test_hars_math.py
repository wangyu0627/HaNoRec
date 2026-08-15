import math
import unittest

from hanorec.hars.math import (
    dynamic_beta,
    model_responsiveness,
    normalize_hardness,
    probability_distance,
    softmax,
)


class HaRSMathTest(unittest.TestCase):
    def test_softmax_is_stable_and_normalized(self):
        values = softmax([1000.0, 1001.0, 1002.0])

        self.assertAlmostEqual(sum(values), 1.0)
        self.assertGreater(values[2], values[1])
        self.assertGreater(values[1], values[0])

    def test_probability_distance_rejects_different_lengths(self):
        with self.assertRaisesRegex(ValueError, "equal length"):
            probability_distance([0.5, 0.5], [1.0])

    def test_harder_pair_receives_lower_lambda(self):
        easy = probability_distance([0.9, 0.1], [0.1, 0.9])
        hard = probability_distance([0.51, 0.49], [0.49, 0.51])

        values = normalize_hardness([easy, hard])

        self.assertGreater(values[0], values[1])
        self.assertAlmostEqual(
            sum(values) / len(values),
            sum(values) / 2,
        )

    def test_responsiveness_trims_two_extremes(self):
        eta = model_responsiveness([-100.0, 1.0, 1.0, 100.0])

        self.assertTrue(math.isfinite(eta))
        self.assertGreater(eta, 0.0)

    def test_responsiveness_handles_zero_mean(self):
        eta = model_responsiveness([-1.0, 1.0])

        self.assertAlmostEqual(eta, 1.0)

    def test_dynamic_beta_is_per_example(self):
        values = dynamic_beta([1.0, 0.5], responsiveness=0.8, beta0=0.1)

        self.assertAlmostEqual(values[0], 0.08)
        self.assertAlmostEqual(values[1], 0.04)

    def test_dynamic_beta_applies_positive_floor(self):
        values = dynamic_beta([0.0], responsiveness=0.0, beta0=0.1, floor=1e-5)

        self.assertEqual(values, [1e-5])

    def test_non_finite_input_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            softmax([0.0, math.inf])

    def test_empty_input_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            normalize_hardness([])

