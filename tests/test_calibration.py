import unittest

import numpy as np

from fahrai_calibration.controls import fit_profile
from fahrai_calibration.metrics import evaluate
from fahrai_calibration.model import fit_logistic, make_population, sample_counts
from fahrai_calibration.verification import check_identities, compare_values


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.population = make_population()
        self.target = self.population.target_mass()

    def test_probability_and_decision_identities(self):
        check_identities()

    def test_sampling_preserves_counts_and_seed(self):
        count, correct = sample_counts(self.population, 0.01, 17, 6000)
        repeated_count, repeated_correct = sample_counts(self.population, 0.01, 17, 6000)
        self.assertEqual(count.sum(), 6000)
        self.assertTrue(np.all((correct >= 0) & (correct <= count)))
        np.testing.assert_array_equal(count, repeated_count)
        np.testing.assert_array_equal(correct, repeated_correct)

    def test_weighted_population_fit_does_not_depend_on_logging_mix(self):
        predictions = []
        for mu in (0.2, 0.05, 0.01):
            source = self.population.action_mass(mu)
            predictions.append(
                fit_logistic(
                    self.population,
                    source,
                    source * self.population.probability,
                    self.target / source,
                    True,
                )
            )
        for prediction in predictions[1:]:
            np.testing.assert_allclose(predictions[0], prediction, atol=1e-7)

    def test_independent_optimizer_matches_at_weak_overlap(self):
        count, correct = sample_counts(self.population, 0.01, 5, 6000)
        weights = self.target / self.population.action_mass(0.01)
        expected = fit_profile(self.population, count, correct, weights, True)
        actual = fit_logistic(self.population, count, correct, weights, True)
        np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=0)

    def test_empty_selection_has_no_conditional_error(self):
        metrics = evaluate(np.full(12, 0.1), self.population.probability, self.target)
        self.assertEqual(metrics["coverage"], 0)
        self.assertIsNone(metrics["selected_error"])

    def test_probability_one_uses_last_calibration_bin(self):
        metrics = evaluate(np.ones(12), self.population.probability, self.target)
        self.assertAlmostEqual(metrics["ece"], self.target @ (1 - self.population.probability))

    def test_empty_fit_reports_missing_evidence(self):
        with self.assertRaisesRegex(ValueError, "positive observation weight"):
            fit_logistic(self.population, np.zeros(12), np.zeros(12), np.ones(12), True)

    def test_comparison_detects_changed_evidence(self):
        with self.assertRaises(AssertionError):
            compare_values({"prediction": [0.8]}, {"prediction": [0.81]})


if __name__ == "__main__":
    unittest.main()
