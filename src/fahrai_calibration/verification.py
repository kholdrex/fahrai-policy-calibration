"""Check probability identities, saved metrics, and reproducibility."""

import math

import numpy as np

from .experiment import METHODS, SUMMARY_METRICS, ExperimentConfig, paired_intervals, summarize
from .metrics import effective_sample_size, evaluate
from .model import fit_logistic, make_population, sample_counts


def compare_values(expected, actual, path="root") -> None:
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            raise AssertionError(f"Different fields at {path}")
        for key in expected:
            compare_values(expected[key], actual[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            raise AssertionError(f"Different lengths at {path}")
        for index, (left, right) in enumerate(zip(expected, actual)):
            compare_values(left, right, f"{path}[{index}]")
    elif isinstance(expected, float):
        if not math.isclose(expected, actual, rel_tol=1e-8, abs_tol=1e-10):
            raise AssertionError(f"Different values at {path}: {expected} != {actual}")
    elif expected != actual:
        raise AssertionError(f"Different values at {path}: {expected} != {actual}")


def check_identities() -> None:
    population = make_population()
    probability = population.probability
    prediction = population.raw_score
    target = population.target_mass()

    rng = np.random.default_rng(991)
    source_mass = rng.uniform(0.1, 1, 24)
    source_mass /= source_mass.sum()
    target_mass = rng.uniform(0.1, 1, 24)
    target_mass /= target_mass.sum()
    correctness = rng.uniform(0, 1, 24)
    scores = np.repeat([0.2, 0.4, 0.6, 0.8], 6)
    weights = target_mass / source_mass
    for score in np.unique(scores):
        selected = scores == score
        source = source_mass[selected] / source_mass[selected].sum()
        destination = target_mass[selected] / target_mass[selected].sum()
        mean_weight = source @ weights[selected]
        mean_correctness = source @ correctness[selected]
        covariance = (
            source @ (weights[selected] * correctness[selected]) - mean_weight * mean_correctness
        )
        np.testing.assert_allclose(
            destination @ correctness[selected] - score,
            mean_correctness - score + covariance / mean_weight,
        )

    for threshold in (0.7, 0.8, 0.9):
        decision = prediction >= threshold
        oracle = probability >= threshold

        def loss(choice):
            return (
                threshold * choice * (1 - probability)
                + (1 - threshold) * (1 - choice) * probability
            )

        np.testing.assert_allclose(
            target @ (loss(decision) - loss(oracle)),
            target @ (np.abs(probability - threshold) * (decision != oracle)),
        )
    expected_brier = target @ (
        probability * (1 - prediction) ** 2 + (1 - probability) * prediction**2
    )
    oracle_brier = target @ (probability * (1 - probability))
    np.testing.assert_allclose(
        expected_brier - oracle_brier, target @ ((prediction - probability) ** 2)
    )

    count = np.full(12, 500)
    correct = np.rint(count * probability)
    np.testing.assert_allclose(
        fit_logistic(population, count, correct, np.ones(12), True),
        fit_logistic(population, count, correct, np.full(12, 3.0), True),
        atol=1e-6,
    )
    oracle_metrics = evaluate(probability, probability, target)
    for key in ("ece", "excess_brier", "regret", "cell_error"):
        np.testing.assert_allclose(oracle_metrics[key], 0, atol=1e-12)
    for mu in (0.2, 0.05, 0.01):
        source = population.action_mass(mu)
        np.testing.assert_allclose(source * (target / source), target)
    np.testing.assert_allclose([0.8 * 0.9 + 0.2 * 0.5, 0.2 * 0.9 + 0.8 * 0.5], [0.82, 0.58])
    np.testing.assert_allclose(
        [0.8 * (1 - 0.58), 0.2 * 0.58, 0.2 * (0.8 * 0.1) + 0.8 * (0.2 * 0.5)], [0.336, 0.116, 0.096]
    )


def check_saved_results(results: dict, rows: list[dict]) -> None:
    config = ExperimentConfig()
    expected_keys = {
        (mu, seed, method)
        for mu in config.logging_probabilities
        for seed in range(config.replications)
        for method in METHODS
    }
    lookup = {(row["mu"], row["seed"], row["method"]): row for row in rows}
    if len(rows) != 3600 or set(lookup) != expected_keys:
        raise AssertionError("Expected exactly 3,600 distinct setting/seed/estimator records.")
    compare_values(results["summary"], summarize(rows, config))
    compare_values(results["paired"], paired_intervals(rows, config))
    population = make_population()
    target = population.target_mass()
    for mu in config.logging_probabilities:
        weights = target / population.action_mass(mu)
        for seed in range(config.replications):
            count, correct = sample_counts(population, mu, seed, config.sample_size)
            compare_values(
                lookup[mu, seed, "Context cells"]["prediction"],
                ((correct + 0.5) / (count + 1)).tolist(),
            )
            compare_values(lookup[mu, seed, "Raw"]["prediction"], population.raw_score.tolist())
            for method in METHODS:
                row = lookup[mu, seed, method]
                prediction = np.array(row["prediction"])
                if prediction.shape != (12,) or not np.all((prediction >= 0) & (prediction <= 1)):
                    raise AssertionError(f"Invalid predictions for {(mu, seed, method)}")
                compare_values(row["ess"], effective_sample_size(count, weights))
                selected = prediction >= 0.8
                coverage = float(target @ selected)
                compare_values(row["coverage"], coverage)
                error = (
                    float(target @ ((1 - population.probability) * selected) / coverage)
                    if coverage > 0
                    else None
                )
                compare_values(row["selected_error"], error)
    for summary in results["summary"]:
        for key in SUMMARY_METRICS:
            if not all(math.isfinite(value) for value in summary[key].values()):
                raise AssertionError("Non-finite summary statistic.")


def compare_experiments(reference: dict, current: dict) -> None:
    for key in ("n", "replications", "target_action1", "seeds"):
        compare_values(reference["config"][key], current["config"][key])
    for key in ("summary", "paired", "population_clipping", "example_predictions"):
        compare_values(reference[key], current[key])
