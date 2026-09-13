"""Sensitivity controls and a second solver for the logistic correction."""

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import expit, logit

from .experiment import METHODS
from .model import FloatArray, Population, fit_logistic, make_population, sample_counts


def direct_metrics(prediction, probability, mass, bins=15, threshold=0.8) -> dict:
    """Reconstruct the four reported metrics separately from the experiment code."""
    groups = np.minimum(np.floor(prediction * bins).astype(int), bins - 1)
    return {
        "ece": float(
            sum(
                abs(mass[groups == b] @ (probability - prediction)[groups == b])
                for b in range(bins)
            )
        ),
        "excess_brier": float(mass @ ((prediction - probability) ** 2)),
        "regret": float(
            mass
            @ (
                np.abs(probability - threshold)
                * ((prediction >= threshold) != (probability >= threshold))
            )
        ),
        "cell_error": float(mass @ np.abs(prediction - probability)),
    }


def fit_profile(
    population: Population,
    count: FloatArray,
    correct: FloatArray,
    weights: FloatArray,
    by_regime: bool,
) -> FloatArray:
    """Solve the intercept by root finding and minimize over the slope."""
    prediction = np.empty(12)
    for regime in ([0, 1] if by_regime else [None]):
        selected = np.ones(12, dtype=bool) if regime is None else population.regime == regime
        design = np.column_stack([np.ones(selected.sum()), logit(population.raw_score[selected])])
        mass = count[selected] * weights[selected]
        successes = correct[selected] * weights[selected]
        total = mass.sum()
        if total <= 0:
            raise ValueError("Cannot fit a correction without positive observation weight.")
        mass, successes = mass / total, successes / total

        def objective(parameters):
            linear = design @ parameters
            return float(mass @ np.logaddexp(0, linear) - successes @ linear)

        def profile(slope):
            def gradient(intercept):
                return float(mass @ expit(intercept + slope * design[:, 1]) - successes.sum())

            if gradient(-15) >= 0:
                intercept = -15.0
            elif gradient(15) <= 0:
                intercept = 15.0
            else:
                intercept = brentq(gradient, -15, 15, xtol=1e-13)
            return objective(np.array([intercept, slope])), intercept

        optimum = minimize_scalar(
            lambda slope: profile(slope)[0],
            bounds=(0.001, 10),
            method="bounded",
            options={"xatol": 1e-12},
        )
        if not optimum.success:
            raise RuntimeError(f"Profile optimizer did not converge: {optimum.message}")
        slope = min([0.001, 10, optimum.x], key=lambda value: profile(value)[0])
        parameters = np.array([profile(slope)[1], slope])
        prediction[selected] = expit(design @ parameters)
    return prediction


def check_saved_predictions(rows: list[dict], population: Population) -> float:
    target = population.target_mass()
    for row in rows:
        metrics = direct_metrics(np.array(row["prediction"]), population.probability, target)
        for metric, value in metrics.items():
            np.testing.assert_allclose(value, row[metric], atol=1e-12)

    lookup = {(row["mu"], row["seed"], row["method"]): row for row in rows}
    differences = []
    corrections = (
        ("Global", False, False),
        ("Regime", False, True),
        ("Weighted global", True, False),
        ("Weighted regime", True, True),
    )
    for mu in (0.2, 0.05, 0.01):
        weights = target / population.action_mass(mu)
        for seed in range(20):
            count, correct = sample_counts(population, mu, seed, 6000)
            for method, weighted, by_regime in corrections:
                prediction = fit_profile(
                    population, count, correct, weights if weighted else np.ones(12), by_regime
                )
                reference = lookup[mu, seed, method]["prediction"]
                differences.append(float(np.max(np.abs(prediction - reference))))
    maximum = max(differences)
    if maximum >= 2e-8:
        raise AssertionError(f"Alternative optimizer exceeds the article's tolerance: {maximum}")
    return maximum


def sensitivity_controls(rows: list[dict], population: Population) -> list[dict]:
    target = population.target_mass()
    records = []
    for mu in (0.2, 0.05, 0.01):
        for method in METHODS:
            predictions = [
                np.array(row["prediction"])
                for row in rows
                if row["mu"] == mu and row["method"] == method
            ]
            bin_errors = {
                str(bins): float(
                    np.mean(
                        [
                            direct_metrics(f, population.probability, target, bins=bins)["ece"]
                            for f in predictions
                        ]
                    )
                )
                for bins in (5, 10, 15, 30)
            }
            regrets = {
                str(threshold): float(
                    np.mean(
                        [
                            direct_metrics(f, population.probability, target, threshold=threshold)[
                                "regret"
                            ]
                            for f in predictions
                        ]
                    )
                )
                for threshold in (0.7, 0.8, 0.9)
            }
            records.append({"mu": mu, "method": method, "bins": bin_errors, "thresholds": regrets})
    return records


def population_controls(population: Population) -> list[dict]:
    generators = {
        "additive": population.probability,
        "no_action_effect": population.state,
        "logistic": expit(
            logit(population.state)
            + (1 - 2 * population.action) * np.where(population.regime == 0, 0.65, 0.30)
        ),
    }
    records = []
    for name, probability in generators.items():
        for mu, target_probability in ((0.2, 0.2), (0.2, 0.8), (0.8, 0.2)):
            source = population.action_mass(mu)
            target = population.action_mass(target_probability)
            weights = target / source
            predictions = {
                "Regime": fit_logistic(population, source, source * probability, np.ones(12), True),
                "Weighted global": fit_logistic(
                    population, source, source * probability, weights, False
                ),
                "Weighted regime": fit_logistic(
                    population, source, source * probability, weights, True
                ),
            }
            if mu == target_probability or name == "no_action_effect":
                np.testing.assert_allclose(
                    predictions["Regime"], predictions["Weighted regime"], atol=1e-7
                )
            for method, prediction in predictions.items():
                records.append(
                    {
                        "generator": name,
                        "mu": mu,
                        "target": target_probability,
                        "method": method,
                        **direct_metrics(prediction, probability, target),
                    }
                )
    return records


def check_no_shift(population: Population) -> float:
    differences = []
    source = population.action_mass(0.2)
    for seed in range(200):
        count, correct = sample_counts(population, 0.2, seed, 6000)
        unweighted = fit_logistic(population, count, correct, np.ones(12), True)
        weighted = fit_logistic(population, count, correct, source / source, True)
        differences.append(float(np.max(np.abs(unweighted - weighted))))
    if max(differences) != 0:
        raise AssertionError("Identical policies produced different corrections.")
    return max(differences)


def run_controls(rows: list[dict]) -> dict:
    population = make_population()
    return {
        "independent_solver_max_prediction_difference": check_saved_predictions(rows, population),
        "no_shift_sample_max_difference": check_no_shift(population),
        "sensitivity": sensitivity_controls(rows, population),
        "population_controls": population_controls(population),
        "saved_prediction_rows_rechecked": len(rows),
    }
