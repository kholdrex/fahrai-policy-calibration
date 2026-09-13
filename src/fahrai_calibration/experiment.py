"""Generate calibration samples and compare six probability estimators."""

import platform
from dataclasses import dataclass

import numpy as np
import scipy

from .metrics import effective_sample_size, evaluate
from .model import fit_logistic, make_population, sample_counts

METHODS = ("Raw", "Global", "Regime", "Weighted global", "Weighted regime", "Context cells")
SUMMARY_METRICS = ("ece", "excess_brier", "regret", "coverage", "cell_error", "ess")


@dataclass(frozen=True)
class ExperimentConfig:
    sample_size: int = 6000
    replications: int = 200
    logging_probabilities: tuple[float, ...] = (0.2, 0.05, 0.01)


def summarize(rows: list[dict], config: ExperimentConfig) -> list[dict]:
    summary = []
    for mu in config.logging_probabilities:
        for method in METHODS:
            subset = [row for row in rows if row["mu"] == mu and row["method"] == method]
            record = {"mu": mu, "method": method}
            for key in SUMMARY_METRICS:
                values = np.array([row[key] for row in subset])
                record[key] = {
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)),
                    "mcse": float(values.std(ddof=1) / np.sqrt(config.replications)),
                }
            summary.append(record)
    return summary


def paired_intervals(rows: list[dict], config: ExperimentConfig) -> list[dict]:
    intervals = []
    for mu in config.logging_probabilities:
        for metric in ("ece", "excess_brier", "regret"):
            weighted = np.array(
                [
                    row[metric]
                    for row in rows
                    if row["mu"] == mu and row["method"] == "Weighted regime"
                ]
            )
            unweighted = np.array(
                [row[metric] for row in rows if row["mu"] == mu and row["method"] == "Regime"]
            )
            difference = weighted - unweighted
            error = difference.std(ddof=1) / np.sqrt(len(difference))
            intervals.append(
                {
                    "mu": mu,
                    "metric": metric,
                    "mean": float(difference.mean()),
                    "mcse": float(error),
                    "interval": [
                        float(difference.mean() - 1.96 * error),
                        float(difference.mean() + 1.96 * error),
                    ],
                }
            )
    return intervals


def run_experiment(config: ExperimentConfig = ExperimentConfig()) -> tuple[dict, list[dict]]:
    population = make_population()
    target = population.target_mass()
    rows = []
    example_predictions = {}
    for mu in config.logging_probabilities:
        weights = target / population.action_mass(mu)
        for seed in range(config.replications):
            count, correct = sample_counts(population, mu, seed, config.sample_size)
            unit_weights = np.ones(12)
            predictions = [
                population.raw_score,
                fit_logistic(population, count, correct, unit_weights, False),
                fit_logistic(population, count, correct, unit_weights, True),
                fit_logistic(population, count, correct, weights, False),
                fit_logistic(population, count, correct, weights, True),
                (correct + 0.5) / (count + 1),
            ]
            for method, prediction in zip(METHODS, predictions, strict=True):
                rows.append(
                    {
                        "mu": mu,
                        "seed": seed,
                        "method": method,
                        "ess": effective_sample_size(count, weights),
                        "prediction": prediction.tolist(),
                        **evaluate(prediction, population.probability, target),
                    }
                )
            if mu == 0.2 and seed == 0:
                example_predictions = {m: f.tolist() for m, f in zip(METHODS, predictions)}

    clipping = []
    for mu in config.logging_probabilities:
        source = population.action_mass(mu)
        weights = target / source
        for cap in (None, 10, 5):
            capped = weights if cap is None else np.minimum(weights, cap)
            prediction = fit_logistic(
                population, source, source * population.probability, capped, True
            )
            clipping.append(
                {"mu": mu, "cap": cap, **evaluate(prediction, population.probability, target)}
            )
    results = {
        "config": {
            "n": config.sample_size,
            "replications": config.replications,
            "target_action1": 0.8,
            "seeds": [0, config.replications - 1],
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "summary": summarize(rows, config),
        "paired": paired_intervals(rows, config),
        "population_clipping": clipping,
        "example_predictions": example_predictions,
    }
    return results, rows
