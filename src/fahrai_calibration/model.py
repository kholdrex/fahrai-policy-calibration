"""Study population and constrained logistic recalibration."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit, logit

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Population:
    regime: NDArray[np.int64]
    state: FloatArray
    action: NDArray[np.int64]
    probability: FloatArray
    raw_score: FloatArray

    def action_mass(self, probability_action_one: float) -> FloatArray:
        if not 0 <= probability_action_one <= 1:
            raise ValueError("Action probability must lie in [0, 1].")
        return np.where(self.action == 1, probability_action_one, 1 - probability_action_one) / 6

    def target_mass(self) -> FloatArray:
        # Preserve the probabilities used to generate the reference results.
        return np.where(self.action == 1, 0.8, 0.2) / 6


def make_population() -> Population:
    cells = [(r, z, a) for r in range(2) for z in (0.55, 0.70, 0.85) for a in range(2)]
    regime, state, action = np.array(cells).T
    regime = regime.astype(int)
    action = action.astype(int)
    probability = state + (1 - 2 * action) * np.where(regime == 0, 0.12, 0.06)
    raw_score = expit(1.15 * logit(state) + np.where(regime == 0, 0.25, -0.15))
    return Population(regime, state, action, probability, raw_score)


def sample_counts(
    population: Population, logging_probability: float, seed: int, size: int
) -> tuple[NDArray, NDArray]:
    rng = np.random.default_rng(seed)
    count = rng.multinomial(size, population.action_mass(logging_probability))
    correct = rng.binomial(count, population.probability)
    return count, correct


def fit_logistic(
    population: Population,
    count: FloatArray,
    correct: FloatArray,
    weights: FloatArray,
    by_regime: bool,
) -> FloatArray:
    """Fit normalized binomial log loss; keep the article's bounds and tolerances."""
    predictions = np.empty(len(population.probability))
    for regime in ([0, 1] if by_regime else [None]):
        selected = (
            np.ones(len(predictions), dtype=bool) if regime is None else population.regime == regime
        )
        design = np.column_stack([np.ones(selected.sum()), logit(population.raw_score[selected])])
        mass = count[selected] * weights[selected]
        successes = correct[selected] * weights[selected]
        total = mass.sum()
        if total <= 0:
            raise ValueError("Cannot fit a correction without positive observation weight.")

        def objective(parameters):
            linear = design @ parameters
            loss = (mass * np.logaddexp(0, linear) - successes * linear).sum() / total
            gradient = design.T @ (mass * expit(linear) - successes) / total
            return loss, gradient

        bounds = [(-15, 15), (0.001, 10)]
        optimum = minimize(
            objective,
            [0, 1],
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": 1e-13, "gtol": 1e-10, "maxiter": 500},
        )
        if not optimum.success:
            optimum = minimize(
                objective,
                optimum.x,
                jac=True,
                method="SLSQP",
                bounds=bounds,
                options={"ftol": 1e-12, "maxiter": 500},
            )
        if not optimum.success:
            raise RuntimeError(f"Logistic correction did not converge: {optimum.message}")
        predictions[selected] = expit(design @ optimum.x)
    return predictions
