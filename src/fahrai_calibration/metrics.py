"""Exact risks on the twelve target-population cells."""

import numpy as np

from .model import FloatArray


def evaluate(
    prediction: FloatArray,
    probability: FloatArray,
    mass: FloatArray,
    bins: int = 15,
    threshold: float = 0.8,
) -> dict:
    bin_index = np.minimum((prediction * bins).astype(int), bins - 1)
    calibration_error = sum(
        abs(
            np.sum(
                mass[bin_index == index]
                * (prediction[bin_index == index] - probability[bin_index == index])
            )
        )
        for index in range(bins)
    )
    selected = prediction >= threshold
    oracle = probability >= threshold
    coverage = float(np.sum(mass * selected))
    selected_error = (
        float(np.sum(mass * (1 - probability) * selected) / coverage) if coverage > 0 else None
    )
    return {
        "ece": float(calibration_error),
        "excess_brier": float(np.sum(mass * (prediction - probability) ** 2)),
        "regret": float(np.sum(mass * np.abs(probability - threshold) * (selected != oracle))),
        "coverage": coverage,
        "selected_error": selected_error,
        "cell_error": float(np.sum(mass * np.abs(prediction - probability))),
    }


def effective_sample_size(count: FloatArray, weights: FloatArray) -> float:
    return float((count @ weights) ** 2 / (count @ (weights * weights)))
