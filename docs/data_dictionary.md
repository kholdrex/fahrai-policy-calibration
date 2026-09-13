# Data dictionary

All files describe the finite synthetic population used in the article. They contain no personal data.

## Population

`population.csv` has 12 rows. `cell_id` is zero based, ordered first by regime (0, 1), then by state (0.55, 0.70, 0.85), then by action (0, 1).

| Field | Meaning |
| --- | --- |
| `regime` | Evidence-regime index, 0 or 1 |
| `state` | Designed baseline probability, denoted Z in the article |
| `action` | Task-action index; action 1 has lower correctness |
| `probability` | True conditional correctness Z + (1 − 2A)δᵣ, with δ₀ = 0.12 and δ₁ = 0.06 |
| `raw_score` | Logistic transform of 1.15 logit(Z) + bᵣ, with b₀ = 0.25 and b₁ = −0.15 |
| `target_mass` | Joint mass of the state–action cell; action-1 probability is 0.80 and each of the six states has mass 1/6 |

## Calibration samples

`calibration_counts.csv` has one row per logging setting, seed, and cell. The key is (`mu`, `seed`, `cell_id`). There are 3 × 200 × 12 = 7,200 rows.

`mu` is the logging probability of action 1: 0.20, 0.05, or 0.01. `seed` ranges from 0 through 199. `n` is the number of sampled observations in the cell; `k` is the number of correct responses. Within a sample, n sums to 6,000 and 0 ≤ k ≤ n. A zero count is a possible sampling result, not a missing record. No sequence of learner interactions is simulated.

## Predictions and metrics

`replications.json` has 3,600 records keyed by (`mu`, `seed`, `method`). `prediction` is a vector of 12 probabilities in `cell_id` order. `predictions.csv` contains the same values, one probability per row.

| Field | Meaning |
| --- | --- |
| `ece` | Target-mass-weighted calibration error using 15 equal-width bins; probability 1 belongs to the last bin |
| `excess_brier` | Target mean of (prediction − true probability)² |
| `regret` | Excess threshold-decision loss relative to the conditional oracle at threshold 0.80 |
| `coverage` | Target mass of cells whose prediction is at least 0.80 |
| `selected_error` | Expected incorrect-response probability conditional on selection; null when coverage is zero |
| `cell_error` | Target mean absolute difference between predicted and true probability |
| `ess` | Weight-concentration diagnostic (Σ nᵢwᵢ)² / Σ nᵢwᵢ²; common to the six estimators in a sample |

`results.json` summarizes each setting and estimator across 200 samples. `sd` uses denominator 199; `mcse` is sd/√200. Paired differences are Weighted regime minus Regime within the same setting and seed. Their intervals are mean ± 1.96 mcse, measuring Monte Carlo precision.

`controls.json` contains calibration-bin counts 5, 10, 15, and 30; decision thresholds 0.70, 0.80, and 0.90; alternative response models; reverse policy shift; and no-shift checks. Population controls fit exact population masses rather than sampled counts.

`table_1.csv` exports means and standard deviations for the adequate-overlap setting (`mu=0.20`). `table_2.csv` exports its bin-sensitivity results for four estimators. The article rounds these files to five and four decimal places, respectively.
