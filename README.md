# FAHRAI policy calibration

Code and synthetic data for *A method for recalibrating response probabilities under changes in task selection in adaptive learning systems*.

The experiment changes how tasks are selected while holding learner states and response probabilities fixed. It compares six estimators on a population of 12 state–action cells. Each of three logging policies uses 200 calibration samples of 6,000 observations. Evaluation sums over the known target population; it does not use a sampled test set.

All data are synthetic. The experiment uses neither learner records nor language-model outputs. It does not evaluate learning over time or a deployed language-model orchestrator.

## Run

The reference calculations used Python 3.12.14, NumPy 2.3.5, and SciPy 1.17.0 on Linux x86_64. Figures were checked with Matplotlib 3.10.8. `environment.json` records the verification environment; `requirements-lock.txt` pins the numerical and plotting dependencies.

From the repository root, using Python 3.12.14:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
python -m unittest discover -s tests -v
python -m fahrai_calibration verify
```

To regenerate every calibration sample and compare all predictions and numerical results with the saved evidence:

```sh
python -m fahrai_calibration verify --reproduce
```

To write a fresh set of results, CSV exports, and figures:

```sh
python -m fahrai_calibration run --output outputs
```

The output directory must be empty. The verification commands leave the reference files unchanged. An optional report can be saved with `--report outputs/verification.json`.

## Files

| Path | Contents |
| --- | --- |
| `src/fahrai_calibration/model.py` | Population, sampling, and constrained logistic fitting |
| `src/fahrai_calibration/experiment.py` | Six estimators, replication summaries, paired intervals, and weight clipping |
| `src/fahrai_calibration/metrics.py` | Exact population calibration and decision metrics |
| `src/fahrai_calibration/controls.py` | Profile-likelihood solver, bin and threshold sensitivity, and population controls |
| `src/fahrai_calibration/verification.py` | Transport, Brier, and decision identities; checks of saved results |
| `src/fahrai_calibration/figures.py` | The two manuscript figures |
| `data/results.json` | Summary statistics, paired intervals, and population clipping results |
| `data/replications.json` | 3,600 records, each containing one estimator's 12 predictions and metrics |
| `data/controls.json` | Saved sensitivity results and alternative-solver checks |
| `data/calibration_counts.csv` | 7,200 cell-count records across the 600 calibration samples |
| `data/predictions.csv` | 43,200 scalar predictions in long format |
| `data/population.csv` | Cell definitions, true probabilities, raw scores, and target masses |
| `data/table_1.csv`, `data/table_2.csv` | Numerical values underlying the two manuscript tables |
| `docs/data_dictionary.md` | Field definitions and sample construction |
| `docs/equations.tex` | Eight displayed equations; numbering differs between journal layouts |
| `figures/` | Publication figures at 300 dpi |
| `verification/reproduction.json` | The recorded verification run |
| `SHA256.json` | Checksums of the reference data and figures |

## What is checked

Verification reconstructs metrics from every saved prediction vector, checks the summary statistics and paired intervals, and regenerates the multinomial and binomial counts from seeds 0–199. A separate profile-likelihood solver checks 240 fitted vectors from 60 samples. Its largest prediction difference must remain below the manuscript's stated bound of 2 × 10⁻⁸.

The full reproduction command also refits every estimator in all 600 samples. Comparisons allow relative error 10⁻⁸ and absolute error 10⁻¹⁰ for floating-point arithmetic. The report separately records whether outputs were exactly equal. Other operating systems and numerical-library versions have not been validated; small numerical differences can affect predictions close to a bin boundary or decision threshold.

The original JSON evidence is retained unchanged. The `date` field in the original `controls.json` records the earlier audit date and is excluded from numerical comparisons. The alternative solver's residual is checked against the stated 2 × 10⁻⁸ bound; its precise value can change with floating-point evaluation. Execution details belong in the separate verification report.

## Scope of the implementation

The raw score omits the task action. Logistic corrections pool actions and use either one map or one map per evidence regime. The context-cell estimator uses action information and half-count smoothing. This is a designed comparison of information and coverage, not a ranking of deployed educational systems.

Randomness comes from NumPy's `default_rng` with PCG64. For each setting and seed, the generator first draws multinomial counts over the 12 cells, then binomial correctness counts conditional on those counts. These aggregated counts are sufficient for the implemented losses. Seeds are reused across logging settings; the six estimators within each sample share its counts.

