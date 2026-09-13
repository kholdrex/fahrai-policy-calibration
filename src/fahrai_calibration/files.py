"""Read saved evidence and export results for inspection."""

import csv
import hashlib
import json
from pathlib import Path

from .model import make_population, sample_counts


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def export_data(output: Path, results: dict, rows: list[dict], controls: dict) -> None:
    population = make_population()
    target = population.target_mass()
    write_csv(
        output / "population.csv",
        [
            {
                "cell_id": i,
                "regime": int(population.regime[i]),
                "state": population.state[i],
                "action": int(population.action[i]),
                "probability": population.probability[i],
                "raw_score": population.raw_score[i],
                "target_mass": target[i],
            }
            for i in range(12)
        ],
    )
    counts = []
    for mu in (0.2, 0.05, 0.01):
        for seed in range(200):
            count, correct = sample_counts(population, mu, seed, 6000)
            counts.extend(
                {"mu": mu, "seed": seed, "cell_id": i, "n": int(count[i]), "k": int(correct[i])}
                for i in range(12)
            )
    write_csv(output / "calibration_counts.csv", counts)
    write_csv(
        output / "predictions.csv",
        [
            {
                "mu": row["mu"],
                "seed": row["seed"],
                "method": row["method"],
                "cell_id": i,
                "prediction": prediction,
            }
            for row in rows
            for i, prediction in enumerate(row["prediction"])
        ],
    )
    table_one = []
    for summary in results["summary"]:
        if summary["mu"] == 0.2:
            record = {"method": summary["method"]}
            for metric in ("ece", "excess_brier", "regret"):
                for statistic in ("mean", "sd"):
                    record[f"{metric}_{statistic}"] = summary[metric][statistic]
            table_one.append(record)
    write_csv(output / "table_1.csv", table_one)
    write_csv(
        output / "table_2.csv",
        [
            {
                "method": row["method"],
                **{f"bins_{b}": row["bins"][b] for b in ("5", "10", "15", "30")},
            }
            for row in controls["sensitivity"]
            if row["mu"] == 0.2
            and row["method"] in ("Regime", "Weighted global", "Weighted regime", "Context cells")
        ],
    )


def check_manifest(root: Path) -> int:
    manifest = read_json(root / "SHA256.json")
    for name, expected in manifest.items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError(f"Checksum mismatch: {name}")
    return len(manifest)
