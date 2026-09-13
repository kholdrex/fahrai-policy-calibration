"""Commands for reproducing the experiment and checking its reference outputs."""

import argparse
import platform
import tempfile
import time
from pathlib import Path

import matplotlib
import numpy as np
import scipy

from .controls import run_controls
from .experiment import run_experiment
from .figures import draw_architecture, draw_overlap
from .files import check_manifest, export_data, read_json, write_json
from .verification import check_identities, check_saved_results, compare_experiments, compare_values


def verify(root: Path, reproduce: bool) -> dict:
    started = time.monotonic()
    manifest_files = check_manifest(root)
    results = read_json(root / "data/results.json")
    rows = read_json(root / "data/replications.json")
    controls = read_json(root / "data/controls.json")
    check_identities()
    check_saved_results(results, rows)
    current_controls = run_controls(rows)
    compared_fields = set(current_controls) - {"independent_solver_max_prediction_difference"}
    compare_values(
        {key: controls[key] for key in compared_fields},
        {key: current_controls[key] for key in compared_fields},
    )

    exact = None
    if reproduce:
        current_results, current_rows = run_experiment()
        compare_experiments(results, current_results)
        compare_values(rows, current_rows)
        exact = rows == current_rows and all(
            results[key] == current_results[key] for key in results if key != "config"
        )
    with tempfile.TemporaryDirectory(prefix="fahrai_exports_") as folder:
        output = Path(folder)
        export_data(output, results, rows, controls)
        for generated in output.glob("*.csv"):
            if generated.read_bytes() != (root / "data" / generated.name).read_bytes():
                raise AssertionError(f"CSV export differs: {generated.name}")

    return {
        "status": "PASS",
        "mode": "full_reproduction" if reproduce else "saved_evidence_check",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "system": platform.system(),
        "machine": platform.machine(),
        "checksum_files": manifest_files,
        "calibration_samples": 600,
        "prediction_vectors": len(rows),
        "scalar_predictions": 12 * len(rows),
        "numerical_tolerance": {"relative": 1e-8, "absolute": 1e-10},
        "regenerated_numerical_outputs_exactly_equal": exact,
        "alternative_optimizer_max_difference": current_controls[
            "independent_solver_max_prediction_difference"
        ],
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the FAHRAI calibration experiment.")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser(
        "verify", help="Check the reference evidence and probability identities"
    )
    check.add_argument(
        "--root", type=Path, default=Path.cwd(), help="Repository or supplement directory"
    )
    check.add_argument(
        "--reproduce", action="store_true", help="Regenerate all 600 calibration samples"
    )
    check.add_argument(
        "--report", type=Path, help="Write a JSON report outside the reference files"
    )
    run = commands.add_parser("run", help="Generate experiment results, CSV exports, and figures")
    run.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    if args.command == "verify":
        if args.report and args.report.resolve().is_relative_to((args.root / "data").resolve()):
            parser.error("The verification report must not overwrite reference data.")
        report = verify(args.root, args.reproduce)
        if args.report:
            write_json(args.report, report)
        import json

        print(json.dumps(report, indent=2))
        return

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("The output directory must be empty; choose a new directory.")
    output.mkdir(parents=True, exist_ok=True)
    results, rows = run_experiment()
    controls = run_controls(rows)
    write_json(output / "results.json", results)
    write_json(output / "replications.json", rows)
    write_json(output / "controls.json", controls)
    export_data(output, results, rows, controls)
    draw_architecture(output / "figures")
    draw_overlap(results, output / "figures")
    print(f"Saved results to {output}")
