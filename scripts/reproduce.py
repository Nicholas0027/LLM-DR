#!/usr/bin/env python3
"""Entry points for the released experiment; live runs require explicit opt-in."""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MODELS = ["deepseek_v4", "nearest_baseline", "highest_fee_baseline", "profit_baseline", "persona_heuristic"]


def simulation_command(config, models):
    command = [sys.executable, "scripts/run_scaled_simulation_experiment.py"]
    for name, value in config.items():
        option = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(option)
            elif name == "route_tool_in_prompt":
                command.append("--no-route-tool-in-prompt")
        elif value is not None:
            command.extend([option, str(value)])
    return command + ["--models", *models]


def run(command):
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/matplotlib"))
    print(shlex.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def evaluate():
    run([
        sys.executable, "scripts/compare_simulation_to_empirical.py",
        "--orders", "data/empirical/orders.csv",
        "--wave-data", "data/empirical/waves.csv",
        "--assignments", "data/persona/rider_persona_assignments.csv",
        "--simulation-run", "reference/simulation",
        "--service-date", "20200215", "--start-hour", "8", "--end-hour", "20",
        "--grid-bins", "12", "--output-dir", "outputs/validation",
    ])
    import pandas as pd

    for reference in sorted((ROOT / "reference/simulation/empirical_validation").glob("*.csv")):
        actual = ROOT / "outputs/validation" / reference.name
        pd.testing.assert_frame_equal(
            pd.read_csv(reference), pd.read_csv(actual), check_dtype=False, rtol=1e-8, atol=1e-8,
        )
    print("Recomputed evaluation matches all frozen reference tables.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["verify", "evaluate", "smoke", "simulate", "personas", "figures"])
    parser.add_argument("--execute", action="store_true", help="Opt in to live simulation API requests.")
    parser.add_argument("--models", nargs="+", default=MODELS)
    args = parser.parse_args()
    if args.task in ("simulate", "smoke"):
        name = "main_experiment.json" if args.task == "simulate" else "smoke_experiment.json"
        config = json.loads((ROOT / "config" / name).read_text())
        models = args.models if args.task == "simulate" else ["persona_heuristic"]
        command = simulation_command(config, models)
        if args.task == "simulate" and not args.execute:
            print(shlex.join(command))
            print("Preview only. Add --execute to allow paid LLM and live routing requests.")
        else:
            run(command)
    elif args.task == "verify":
        run([sys.executable, "scripts/verify_release.py"])
    elif args.task == "evaluate":
        evaluate()
    elif args.task == "personas":
        run([
            sys.executable, "scripts/persona_clustering_experiment.py",
            "--wave-data", "data/empirical/waves.csv",
            "--order-data", "data/empirical/orders_filtered.tsv", "--output-dir", "outputs/persona",
            "--k", "4", "--min-waves", "1", "--seed", "42", "--k-min", "2", "--k-max", "8",
            "--n-init", "80", "--stability-runs", "40",
        ])
        import pandas as pd

        columns = ["courier_id", "cluster", "persona"]
        pd.testing.assert_frame_equal(
            pd.read_csv(ROOT / "data/persona/rider_persona_assignments.csv")[columns],
            pd.read_csv(ROOT / "outputs/persona/rider_persona_assignments.csv")[columns],
        )
        print("All 710 reference persona assignments reproduced.")
    else:
        evaluate()
        run([
            sys.executable, "scripts/plot_persona_delivery_patterns.py",
            "--assignments", "data/persona/rider_persona_assignments.csv",
            "--output-dir", "outputs/figures",
        ])
        run([
            sys.executable, "scripts/plot_scaled_simulation_figures.py",
            "--simulation-run", "reference/simulation", "--model", "deepseek_v4", "--model-label", "LLM-DR",
            "--output-dir", "outputs/figures", "--validation-dir", "outputs/validation",
            "--empirical-rider-features", "data/persona/rider_persona_features.csv",
            "--persona-assignments", "data/persona/rider_persona_assignments.csv",
            "--orders", "data/empirical/orders.csv", "--wave-data", "data/empirical/waves.csv",
            "--service-date", "20200215", "--start-hour", "8", "--end-hour", "20",
        ])


if __name__ == "__main__":
    main()
