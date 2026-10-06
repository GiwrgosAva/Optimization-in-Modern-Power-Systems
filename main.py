"""Entry point: load one question's data, build and solve the model, save results and figures.

    python main.py                          # base case of Q1_caseA
    python main.py --question Q2_linear     # another case
    python main.py --scenarios              # also run the example sensitivity scenarios

Results (CSV, TXT, PNG) are written to ``results/<question>/``. Extend ``run_scenarios``
with your own scenarios, or add a new function per question, as your analysis grows.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np


from src.data_loader import load_question, list_questions
from src.model import FlexibleConsumerModel, LinearDisutilityModel, Results
from src.model import FlexibleConsumerModel, LinearDisutilityModel, QuadraticDisutilityModel, Results
from src.plotting import plot_duals, plot_inputs, plot_scenario_comparison, plot_schedule
from src.scenarios import scale_prices, scale_pv, set_tariffs

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def run_base_case(question: str, out: Path, show: bool) -> Results | None:
    data = load_question(question)
    print(data.summary(), "\n")
    plot_inputs(data, save_to=out / "inputs.png")

    if "Q2_linear" in question:
        model = LinearDisutilityModel(data).build()
    elif "Q2_quadratic" in question:
        model = QuadraticDisutilityModel(data).build()
    else:
        model = FlexibleConsumerModel(data).build()
    try:
        results = model.solve()
    except NotImplementedError as e:
        print(f"[skipped] {e}")
        return None

    print(results, "\n")
    results.save(out)
    plot_schedule(results, data, save_to=out / "schedule.png")
    plot_duals(results, data, save_to=out / "duals.png")
    if show:
        matplotlib.pyplot.show()
    return results


def run_scenarios(question: str, out: Path) -> dict[str, Results]:
    """Example sensitivity analysis. Replace with the scenarios you design in Question 1.g."""
    base = load_question(question)
    scenarios = {
        "base": base,
        "flat_prices": scale_prices(base, factor=0.0, keep_mean=True),
        "double_spread": scale_prices(base, factor=2.0, keep_mean=True),
        "no_tariffs": set_tariffs(base, import_tariff=0.0, export_tariff=0.0),
        "no_pv": scale_pv(base, factor=0.0),
    }
    runs: dict[str, Results] = {}
    for name, data in scenarios.items():
        results = FlexibleConsumerModel(data).build().solve()
        results.save(out, tag=name)
        runs[name] = results
        print(f"{name:>14}: cost {results.objective:8.2f} DKK | import {results.hourly['import'].sum():5.1f} kWh"
              f" | export {results.hourly['export'].sum():5.1f} kWh")
    plot_scenario_comparison(runs, "objective", save_to=out / "scenarios_cost.png")
    return runs

def run_linear_sweep(question: str, out: Path) -> dict[str, Results]:
    """Sweep the linear disutility parameter c^L for question 2b"""
    runs: dict[str, Results] = {}

    c_values = np.linspace(0.0, 3.0, 7)

    for c in c_values:
        name = f"cL_{c:.2f}"

        data = load_question(question)
        data.linear_disutility = c

        model = LinearDisutilityModel(data).build()
        results = model.solve()

        results.save(out, tag=name)
        runs[name] = results
        print(f"{name:>14}: cost {results.objective:8.2f} DKK | total load {results.hourly['load'].sum():5.1f} kWh")

        plot_scenario_comparison(runs, "objective", save_to=out / "linear_sweep_cost.png")
        return runs

def run_quadratic_sweep(question: str, out: Path) -> dict[str, Results]:
    """Sweep the quadratic disutility parameter c^Q for question 2c"""
    runs: dict[str, Results] = {}

    c_values = np.linspace(0.0, 0.5, 7)

    for c in c_values:
        name = f"cQ_{c:.3f}"
        
        data = load_question(question)
        data.quadratic_disutility = c

        model = QuadraticDisutilityModel(data).build()
        results = model.solve()

        results.save(out, tag=name)
        runs[name] = results
        print(f"{name:>14}: cost {results.objective:8.2f} DKK | total load {results.hourly['load'].sum():5.1f} kWh")

    plot_scenario_comparison(runs, "objective", save_to=out / "quadratic_sweep_cost.png")
    return runs    


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--question", default="Q1_caseA", choices=list_questions(), help="data case to use")
    parser.add_argument("--scenarios", action="store_true", help="also run the example sensitivity scenarios")
    parser.add_argument("--show", action="store_true", help="open the figures in a window")
    args = parser.parse_args()

    out = RESULTS_DIR / args.question
    out.mkdir(parents=True, exist_ok=True)
    if not args.show:
        matplotlib.use("Agg")

    base = run_base_case(args.question, out, args.show)
    if args.scenarios and base is not None:
        if args.question == "Q2_linear":
            run_linear_sweep(args.question, out)
        elif args.question == "Q2_quadratic":
            run_quadratic_sweep(args.question, out)
        else:
            run_scenarios(args.question, out)
    print(f"\nOutputs written to {out}")


if __name__ == "__main__":
    main()

def run_quadratic_sweep(question: str, out: Path) -> dict[str, Results]:
    """Sweep the quadratic disutility parameter c^Q for question 2c"""
    runs: dict[str, Results] = {}

    c_values = np.linspace(0.0, 0.5, 7)

    for c in c_values:
        name = f"cQ_{c:.3f}"

        data = load_question(question)
        data.quadratic_disutility = c

        model = QuadraticDisutilityModel(data).build()
        results = model.solve()

        results.save(out, tag=name)
        runs[name] = results
        print(f"{name:>14}: cost {results.objective:8.2f} DKK | total load {results.hourly['load'].sum():5.1f} kWh")

    plot_scenario_comparison(runs, "objective", save_to=out / "quadratic_sweep_cost.png")
    return runs