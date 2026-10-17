"""Entry point: load one question's data, build and solve the model, save results and figures.

    python main.py                          # base case of Q1_caseA
    python main.py --question Q2_linear     # another case
    python main.py --scenarios              # also run the example sensitivity scenarios

Results (CSV, TXT, PNG) are written to ``results/<question>/``. Extend ``run_scenarios``
with your own scenarios, or add a new function per question, as your analysis grows.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from src.data_loader import load_question, list_questions
from src.model import (FlexibleConsumerModel, LinearDisutilityModel, QuadraticDisutilityModel,
                       TemporalDisutilityModel, MinEnergyModel, Results)
from src.plotting import (plot_duals, plot_inputs, plot_linear_sweep, plot_load_comparison, plot_load_vs_reference,
                          plot_price_structure, plot_quadratic_sweep, plot_scenario_comparison, plot_schedule)
from src.scenarios import (linear_sweep_metrics, linear_thresholds, quadratic_sweep_metrics, quadratic_thresholds,
                           price_structure_metrics, scale_prices, scale_pv, set_tariffs, solve_linear, solve_model,
                           solve_quadratic, square_wave_prices)

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def run_base_case(question: str, out: Path, show: bool) -> Results | None:
    data = load_question(question)
    print(data.summary(), "\n")
    plot_inputs(data, save_to=out / "inputs.png")

    if "Q2_linear" in question:
        model = LinearDisutilityModel(data).build()
    elif "Q2_quadratic" in question:
        model = QuadraticDisutilityModel(data).build()
    elif question == "Q3":
        model = MinEnergyModel(data).build()
    else:
        model = FlexibleConsumerModel(data).build()
    print("Model used:", type(model).__name__)
    try:
        results = model.solve()
    except NotImplementedError as e:
        print(f"[skipped] {e}")
        return None

    print(results, "\n")
    results.save(out)
    plot_schedule(results, data, save_to=out / "schedule.png")
    plot_duals(results, data, save_to=out / "duals.png")
    if question == "Q3":
        compare_with_unconstrained(results, data, out)
    if show:
        matplotlib.pyplot.show()
    return results


def compare_with_unconstrained(results: Results, data, out: Path) -> None:
    """Question 3.(e): same data without the minimum-energy requirement (= Question 2.(c)) against the base case."""
    free = QuadraticDisutilityModel(data).build().solve()
    rows = {}
    for name, r in (("without requirement (Q2.c)", free), ("with requirement (Q3)", results)):
        procurement = r.meta["procurement_cost"]
        total = -r.objective
        rows[name] = {"energy [kWh]": r.hourly["load"].sum(), "procurement [DKK]": procurement,
                      "disutility [DKK]": total - procurement, "total cost [DKK]": total,
                      "hours above reference": int((r.hourly["load"] > r.hourly["reference_load"] + 1e-6).sum())}
    table = pd.DataFrame(rows).T
    print(table.round(2).to_string(), "\n")
    table.to_csv(out / "Q3_comparison.csv")
    hourly = pd.DataFrame({"reference": results.hourly["reference_load"], "Q2c": free.hourly["load"],
                           "Q3": results.hourly["load"]})
    hourly["increase"] = hourly["Q3"] - hourly["Q2c"]
    hourly["marginal_price"] = results.hourly["dual_balance"]
    hourly.to_csv(out / "Q3_comparison_hourly.csv", index_label="hour")
    plot_load_comparison(results, free, data, save_to=out / "load_comparison.png")

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


# ----------------------------------------------------------------------------------------------
def run_linear_sweep(question: str, out: Path) -> pd.DataFrame:
    """Question 2.(b).iv: base case against the reference profile, then a sweep over c^L."""
    base_results, base = solve_linear(question, load_question(question).linear_disutility)
    plot_load_vs_reference(base_results, base, save_to=out / "linear_base_schedule.png",
                           title=f"{question}: optimal load vs reference (c^L = {base.linear_disutility})")

    th = linear_thresholds(base)
    print("Hours where a threshold matters (hypothesis of 2.b.ii):")
    print(th[th.lower_matters | th.upper_matters].round(2).to_string(), "\n")

    # Fine grid for the figure (offset so that no value coincides exactly with a threshold, where optima are not unique)
    grid = np.round(np.arange(0.055, 3.0, 0.01), 3)
    fine = pd.DataFrame([linear_sweep_metrics(*solve_linear(question, c), c) for c in grid])
    fine.to_csv(out / "linear_sweep_fine.csv", index=False)
    plot_linear_sweep(fine, base, save_to=out / "linear_sweep.png")

    # Table for the report: one value of c^L inside each interval between two thresholds
    report_values = [0.50, 1.40, 1.43, 1.47, 1.55, 1.65, 1.80, 1.95, 2.05, 2.50]
    table = pd.DataFrame([linear_sweep_metrics(*solve_linear(question, c), c) for c in report_values])
    table.to_csv(out / "linear_sweep_table.csv", index=False)
    print(table.round(2).to_string(index=False))
    return table


def run_quadratic_sweep(question: str, out: Path) -> pd.DataFrame:
    """Question 2.(c).iv: base case against the reference profile, then a sweep over c^Q."""
    base_results, base = solve_quadratic(question, load_question(question).quadratic_disutility)
    plot_load_vs_reference(base_results, base, save_to=out / "quadratic_base_schedule.png",
                           title=f"{question}: optimal load vs reference (c^Q = {base.quadratic_disutility})")

    th = quadratic_thresholds(base)
    print("Hourly values of c^Q at which the regime changes (hypothesis of 2.c.ii):")
    print(th[th[["leave_Lmin", "reach_PV_level", "leave_PV_level"]].notna().any(axis=1)].round(3).to_string(), "\n")

    # Fine, logarithmically spaced grid for the figure (the load approaches the reference only asymptotically)
    grid = np.round(np.geomspace(0.05, 20.0, 240), 5)
    fine = pd.DataFrame([quadratic_sweep_metrics(*solve_quadratic(question, c), c) for c in grid])
    fine.to_csv(out / "quadratic_sweep_fine.csv", index=False)
    plot_quadratic_sweep(fine, base, save_to=out / "quadratic_sweep.png")

    # Table for the report
    report_values = [0.05, 0.10, 0.20, 0.50, 1.00, 2.00, 5.00, 10.0, 20.0]
    table = pd.DataFrame([quadratic_sweep_metrics(*solve_quadratic(question, c), c) for c in report_values])
    table.to_csv(out / "quadratic_sweep_table.csv", index=False)
    print(table.round(2).to_string(index=False))
    return table


def run_temporal_base_case(out: Path, show: bool = False) -> Results:
    """Question 2.(e): base case of the time-coupled disutility model (cost parameters: see TemporalDisutilityModel).

    No data file is shipped for this question (see the README): the Q2_quadratic case is used and the
    parameters 'memory' and 'cost' are set in code (defaults of TemporalDisutilityModel: memory 0.7, cost c^Q (1-memory)^2).
    """
    data = load_question("Q2_quadratic")
    data.question = "Q2_temporal"                       # name used for the output files and figure titles
    model = TemporalDisutilityModel(data)               # memory = 0.7, cost = c^Q * (1 - 0.7)^2 = 0.09
    print(f"Temporal model: memory a = {model.memory}, cost c^T = {model.cost:.3f} DKK/kWh^2\n")
    results = model.build().solve()
    print(results, "\n")
    results.save(out)
    plot_inputs(data, save_to=out / "inputs.png")
    plot_schedule(results, data, save_to=out / "schedule.png")
    plot_duals(results, data, save_to=out / "duals.png")
    plot_load_vs_reference(results, data, save_to=out / "temporal_base_schedule.png",
                           title=f"Q2_temporal: optimal load vs reference (a = {model.memory}, c^T = {model.cost:.3f})")
    if show:
        matplotlib.pyplot.show()
    return results


def run_price_structure_experiment(out: Path) -> pd.DataFrame:
    """Question 2.(e).iv: same prices in a different order, for the linear, quadratic and time-coupled model.

    Design (all choices isolate the ORDER of the prices):
    * prices: square wave between mean +/- std of the real price profile (12 high + 12 low hours), with blocks of
      1, 2, 3, 4, 6 or 12 hours; every block length therefore has the same mean, the same spread and the same set of prices;
    * flat reference (same daily energy as the real one) and no PV: every hour is identical, so a separable model
      cannot react to the order; both phases of the wave (starting high / starting low) are solved and averaged;
    * parameters of the base cases of (b), (c) and (e).
    """
    base = load_question("Q2_quadratic")
    base = replace(base, linear_disutility=load_question("Q2_linear").linear_disutility)
    base = scale_pv(base, 0.0)
    ref = np.full(base.n_hours, base.reference_load.mean())
    mean, amp = base.energy_price.mean(), base.energy_price.std()
    rows, examples = [], {}
    for block in [1, 2, 3, 4, 6, 12]:
        for model in ["linear", "quadratic", "temporal"]:
            runs = []
            for start_high in (True, False):
                prices = square_wave_prices(mean, amp, block, start_high)
                data = replace(base, energy_price=prices, reference_load=ref)
                res, disut = solve_model(model, data)
                runs.append(price_structure_metrics(res, data, disut))
                if model == "temporal" and start_high and block in (1, 12):
                    examples[block] = (f"Time-coupled model, blocks of {block} h", prices, res)
            row = pd.DataFrame(runs).mean().to_dict()
            rows.append({"model": model, "block_hours": block, **row})
    table = pd.DataFrame(rows)
    table.to_csv(out / "price_structure_table.csv", index=False)
    plot_price_structure(table, [examples[1], examples[12]], save_to=out / "price_structure.png")
    wide = table.pivot(index="block_hours", columns="model", values=["surplus", "total_abs_deviation"]).round(2)
    print(f"prices: mean {mean:.2f} +/- {amp:.2f} DKK/kWh, flat reference {ref[0]:.2f} kWh/h, no PV")
    print(wide.to_string())
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--question", default="Q1_caseA", choices=list_questions(), help="data case to use")
    parser.add_argument("--scenarios", action="store_true", help="also run the example sensitivity scenarios")
    parser.add_argument("--temporal", action="store_true", help="Question 2.(e): base case of the time-coupled disutility model")
    parser.add_argument("--prices", action="store_true", help="Question 2.(e).iv: effect of the temporal structure of the prices")
    parser.add_argument("--show", action="store_true", help="open the figures in a window")
    args = parser.parse_args()

    if args.prices:                       # Question 2.(e).iv, independent of --question
        out = RESULTS_DIR / "Q2_price_structure"
        out.mkdir(parents=True, exist_ok=True)
        if not args.show:
            matplotlib.use("Agg")
        run_price_structure_experiment(out)
        print(f"\nOutputs written to {out}")
        return

    if args.temporal:                     # Question 2.(e), independent of --question
        out = RESULTS_DIR / "Q2_temporal"
        out.mkdir(parents=True, exist_ok=True)
        if not args.show:
            matplotlib.use("Agg")
        run_temporal_base_case(out, args.show)
        print(f"\nOutputs written to {out}")
        return

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