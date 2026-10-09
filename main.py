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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loader import load_question, list_questions
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


# ----------------------------------------------------------------------------------------------
# Question 2.(b).iv - helpers for the sweep over the linear disutility c^L
# ----------------------------------------------------------------------------------------------
def linear_thresholds(data) -> pd.DataFrame:
    """Hourly values of c^L at which the load of the linear model jumps (hypothesis of Question 2.b.ii)."""
    p_imp = data.energy_price + data.import_tariff                    # effective import price
    p_exp = data.energy_price - data.export_tariff                    # effective export price
    ref, pv = data.reference_load, data.pv_available
    lower = np.maximum(p_exp, data.pv_marginal_cost)                  # below it: load = L_min
    lower = np.where(data.pv_marginal_cost < p_imp, lower, p_imp)     # PV never used if c_PV >= import price
    own_pv = np.maximum(data.load_min_kWh, np.minimum(pv, ref))       # level covered by own PV
    return pd.DataFrame({
        "p_imp": p_imp, "p_exp": p_exp, "ref": ref, "pv_available": pv,
        "lower_threshold": lower,                                      # max(p_exp, c_PV)
        "upper_threshold": p_imp,                                      # p_imp
        "lower_matters": own_pv > data.load_min_kWh + 1e-9,            # own-PV level differs from L_min
        "upper_matters": ref > own_pv + 1e-9,                          # reference differs from own-PV level
    })


def linear_sweep_metrics(results: Results, data, cL: float) -> dict:
    """Quantities requested in Question 2.(b).iv for one run of the linear model."""
    h, tol = results.hourly, 1e-6
    p_imp = h["price"] + data.import_tariff
    p_exp = h["price"] - data.export_tariff
    deviation = (h["load"] - h["reference_load"]).abs()
    procurement = (p_imp * h["import"] - p_exp * h["export"] + data.pv_marginal_cost * h["pv"]).sum()
    disutility = cL * deviation.sum()
    at_bound = (h["load"] <= data.load_min_kWh + tol) | (h["load"] >= data.load_max_kWh - tol)
    return {
        "cL": cL,
        "procurement_cost": procurement,                      # DKK: grid purchases - export revenue + PV production
        "disutility": disutility,                             # DKK
        "total_cost": procurement + disutility,               # DKK = - objective
        "energy_consumed": h["load"].sum(),                   # kWh
        "total_abs_deviation": deviation.sum(),               # kWh
        "hours_bound_binding": int(((deviation > tol) & at_bound).sum()),   # deviation stopped by L_min / L_max
        "hours_deviating": int((deviation > tol).sum()),
        "import_kWh": h["import"].sum(),
        "export_kWh": h["export"].sum(),
    }


def plot_linear_sweep(table: pd.DataFrame, data, save_to=None) -> None:
    """Daily quantities versus c^L, with the hourly thresholds of the hypothesis as dotted vertical lines."""
    th = linear_thresholds(data)
    lines = sorted(set(np.round(th.loc[th.lower_matters, "lower_threshold"], 4))
                   | set(np.round(th.loc[th.upper_matters, "upper_threshold"], 4)))
    fig, ax = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
    ax[0, 0].plot(table.cL, table.procurement_cost, label="procurement cost")
    ax[0, 0].plot(table.cL, table.disutility, label="disutility")
    ax[0, 0].plot(table.cL, table.total_cost, "k--", label="total (= -objective)")
    ax[0, 0].set_ylabel("DKK"); ax[0, 0].legend(fontsize=8)
    ax[0, 1].plot(table.cL, table.energy_consumed)
    ax[0, 1].axhline(data.reference_load.sum(), color="gray", ls="--", label="reference profile")
    ax[0, 1].set_ylabel("daily energy consumed [kWh]"); ax[0, 1].legend(fontsize=8)
    ax[1, 0].plot(table.cL, table.total_abs_deviation)
    ax[1, 0].set_ylabel("total absolute deviation [kWh]")
    ax[1, 1].step(table.cL, table.hours_bound_binding, where="post", label="deviation stopped by a load bound")
    ax[1, 1].step(table.cL, table.hours_deviating, where="post", label="hours with any deviation")
    ax[1, 1].set_ylabel("number of hours"); ax[1, 1].legend(fontsize=8, loc="upper right")
    for a in ax.ravel():
        a.grid(alpha=0.3)
        for x in lines:
            a.axvline(x, color="gray", ls=":", lw=0.8)
    for a in ax[1]:
        a.set_xlabel("c^L [DKK/kWh]")
    fig.suptitle("Linear disutility: sweep over c^L (dotted lines: hypothesised thresholds)")
    fig.tight_layout()
    if save_to is not None:
        fig.savefig(save_to, dpi=150)
    plt.close(fig)


def plot_load_vs_reference(results: Results, data, save_to=None, title="") -> None:
    """Optimal hourly load against the reference profile and the available PV."""
    h = results.hourly
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.step(h.index, h["reference_load"], where="mid", color="tab:blue", ls=":", lw=2, label="reference load")
    ax.step(h.index, h["load"], where="mid", color="tab:red", lw=2, label="optimal load")
    ax.fill_between(h.index, 0, h["pv_available"], step="mid", color="orange", alpha=0.25, label="PV available")
    ax.set_xlabel("hour"); ax.set_ylabel("kWh/h"); ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_to is not None:
        fig.savefig(save_to, dpi=150)
    plt.close(fig)


def _solve_linear(question: str, cL: float):
    data = load_question(question)
    data.linear_disutility = float(cL)
    return LinearDisutilityModel(data).build().solve(), data


def run_linear_sweep(question: str, out: Path) -> pd.DataFrame:
    """Question 2.(b).iv: base case against the reference profile, then a sweep over c^L."""
    base_results, base = _solve_linear(question, load_question(question).linear_disutility)
    plot_load_vs_reference(base_results, base, save_to=out / "linear_base_schedule.png",
                           title=f"{question}: optimal load vs reference (c^L = {base.linear_disutility})")

    th = linear_thresholds(base)
    print("Hours where a threshold matters (hypothesis of 2.b.ii):")
    print(th[th.lower_matters | th.upper_matters].round(2).to_string(), "\n")

    # Fine grid for the figure (offset so that no value coincides exactly with a threshold, where optima are not unique)
    grid = np.round(np.arange(0.055, 3.0, 0.01), 3)
    fine = pd.DataFrame([linear_sweep_metrics(*_solve_linear(question, c), c) for c in grid])
    fine.to_csv(out / "linear_sweep_fine.csv", index=False)
    plot_linear_sweep(fine, base, save_to=out / "linear_sweep.png")

    # Table for the report: one value of c^L inside each interval between two thresholds
    report_values = [0.50, 1.40, 1.43, 1.47, 1.55, 1.65, 1.80, 1.95, 2.05, 2.50]
    table = pd.DataFrame([linear_sweep_metrics(*_solve_linear(question, c), c) for c in report_values])
    table.to_csv(out / "linear_sweep_table.csv", index=False)
    print(table.round(2).to_string(index=False))
    return table


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