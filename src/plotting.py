"""Matplotlib figures for the input data and the optimisation results.

Every function returns the ``Figure`` and optionally saves it, so the same code works in a
script (``python main.py``) and in a notebook (``plot_schedule(results, data);``).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data_loader import InputData
from .model import Results
from .scenarios import linear_thresholds, quadratic_thresholds


def _finish(fig: plt.Figure, save_to: Path | str | None) -> plt.Figure:
    fig.tight_layout()
    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=150)
    return fig


def plot_inputs(data: InputData, save_to: Path | str | None = None) -> plt.Figure:
    """Hourly prices (with tariffs) and available PV / load preferences, side by side."""
    h = data.hours
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))

    ax1.step(h, data.energy_price, where="mid", label="energy price", color="k")
    ax1.step(h, data.energy_price + data.import_tariff, where="mid", ls="--", label="price + import tariff")
    ax1.step(h, data.energy_price - data.export_tariff, where="mid", ls=":", label="price - export tariff")
    ax1.set(xlabel="hour", ylabel="DKK/kWh", title="Electricity prices")
    ax1.legend(fontsize=8)

    ax2.fill_between(h, data.pv_available, step="mid", alpha=0.4, color="orange", label="PV available")
    ax2.axhline(data.load_max_kWh, color="C3", ls="--", label="max load")
    if data.load_min_kWh > 0:
        ax2.axhline(data.load_min_kWh, color="C3", ls=":", label="min load")
    if data.reference_load is not None:
        ax2.step(h, data.reference_load, where="mid", color="C0", label="reference load")
    ax2.set(xlabel="hour", ylabel="kWh/h", title="PV and load preferences")
    ax2.legend(fontsize=8)
    fig.suptitle(f"Input data - {data.question}", fontsize=11)
    return _finish(fig, save_to)


def plot_schedule(results: Results, data: InputData, save_to: Path | str | None = None) -> plt.Figure:
    """Optimal schedule: load, PV used, import/export, with prices on a second axis."""
    hr = results.hourly
    h = hr.index.to_numpy()
    fig, ax = plt.subplots(figsize=(11, 4.2))

    width = 0.8
    if "load" in hr:
        ax.bar(h, hr["load"], width, color="C0", alpha=0.7, label="load")
    if "pv" in hr:
        ax.bar(h, -hr["pv"], width, color="orange", alpha=0.7, label="PV used (negative = generation)")
    if "pv_available" in hr and "pv" in hr:
        ax.step(h, -hr["pv_available"], where="mid", color="orange", ls="--", lw=1, label="PV available")
    if "import" in hr and "export" in hr:
        ax.plot(h, hr["import"] - hr["export"], "k.-", label="net import (+) / export (-)")
    if "reference_load" in hr:
        ax.step(h, hr["reference_load"], where="mid", color="C0", ls=":", label="reference load")
    ax.axhline(0, color="grey", lw=0.8)
    if data.consumption_utility is not None:  # Question 1: the objective is a surplus
        value = f"net utility {results.objective:.2f} DKK"
    else:  # Question 2/3: objective = -(procurement + disutility)
        value = f"total cost {-results.objective:.2f} DKK"
    ax.set(xlabel="hour", ylabel="kWh/h", title=f"Optimal schedule - {results.question} ({value})")

    ax2 = ax.twinx()
    ax2.step(h, hr["price"], where="mid", color="C3", lw=1.2, label="energy price")
    ax2.set_ylabel("DKK/kWh", color="C3")

    lines, labels = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(lines + l2, labels + lb2, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    return _finish(fig, save_to)


def plot_duals(results: Results, data: InputData, save_to: Path | str | None = None) -> plt.Figure:
    """Hourly dual variables (all ``dual_*`` columns) against the price signals."""
    hr = results.hourly
    dual_cols = [c for c in hr.columns if c.startswith("dual_")]
    fig, ax = plt.subplots(figsize=(11, 4))
    h = hr.index.to_numpy()
    for c in dual_cols:
        ax.step(h, hr[c], where="mid", label=c.removeprefix("dual_"))
    ax.step(h, data.energy_price + data.import_tariff, where="mid", color="grey", ls="--", lw=1, label="price + import tariff")
    ax.step(h, data.energy_price - data.export_tariff, where="mid", color="grey", ls=":", lw=1, label="price - export tariff")
    ax.set(xlabel="hour", ylabel="DKK/kWh", title=f"Dual variables - {results.question}")
    ax.legend(fontsize=8, ncol=3)
    return _finish(fig, save_to)


def plot_scenario_comparison(
    runs: dict[str, Results], metric: str = "objective", save_to: Path | str | None = None
) -> plt.Figure:
    """Bar chart of one metric across scenarios. ``metric`` is ``"objective"`` or the name of an
    hourly column whose daily sum is compared (e.g. ``"import"``, ``"export"``, ``"load"``)."""
    names = list(runs)
    if metric == "objective":
        values = [r.objective for r in runs.values()]
        ylabel = "daily cost [DKK]"
    else:
        values = [r.hourly[metric].sum() for r in runs.values()]
        ylabel = f"daily {metric} [kWh]"
    fig, ax = plt.subplots(figsize=(max(5, 1.2 * len(names)), 3.8))
    ax.bar(names, values, color="C0")
    ax.set(ylabel=ylabel, title=f"Scenario comparison - {metric}")
    ax.tick_params(axis="x", rotation=20)
    return _finish(fig, save_to)


# ----------------------------------------------------------------------------------------------
# Figures for the sweeps over c^L and c^Q (Questions 2.b.iv and 2.c.iv)
# ----------------------------------------------------------------------------------------------
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



def _plot_sweep(table: pd.DataFrame, xcol: str, xlabel: str, lines, ref_energy: float, counts: dict,
                title: str, save_to=None, logx: bool = False) -> None:
    """Four panels (costs, energy, deviation, hour counts) against the sweep parameter, thresholds as dotted lines."""
    fig, ax = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
    x = table[xcol]
    ax[0, 0].plot(x, table.procurement_cost, label="procurement cost")
    ax[0, 0].plot(x, table.disutility, label="disutility")
    ax[0, 0].plot(x, table.total_cost, "k--", label="total (= -objective)")
    ax[0, 0].set_ylabel("DKK"); ax[0, 0].legend(fontsize=8)
    ax[0, 1].plot(x, table.energy_consumed)
    ax[0, 1].axhline(ref_energy, color="gray", ls="--", label="reference profile")
    ax[0, 1].set_ylabel("daily energy consumed [kWh]"); ax[0, 1].legend(fontsize=8)
    ax[1, 0].plot(x, table.total_abs_deviation)
    ax[1, 0].set_ylabel("total absolute deviation [kWh]")
    for col, label in counts.items():
        ax[1, 1].step(x, table[col], where="post", label=label)
    ax[1, 1].set_ylabel("number of hours"); ax[1, 1].legend(fontsize=8, loc="upper right")
    for a in ax.ravel():
        a.grid(alpha=0.3)
        if logx:
            a.set_xscale("log")
        for v in lines:
            a.axvline(v, color="gray", ls=":", lw=0.8, alpha=0.6)
    for a in ax[1]:
        a.set_xlabel(xlabel)
    fig.suptitle(title)
    fig.tight_layout()
    if save_to is not None:
        fig.savefig(save_to, dpi=150)
    plt.close(fig)



def plot_linear_sweep(table: pd.DataFrame, data, save_to=None) -> None:
    """Daily quantities versus c^L, with the hourly thresholds of the hypothesis as dotted vertical lines."""
    th = linear_thresholds(data)
    lines = sorted(set(np.round(th.loc[th.lower_matters, "lower_threshold"], 4))
                   | set(np.round(th.loc[th.upper_matters, "upper_threshold"], 4)))
    _plot_sweep(table, "cL", "c^L [DKK/kWh]", lines, data.reference_load.sum(),
                {"hours_bound_binding": "deviation stopped by a load bound", "hours_deviating": "hours with any deviation"},
                "Linear disutility: sweep over c^L (dotted lines: hypothesised thresholds)", save_to)


# ----------------------------------------------------------------------------------------------


def plot_quadratic_sweep(table: pd.DataFrame, data, save_to=None) -> None:
    """Daily quantities versus c^Q (log axis), with the hourly regime changes of the hypothesis as dotted lines."""
    th = quadratic_thresholds(data)
    values = np.round(th[["leave_Lmin", "reach_PV_level", "leave_PV_level"]].to_numpy().ravel(), 4)
    lines = sorted(set(v for v in values if not np.isnan(v)))
    _plot_sweep(table, "cQ", "c^Q [DKK/kWh^2]", lines, data.reference_load.sum(),
                {"hours_bound_binding": "load at L_min", "hours_on_pv_level": "load at the available PV",
                 "hours_deviating": "hours with any deviation"},
                "Quadratic disutility: sweep over c^Q (dotted lines: hypothesised regime changes)", save_to, logx=True)

# ----------------------------------------------------------------------------------------------
# Figure for Question 2.e.iv
# ----------------------------------------------------------------------------------------------
def plot_price_structure(table, examples, save_to=None) -> None:
    """Left: surplus and total deviation against the block length of the high prices, one line per model.
    Right: load of the time-coupled model for the fastest and the slowest price alternation.

    ``table`` has columns model, block_hours, surplus, total_abs_deviation (averaged over the two phases).
    ``examples`` is a list of (title, prices, results) for the right-hand panels.
    """
    styles = {"linear": ("tab:green", "o"), "quadratic": ("tab:red", "s"), "temporal": ("tab:blue", "^")}
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 2 + 0, width_ratios=[1, 1.2])
    ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    for model, (color, marker) in styles.items():
        t = table[table.model == model].sort_values("block_hours")
        ax1.plot(t.block_hours, t.surplus, marker=marker, color=color, label=model)
        ax2.plot(t.block_hours, t.total_abs_deviation, marker=marker, color=color, label=model)
    for ax, ylabel in ((ax1, "daily surplus [DKK]"), (ax2, "total absolute deviation [kWh]")):
        ax.set_xscale("log"); ax.set_xticks(sorted(table.block_hours.unique())); ax.set_xticklabels(sorted(table.block_hours.unique()))
        ax.set_xlabel("length of a block of high prices [h]"); ax.set_ylabel(ylabel); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    for k, (title, prices, res) in enumerate(examples[:2]):
        ax = fig.add_subplot(gs[k, 1])
        h = res.hourly
        ax.step(h.index, h["reference_load"], where="mid", color="tab:blue", ls=":", lw=2, label="reference load")
        ax.step(h.index, h["load"], where="mid", color="tab:blue", lw=2, label="load (time-coupled model)")
        ax.set_xlabel("hour"); ax.set_ylabel("kWh/h"); ax.set_title(title, fontsize=9); ax.grid(alpha=0.3)
        ax2b = ax.twinx(); ax2b.step(h.index, prices, where="mid", color="tab:red", lw=1.2); ax2b.set_ylabel("energy price [DKK/kWh]", color="tab:red")
        if k == 1:
            ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    if save_to is not None:
        fig.savefig(save_to, dpi=150)
    plt.close(fig)


def plot_load_comparison(results: Results, unconstrained: Results, data: InputData,
                         save_to: Path | str | None = None) -> plt.Figure:
    """Question 3.(e): load with the minimum-energy requirement, load of Question 2.(c) and the reference profile."""
    h, h0 = results.hourly, unconstrained.hourly
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.fill_between(h.index, 0, h["pv_available"], step="mid", color="orange", alpha=0.2, label="PV available")
    ax.step(h.index, h["reference_load"], where="mid", color="tab:blue", ls=":", lw=2, label="reference load")
    ax.step(h0.index, h0["load"], where="mid", color="tab:green", lw=2,
            label=f"Question 2.(c) ({h0['load'].sum():.1f} kWh/day)")
    ax.step(h.index, h["load"], where="mid", color="tab:red", lw=2,
            label=f"with $E^{{min}}$ = {data.min_daily_energy_kWh:.0f} kWh ({h['load'].sum():.1f} kWh/day)")
    ax.set(xlabel="hour", ylabel="kWh/h", title="Optimal load - Q3 base case")
    ax.legend(); ax.grid(alpha=0.3)
    return _finish(fig, save_to)