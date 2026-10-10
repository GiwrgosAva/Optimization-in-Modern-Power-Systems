"""Derive sensitivity scenarios from a base ``InputData`` without touching the JSON files.

Each helper returns a *new* ``InputData`` (the base data is never modified), so scenarios can
be chained::

    from src.data_loader import load_question
    from src.scenarios import scale_prices, set_tariffs, scale_pv

    base = load_question("Q1_caseA")
    high_spread = scale_prices(base, factor=2.0, keep_mean=True)      # same mean, doubled spread
    no_export_tariff = set_tariffs(high_spread, export_tariff=0.0)

Add your own helpers here (e.g. shifting the price peak towards the PV peak, changing the
minimum daily energy or the reference profile) and document them in the README.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .data_loader import InputData, load_question
from .model import LinearDisutilityModel, QuadraticDisutilityModel, Results


def scale_prices(data: InputData, factor: float, keep_mean: bool = False) -> InputData:
    """Multiply hourly prices by ``factor``. With ``keep_mean=True`` only the spread around the
    mean is scaled, which isolates the effect of price variability from the effect of price level."""
    p = data.energy_price
    new_p = p.mean() + factor * (p - p.mean()) if keep_mean else factor * p
    return replace(data, energy_price=np.clip(new_p, 0.0, None))   # prices stay non-negative


def shift_prices(data: InputData, delta: float) -> InputData:
    """Add ``delta`` DKK/kWh to every hourly price (changes the mean, keeps the spread)."""
    return replace(data, energy_price=np.clip(data.energy_price + delta, 0.0, None))


def roll_prices(data: InputData, hours: int) -> InputData:
    """Shift the price profile in time by ``hours`` (positive = later). Useful to study the
    temporal correlation between the price peak and the PV peak."""
    return replace(data, energy_price=np.roll(data.energy_price, hours))


def set_tariffs(data: InputData, import_tariff: float | None = None, export_tariff: float | None = None) -> InputData:
    """Override the import and/or export grid tariff (DKK/kWh)."""
    return replace(
        data,
        import_tariff=data.import_tariff if import_tariff is None else import_tariff,
        export_tariff=data.export_tariff if export_tariff is None else export_tariff,
    )


def scale_pv(data: InputData, factor: float) -> InputData:
    """Scale the available PV production (e.g. 0.5 for a cloudy day, 0.0 for no PV)."""
    profile = np.clip(data.pv_profile * factor, 0.0, 1.0)
    return replace(data, pv_profile=profile, pv_available=data.pv_max_kW * profile)


def set_load_preferences(
    data: InputData,
    load_max_kWh: float | None = None,
    load_min_kWh: float | None = None,
    min_daily_energy_kWh: float | None = None,
) -> InputData:
    """Override the hourly load bounds and/or the minimum daily energy requirement."""
    return replace(
        data,
        load_max_kWh=data.load_max_kWh if load_max_kWh is None else load_max_kWh,
        load_min_kWh=data.load_min_kWh if load_min_kWh is None else load_min_kWh,
        min_daily_energy_kWh=data.min_daily_energy_kWh if min_daily_energy_kWh is None else min_daily_energy_kWh,
    )


# ----------------------------------------------------------------------------------------------
# Question 2.b.iv / 2.c.iv - sweeps over the disutility coefficients
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



def solve_linear(question: str, cL: float):
    """Solve the linear-disutility model of ``question`` with disutility coefficient ``cL``."""
    data = load_question(question)
    data.linear_disutility = float(cL)
    return LinearDisutilityModel(data).build().solve(), data


def quadratic_thresholds(data) -> pd.DataFrame:
    """Hourly values of c^Q at which the load of the quadratic model changes regime (hypothesis of Question 2.c.ii)."""
    p_imp = data.energy_price + data.import_tariff
    p_exp = data.energy_price - data.export_tariff
    ref, pv, lmin = data.reference_load, data.pv_available, data.load_min_kWh
    value_pv = np.maximum(p_exp, data.pv_marginal_cost)                # value of a kWh of own PV energy
    value_pv = np.where(data.pv_marginal_cost < p_imp, value_pv, p_imp)
    above_min = ref > lmin + 1e-9                                      # reference above L_min
    above_pv = ref > np.maximum(pv, lmin) + 1e-9                       # reference above the available PV
    with np.errstate(divide="ignore", invalid="ignore"):
        leave_lmin = np.where(above_min, value_pv / (2 * (ref - lmin)), np.nan)    # load leaves L_min
        reach_pv = np.where(above_pv, value_pv / (2 * (ref - pv)), np.nan)         # load reaches the PV level
        leave_pv = np.where(above_pv, p_imp / (2 * (ref - pv)), np.nan)            # load leaves the PV level
    return pd.DataFrame({"ref": ref, "pv_available": pv, "leave_Lmin": leave_lmin,
                         "reach_PV_level": reach_pv, "leave_PV_level": leave_pv})



def quadratic_sweep_metrics(results: Results, data, cQ: float) -> dict:
    """Quantities requested in Question 2.(c).iv for one run of the quadratic model."""
    h, tol = results.hourly, 1e-6
    p_imp = h["price"] + data.import_tariff
    p_exp = h["price"] - data.export_tariff
    deviation = (h["load"] - h["reference_load"]).abs()
    procurement = (p_imp * h["import"] - p_exp * h["export"] + data.pv_marginal_cost * h["pv"]).sum()
    disutility = cQ * (deviation ** 2).sum()
    at_bound = (h["load"] <= data.load_min_kWh + tol) | (h["load"] >= data.load_max_kWh - tol)
    on_pv_level = ((h["load"] - h["pv_available"]).abs() <= tol) & ~at_bound
    return {
        "cQ": cQ,
        "procurement_cost": procurement,                      # DKK: grid purchases - export revenue + PV production
        "disutility": disutility,                             # DKK
        "total_cost": procurement + disutility,               # DKK = - objective
        "energy_consumed": h["load"].sum(),                   # kWh
        "total_abs_deviation": deviation.sum(),               # kWh
        "hours_bound_binding": int(((deviation > tol) & at_bound).sum()),     # deviation stopped by L_min / L_max
        "hours_on_pv_level": int(((deviation > tol) & on_pv_level).sum()),    # load held at the available PV
        "hours_deviating": int((deviation > tol).sum()),
        "import_kWh": h["import"].sum(),
        "export_kWh": h["export"].sum(),
    }



def solve_quadratic(question: str, cQ: float):
    """Solve the quadratic-disutility model of ``question`` with disutility coefficient ``cQ``."""
    data = load_question(question)
    data.quadratic_disutility = float(cQ)
    return QuadraticDisutilityModel(data).build().solve(), data