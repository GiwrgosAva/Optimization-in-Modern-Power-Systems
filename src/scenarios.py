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
from .model import (BatteryModel, LinearDisutilityModel, MinEnergyModel, QuadraticDisutilityModel,
                    TemporalDisutilityModel, Results)

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

# ----------------------------------------------------------------------------------------------
# Question 2.e.iv - effect of the temporal structure of the prices
# ----------------------------------------------------------------------------------------------
def square_wave_prices(mean: float, amplitude: float, block_hours: int, start_high: bool = True, n_hours: int = 24) -> np.ndarray:
    """Hourly prices alternating between ``mean + amplitude`` and ``mean - amplitude`` in blocks of ``block_hours`` hours.

    All block lengths contain the same set of prices (12 high and 12 low hours): only their order changes.
    """
    t = np.arange(n_hours)
    high = ((t // block_hours) % 2 == 0) == start_high
    return np.where(high, mean + amplitude, mean - amplitude)


def solve_model(model: str, data: InputData):
    """Solve 'linear', 'quadratic' or 'temporal' for the given data. Returns (results, disutility in DKK)."""
    ref = data.reference_load
    if model == "linear":
        res = LinearDisutilityModel(data).build().solve()
        return res, data.linear_disutility * float(np.abs(res.hourly["load"] - ref).sum())
    if model == "quadratic":
        res = QuadraticDisutilityModel(data).build().solve()
        return res, data.quadratic_disutility * float(((res.hourly["load"] - ref) ** 2).sum())
    if model == "temporal":
        m = TemporalDisutilityModel(data).build()
        res = m.solve()
        return res, m.cost * float((res.hourly["state"] ** 2).sum())
    raise ValueError(f"unknown model {model!r}")


def price_structure_metrics(results: Results, data: InputData, disutility: float) -> dict:
    """Daily surplus and deviations of one run.

    surplus = (cost of consuming exactly the reference) - (procurement cost) - (disutility)
            = value of flexibility, in DKK/day. Consuming the reference has zero disutility in every model,
              so this number is measured against the same baseline for all models.
    """
    h = results.hourly
    p_imp = h["price"] + data.import_tariff
    p_exp = h["price"] - data.export_tariff
    procurement = (p_imp * h["import"] - p_exp * h["export"] + data.pv_marginal_cost * h["pv"]).sum()
    baseline = (p_imp * h["reference_load"]).sum()          # no PV in this experiment: the reference is bought from the grid
    deviation = h["load"] - h["reference_load"]
    return {
        "surplus": baseline - procurement - disutility,
        "procurement_cost": procurement,
        "disutility": disutility,
        "total_abs_deviation": float(deviation.abs().sum()),
        "energy_consumed": float(h["load"].sum()),
        "max_discomfort_state": float(h["state"].abs().max()) if "state" in h else float("nan"),
    }

def solve_min_energy(question: str, min_energy: float | None = None, cQ: float | None = None,
                     load_max: float | None = None):
    """Solve the model of Question 3 and, on the same data, the model without the requirement (Question 2.c).

    ``min_energy`` (kWh), ``cQ`` (DKK/kWh^2) and ``load_max`` (kWh/h) override the values of the data files.
    Returns (results with requirement, results without requirement, data, E^min used).
    """
    data = load_question(question)
    if cQ is not None:
        data.quadratic_disutility = float(cQ)
    if load_max is not None:
        data.load_max_kWh = float(load_max)
    model = MinEnergyModel(data, min_energy=min_energy)
    with_req = model.build().solve()
    free = QuadraticDisutilityModel(data).build().solve()
    return with_req, free, data, model.min_energy

def min_energy_metrics(with_req: Results, free: Results, data: InputData, e_min: float) -> dict:
    """Quantities compared in Question 3.(f) for one run (costs in DKK/day, energies in kWh)."""
    h, h0, tol = with_req.hourly, free.hourly, 1e-6
    total = -with_req.objective
    procurement = with_req.meta["procurement_cost"]
    return {
        "E_min": e_min,
        "cQ": data.quadratic_disutility,
        "L_max": data.load_max_kWh,
        "energy_free": h0["load"].sum(),
        "energy": h["load"].sum(),
        "procurement_cost": procurement,
        "disutility": total - procurement,
        "total_cost": total,
        "requirement_cost": total + free.objective,
        "mu": with_req.duals["mu"],
        "hours_above_ref": int((h["load"] > h["reference_load"] + tol).sum()),
        "hours_at_Lmax": int((h["load"] >= data.load_max_kWh - tol).sum()),
        "hours_at_Lmin": int((h["load"] <= data.load_min_kWh + tol).sum()),
        "max_increase": float((h["load"] - h0["load"]).max()),
    }

def set_battery(data: InputData, capacity: float | None = None, power: float | None = None,
                efficiency: float | None = None) -> InputData:
    """Change the battery capacity (kWh), the charging and discharging power (kW, both) and/or the charging and
    discharging efficiency (both). The initial state of charge keeps its share of the capacity."""
    new = {}
    if capacity is not None:
        new.update(battery_capacity_kWh=capacity,
                   battery_initial_soc_kWh=data.battery_initial_soc_kWh / data.battery_capacity_kWh * capacity)
    if power is not None:
        new.update(battery_max_charge_kW=power, battery_max_discharge_kW=power)
    if efficiency is not None:
        new.update(battery_charging_efficiency=efficiency, battery_discharging_efficiency=efficiency)
    return replace(data, **new)


def solve_battery(data: InputData):
    """Solve the model with the battery (Question 3.g) and, on the same data, without it (Question 3.d)."""
    with_battery = BatteryModel(data).build().solve()
    without = MinEnergyModel(data).build().solve()
    return with_battery, without


def battery_metrics(with_battery: Results, without: Results) -> dict:
    """Value of the battery = net utility with the battery - net utility without it (= reduction of the total cost, DKK/day)."""
    h = with_battery.hourly
    return {
        "cost_without": -without.objective,
        "cost_with": -with_battery.objective,
        "value": with_battery.objective - without.objective,
        "charged_kWh": h["charge"].sum(),
        "mu_without": without.duals["mu"],
        "mu_with": with_battery.duals["mu"],
        "max_overlap": float(np.minimum(h["charge"], h["discharge"]).max()),
    }