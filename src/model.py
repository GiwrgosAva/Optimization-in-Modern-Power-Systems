"""Optimisation model of a single flexible consumer, implemented with gurobipy.

The class below separates the three steps you will repeat for every question:

    model = FlexibleConsumerModel(data)   # 1. hand over the input data
    model.build()                         # 2. declare variables, objective, constraints
    results = model.solve()               # 3. optimise and collect primal AND dual values

``build()`` is the only method you need to complete for Question 1; the other questions
are variations of it (a different objective, an extra constraint). Copy this file or
subclass ``FlexibleConsumerModel`` and override ``build()`` to keep one model per question.

Two conventions make the dual variables easy to read out afterwards:

* Every constraint family is stored in ``self.con`` under a descriptive name, e.g.
  ``self.con["balance"] = self.m.addConstrs(...)``. ``solve()`` then returns the dual value
  (shadow price, Gurobi attribute ``Pi``) of every constraint in ``self.con`` automatically.
* Bounds that you want a dual for must be written as explicit constraints (``addConstr``),
  not as variable bounds (``lb=``/``ub=``). Gurobi reports the sensitivity of a variable
  bound in the reduced cost (``RC``), not in ``Pi``.
* Duals of quadratic constraints (``m.addQConstr``) are read from ``QCPi`` and require ``QCPDual = 1`` (set below).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

from .data_loader import InputData


@dataclass
class Results:
    """Primal and dual solution of one model run."""

    question: str
    status: str
    objective: float
    hourly: pd.DataFrame                   # one row per hour: variables, prices, hourly duals
    duals: dict[str, float] = field(default_factory=dict)   # duals of non-hourly constraints
    meta: dict = field(default_factory=dict)                 # anything else worth keeping (scenario name, ...)

    def save(self, folder: Path | str, tag: str = "") -> None:
        """Write ``hourly`` to CSV and the scalar values to a small text file."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        stem = f"{self.question}{'_' + tag if tag else ''}"
        self.hourly.to_csv(folder / f"{stem}_hourly.csv", index_label="hour")
        with open(folder / f"{stem}_summary.txt", "w", encoding="utf-8") as f:
            f.write(f"status    : {self.status}\nobjective : {self.objective:.4f} DKK\n")
            for k, v in self.duals.items():
                f.write(f"dual[{k}] : {v:.4f}\n")

    def __str__(self) -> str:
        cols = [c for c in self.hourly.columns if not c.startswith("dual_")]
        return (
            f"status: {self.status} | objective: {self.objective:.2f} DKK\n"
            f"daily totals (kWh): " + ", ".join(f"{c}={self.hourly[c].sum():.1f}" for c in cols if c in ("import", "export", "load", "pv"))
            + (f"\nduals: {self.duals}" if self.duals else "")
        )


def _dual(c) -> float:
    """Dual value of a linear (``Pi``) or quadratic (``QCPi``, requires QCPDual=1) constraint."""
    return c.QCPi if isinstance(c, gp.QConstr) else c.Pi


class FlexibleConsumerModel:
    """Consumption problem of one consumer over a 24-hour horizon (Question 1); extend it for Questions 2 and 3."""

    def __init__(self, data: InputData, name: str = "flexible_consumer", verbose: bool = False):
        self.data = data
        self.T = range(data.n_hours)
        self.m = gp.Model(name)
        self.m.Params.OutputFlag = 1 if verbose else 0
        self.m.Params.QCPDual = 1          # only relevant if you add a quadratic constraint (none is needed in Assignment 1)
        self.var: dict[str, gp.tupledict | gp.Var] = {}   # decision variables by name
        self.con: dict[str, gp.tupledict | gp.Constr] = {}  # constraints by name (duals read from here)

    # ------------------------------------------------------------------ 2. build
    def build(self) -> "FlexibleConsumerModel":
        """Declare decision variables, objective and constraints.

        TODO (Question 1): complete this method with the variables, objective and constraints
        of the problem you formulated in Question 1. Keep the naming pattern below so
        that ``solve()`` can return the primal and dual values automatically.
        """
        d, m, T = self.data, self.m, self.T

        # --- Decision variables --------------------------------------------------------

        self.var["import"] = m.addVars(
            T, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="import"
        )

        self.var["export"] = m.addVars(
            T, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="export"
        )

        self.var["load"] = m.addVars(
            T, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="load"
        )

        self.var["pv"] = m.addVars(
            T, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="pv"
        )


        p_imp = self.var["import"]
        p_exp = self.var["export"]
        l     = self.var["load"]
        g_pv  = self.var["pv"]

        # --- Objective ---------------------------------------------------------------
        if d.consumption_utility is not None:
            net_utility = gp.quicksum(
            d.consumption_utility * l[t]
            - d.pv_marginal_cost * g_pv[t]
            - (d.energy_price[t] + d.import_tariff) * p_imp[t]
            + (d.energy_price[t] - d.export_tariff) * p_exp[t]
            for t in T
        )

            m.setObjective(net_utility, GRB.MAXIMIZE)

        m.update()
        #return self

        # --- Constraints -------------------------------------------------------------
        # 1. Hourly Power Balance: Load + Export = PV + Import

        self.con["balance"] = m.addConstrs(
            (l[t] + p_exp[t] == g_pv[t] + p_imp[t] for t in T), name="balance"
        )

        # 2. PV availability Upper Bound: 0 <= g_PV <= PV_MAX

        self.con["pv_max"] = m.addConstrs(
            (g_pv[t] - d.pv_available[t] <= 0 for t in T),
            name="pv_max"
        )

        # 3. Load Capacity Upper Bound: 0 <= l <= LOAD_MAX

        self.con["load_max"] = m.addConstrs(
            (l[t] - d.load_max_kWh <= 0 for t in T),
            name="load_max"
        )

        # 4. Load Capacity Lower Bound: 0 <= l

        self.con["load_min"] = m.addConstrs(
            (d.load_min_kWh - l[t] <= 0 for t in T),
            name="load_min"
        )

        # 5. Non negativity constraints

        self.con["import_min"] = m.addConstrs(
            (p_imp[t] >= 0 for t in T),
            name="import_min"
        )

        self.con["export_min"] = m.addConstrs(
            (p_exp[t] >= 0 for t in T),
            name="export_min"
        )

        self.con["pv_min"] = m.addConstrs(
            (g_pv[t] >= 0 for t in T),
            name="pv_min"
        )

        m.update()
        return self


    # ------------------------------------------------------------------ 3. solve
    def solve(self) -> Results:
        """Optimise and return primal values, objective and dual values."""
        m = self.m
        m.update()
        if m.NumConstrs == 0 and m.NumQConstrs == 0:
            raise NotImplementedError(
                "The model has no constraints: complete FlexibleConsumerModel.build() in src/model.py first."
            )
        m.optimize()
        status = _status_name(m.Status)
        if m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Optimisation ended with status {status}. Check the model (m.computeIIS() helps for infeasibility).")
        return self._extract_results(status)

    # --------------------------------------------------------------- extraction
    def _extract_results(self, status: str) -> Results:
        d, T = self.data, list(self.T)
        hourly = pd.DataFrame(index=pd.Index(T, name="hour"))
        hourly["price"] = d.energy_price
        hourly["pv_available"] = d.pv_available
        if d.reference_load is not None:
            hourly["reference_load"] = d.reference_load

        # Primal values: every hourly variable family in self.var becomes a column
        for name, v in self.var.items():
            if isinstance(v, gp.tupledict):
                hourly[name] = [v[t].X for t in T]
        scalars = {name: v.X for name, v in self.var.items() if isinstance(v, gp.Var)}

        # Dual values: every constraint family in self.con becomes a 'dual_<name>' column or scalar
        duals: dict[str, float] = {}
        for name, c in self.con.items():
            try:
                if isinstance(c, gp.tupledict):
                    hourly[f"dual_{name}"] = [_dual(c[t]) for t in T]
                else:
                    duals[name] = _dual(c)
            except (AttributeError, gp.GurobiError):
                # No duals available (e.g. model with integer variables)
                pass

        return Results(
            question=d.question,
            status=status,
            objective=self.m.ObjVal,
            hourly=hourly,
            duals=duals,
            meta={"scalar_variables": scalars},
        )


_STATUS = {
    GRB.OPTIMAL: "OPTIMAL", GRB.INFEASIBLE: "INFEASIBLE", GRB.UNBOUNDED: "UNBOUNDED",
    GRB.INF_OR_UNBD: "INF_OR_UNBD", GRB.TIME_LIMIT: "TIME_LIMIT", GRB.SUBOPTIMAL: "SUBOPTIMAL",
    GRB.NUMERIC: "NUMERIC", GRB.INTERRUPTED: "INTERRUPTED",
}


def _status_name(code: int) -> str:
    return _STATUS.get(code, f"STATUS_{code}")

class LinearDisutilityModel(FlexibleConsumerModel):
    """Question 2.(b): Flexible Consumer with a linear disutility penalty."""

    def build(self) -> "LinearDisutilityModel":
        # 1. Inherit variables and physical constraints from Q1.
        super().build()

        d, m, T = self.data, self.m, self.T
        l = self.var["load"]
        p_imp = self.var["import"]
        p_exp = self.var["export"]
        g_pv = self.var["pv"]

        # 2. Add the continuous auxiliary variable Delta
        self.var["delta"] = m.addVars(
            T, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="delta"
        )
        delta = self.var["delta"]

        # 3. Add the two linear constraints for absolute deviation
        self.con["delta_pos"] = m.addConstrs(
            (delta[t] >= l[t] - d.reference_load[t] for t in T), name="delta_pos"
        )

        self.con["delta_neg"] = m.addConstrs(
            (delta[t] >= d.reference_load[t] - l[t] for t in T), name="delta_neg"
        )

        # 4. Update the objective function
        net_utility = gp.quicksum(
            - d.pv_marginal_cost * g_pv[t]
            - (d.energy_price[t] + d.import_tariff) * p_imp[t]
            + (d.energy_price[t] - d.export_tariff) * p_exp[t]
            - d.linear_disutility * delta[t]
            for t in T
        )

        m.setObjective(net_utility, GRB.MAXIMIZE)
        m.update()
        return self

class QuadraticDisutilityModel(FlexibleConsumerModel):
    """Question 2.(c): Flexible Consumer with a quadratic disutility penalty."""

    def build(self) -> "QuadraticDisutilityModel":
        # 1. Inherit variables and physical constraints from Q1.
        super().build()

        d, m, T = self.data, self.m, self.T
        l = self.var["load"]
        p_imp = self.var["import"]
        p_exp = self.var["export"]
        g_pv = self.var["pv"]

        # 2. Update the objective function
        net_utility = gp.quicksum(
            - d.pv_marginal_cost * g_pv[t]
            - (d.energy_price[t] + d.import_tariff) * p_imp[t]
            + (d.energy_price[t] - d.export_tariff) * p_exp[t]
            - d.quadratic_disutility * (l[t] - d.reference_load[t]) * (l[t] - d.reference_load[t])
            for t in T
        )

        m.setObjective(net_utility, GRB.MAXIMIZE)
        m.update()
        return self

