# 46750 - Assignment 1: Demand-Side Flexibility in Active Distribution Grids

Code of **Group G11** for Assignment 1 of *46750 - Optimization in Modern Power Systems* (DTU).
It solves the optimisation problems of the report for a flexible consumer with PV and a small battery:
Question 1 (hourly consumption decision), Question 2 (linear, quadratic and time-coupled disutility)
and Question 3 (minimum daily energy requirement and battery). All models are written with `gurobipy`.
Every number and figure of the report is produced by one command (Section 3). The input data in
`data/` are used as provided.

## 1. Setup

### 1.1 Python environment

Use one of the two options (both install the same packages).

**Option A - pip and venv**
```bash
python -m venv venv
source venv/bin/activate            # macOS / Linux
# venv\Scripts\activate.bat         # Windows cmd
# venv\Scripts\Activate.ps1         # Windows PowerShell
pip install -r requirements.txt
```

**Option B - conda**
```bash
conda env create -f environment.yaml
conda activate 46750-a1
```

### 1.2 Gurobi licence

The code uses [Gurobi](https://www.gurobi.com) through the `gurobipy` package. The package ships with a
restricted licence that is large enough for this assignment (models up to 2000 variables/constraints), so
`python main.py` works out of the box. For unrestricted use, request a free
[academic licence](https://www.gurobi.com/academia/academic-program-and-licenses/) with your DTU e-mail
and activate it with `grbgetkey <your-key>` (on the DTU network or VPN).

### 1.3 Check that everything works
```bash
python main.py
```
This solves the base case of `Q1_caseA`, prints a summary of the input data, the objective, the daily
totals and the dual values, and saves the figures and tables to `results/Q1_caseA/`.

## 2. Repository structure

```
main.py                  Entry point: load data -> build model -> solve -> save results, figures and experiments
src/
  data_loader.py         load_question("Q3_battery") -> InputData (all parameters, with units)
  model.py               One model class per question (table below): build() / solve() -> Results
  scenarios.py           Helpers that derive scenarios from a base InputData, and the metrics of each experiment
  plotting.py            Figures for inputs, schedules, duals, sweeps and comparisons
data/
  appliance_params.json  SHARED catalogue: every PV system, flexible load and battery - see Section 4
  bus_params.json        SHARED grid connection: prices (both days) and tariffs
  params_<case>.json     One small file per case: Q1_caseA, Q1_caseB, Q2_linear, Q2_quadratic, Q3, Q3_battery
results/                 Written by main.py (git-ignored): one folder per case, with CSV, TXT and PNG files
requirements.txt, environment.yaml, LICENSE, .gitignore
```

Model classes in `src/model.py` (each question extends the model of the previous one):

| Class | Question | Model |
|---|---|---|
| `FlexibleConsumerModel` | 1 | Utility of consumption, PV, import and export |
| `LinearDisutilityModel` | 2.(b) | Linear disutility of deviating from the reference profile |
| `QuadraticDisutilityModel` | 2.(c) | Quadratic disutility of deviating from the reference profile |
| `TemporalDisutilityModel` | 2.(e) | Time-coupled disutility (memory factor `a`, cost `c^T = c^Q (1-a)^2`) |
| `MinEnergyModel` | 3.(a)-(f) | Quadratic model with the minimum daily energy constraint, sum of loads >= `E_min` |
| `BatteryModel` | 3.(g) | `MinEnergyModel` with a battery: energy balance, state of charge, power limits, final state of charge equal to the initial one |

## 3. How to run the experiments

Run all commands from the repository root. Results are written to `results/<case>/`, and the tables of the
report are printed to the screen.

| Report | Command | Main outputs |
|---|---|---|
| Question 1, case A / case B | `python main.py --question Q1_caseA` / `Q1_caseB` | `inputs.png`, `schedule.png`, `duals.png`, `<case>_hourly.csv`, `<case>_summary.txt` |
| Question 2.(b), linear model, base case | `python main.py --question Q2_linear` | `inputs.png`, `schedule.png`, `duals.png`, `Q2_linear_hourly.csv`, `Q2_linear_summary.txt` |
| Question 2.(b), sweep over `c^L` | `python main.py --question Q2_linear --scenarios` | `linear_base_schedule.png`, `linear_sweep.png`, `linear_sweep_cost.png`, `linear_sweep_table.csv`, `linear_sweep_fine.csv`, one hourly CSV and summary per value of `c^L` |
| Question 2.(c), quadratic model, base case | `python main.py --question Q2_quadratic` | `inputs.png`, `schedule.png`, `duals.png`, `Q2_quadratic_hourly.csv`, `Q2_quadratic_summary.txt` |
| Question 2.(c), sweep over `c^Q` | `python main.py --question Q2_quadratic --scenarios` | `quadratic_base_schedule.png`, `quadratic_sweep.png`, `quadratic_sweep_cost.png`, `quadratic_sweep_table.csv`, `quadratic_sweep_fine.csv`, one hourly CSV and summary per value of `c^Q` |
| Question 2.(e), time-coupled model | `python main.py --temporal` | `results/Q2_temporal/`: `temporal_base_schedule.png`, `schedule.png`, `duals.png`, `Q2_temporal_hourly.csv`, `Q2_temporal_summary.txt` |
| Question 2.(e).iv, temporal structure of the prices | `python main.py --prices` | `results/Q2_price_structure/`: `price_structure.png`, `price_structure_table.csv` |
| Question 3.(d)-(e), minimum energy, base case | `python main.py --question Q3` | `load_comparison.png`, `Q3_comparison.csv`, `Q3_comparison_hourly.csv` |
| Question 3.(f), sweeps over `E_min`, `c^Q` and `L_max` | `python main.py --question Q3 --scenarios` | `sweep_Emin_costs.png`, `sweep_Emin_hours.png`, `sweep_cQ.png`, CSV files |
| Question 3.(g), battery, base case | `python main.py --question Q3_battery` | `battery_schedule.png`, `Q3_battery_comparison.csv`, `Q3_battery_comparison_hourly.csv` |
| Question 3.(g), battery sensitivity | `python main.py --question Q3_battery --scenarios` | `battery_sweeps_environment.png`, `battery_sweeps_battery.png`, `battery_sweep_*.csv` |

Add `--show` to open the figures in a window.

**Use it from a notebook or your own script** (run from the repository root):
```python
from src.data_loader import load_question
from src.model import BatteryModel
from src.plotting import plot_schedule

data = load_question("Q3_battery")
results = BatteryModel(data).build().solve()
print(results)                    # objective, daily totals, scalar duals
results.hourly                    # DataFrame: one row per hour with variables, prices and hourly duals
plot_schedule(results, data)
```

**Conventions used in the code**

* Every variable family is stored in `self.var[<name>]` and every constraint family in `self.con[<name>]`.
  `solve()` returns the hourly values of each variable family as a column of `results.hourly`, and the dual
  value of every constraint as a column `dual_<name>` (and in `results.duals` for single constraints).
  The families `import`, `export`, `load` and `pv` are used by the standard plots.
* Bounds with a dual of interest are written as explicit constraints, not as variable bounds, because Gurobi
  reports the sensitivity of a variable bound in the reduced cost (`RC`) and not in `Pi`.
* **Sign convention of the duals.** The duals are the Gurobi attribute `Pi`, the sensitivity of the objective
  to the right-hand side of each constraint. The models maximise the net utility. The multiplier `mu` of the
  minimum-energy constraint is stored with the opposite sign (`results.duals["mu"]`, in DKK/kWh), so it is
  positive and is the marginal cost of the requirement. The multiplier of the final state of charge
  (`soc_final`) is reported as given by Gurobi.
* Scenarios (prices, tariffs, PV, battery capacity, power and efficiency) are derived in code with the
  functions of `src/scenarios.py`, and the JSON files are never modified.

## 4. Input data

All input data lives in `data/`: two **shared** files plus one small **case** file per
question. The shared files form a catalogue of every appliance used anywhere in the assignment;
each `params_<case>.json` composes the consumer of one case by picking appliances from the
catalogue by ID. All time series have 24 hourly values (hour 0 to 23).

**Units.** Every field name carries its unit. Fields ending in `_kW` are power ratings (kW);
fields ending in `_kWh_per_hour` (or `_KWh_per_hour` - same unit despite the capital K) are hourly
energy bounds (kWh consumed within one hour); fields ending in `_kWh` are energy amounts;
`_DKK_per_kWh` are prices/costs and `_DKK_per_kWh2` is the quadratic disutility coefficient.
**Capacity factors and ratios** (`hourly_profile_ratio`, `reference_load_capacity_factor`,
`initial_state_of_charge_ratio`, and the `_next_day` profiles) are **dimensionless fractions in
[0, 1] of the corresponding maximum** - multiply by it to get physical values. With 1-hour periods,
a power rating in kW and an energy amount in kWh/h are numerically interchangeable; the loader
does every conversion for you and exposes physical quantities:
`pv_available = max_power_kW x hourly_profile_ratio` (kWh/h),
`reference_load = max_load_KWh_per_hour x reference_load_capacity_factor` (kWh/h), and
`battery_initial_soc_kWh = storage_capacity_kWh x initial_state_of_charge_ratio` (kWh).

| Question in the assignment | Case | Appliances (from the catalogue) |
|---|---|---|
| Question 1 - hourly consumption decision (utility u, PV cost c_PV), case A: c_PV < u | `Q1_caseA` | `PV_01_A` + `FL_01` (a load with `consumption_utility_DKK_per_kWh`; no reference profile, no daily requirement) |
| Question 1, case B: a more expensive PV, c_PV > u | `Q1_caseB` | `PV_01_B` + `FL_01` (`PV_01_B` differs from `PV_01_A` only in `marginal_cost_DKK_per_kWh`) |
| Question 2 - linear disutility of deviating from a reference profile | `Q2_linear` | `PV_01_A` + `FL_02_L` (reference profile, `linear_disutility_DKK_per_kWh`; same hourly bounds as Question 1; no utility) |
| Question 2 - quadratic disutility | `Q2_quadratic` | `PV_01_A` + `FL_02_Q` (as `FL_02_L` with `quadratic_disutility_DKK_per_kWh2` instead) |
| Question 3 - minimum daily energy requirement | `Q3` | `PV_01_A` + `FL_03` (as `FL_02_Q` plus `min_total_energy_per_day_kWh`, the E_min of the daily requirement) |
| Question 3.(g) - the same consumer with a small battery | `Q3_battery` | `PV_01_A` + `FL_03` + `BAT_01` |

No data file is shipped for Question 2.(e). It starts from the `Q2_quadratic` case, and the memory factor and
the cost parameter of the time-coupled model are set in code (defaults of `TemporalDisutilityModel`).

`params_<case>.json` - the case file
: `consumers[]`: `consumer_id`, `connection_bus` (a `bus_id` of `bus_params.json`), `list_appliances`
  (IDs of the consumer's DERs, loads and storages in `appliance_params.json`). `hyperparameters` is
  unused in Assignment 1 (`null`).

`appliance_params.json` - the shared catalogue, one list per appliance type (`null` if none)
: **DERs**: `DER_id`, `DER_type` (`"PV"`), `max_power_kW` (peak power, kW), `marginal_cost_DKK_per_kWh` (cost of every kWh produced), `hourly_profile_ratio` (dimensionless fraction of `max_power_kW` available each hour; available energy in kWh/h = `max_power_kW` x ratio), `hourly_profile_ratio_next_day` (next-day forecast, not used in the models)
: **loads**: `load_id`, `load_type`, `max_load_KWh_per_hour` and `min_load_KWh_per_hour` (hourly consumption bounds, kWh/h), `consumption_utility_DKK_per_kWh` (value of every kWh consumed, Question 1), `linear_disutility_DKK_per_kWh` and `quadratic_disutility_DKK_per_kWh2` (Questions 2-3), `min_total_energy_per_day_kWh` (the E_min of Question 3), `reference_load_capacity_factor` (dimensionless fraction of `max_load_KWh_per_hour`; preferred consumption in kWh/h = max load x factor), `reference_hourly_profile_ratio_next_day` (next-day forecast, not used in the models). Fields a load does not use are `null`. The hourly bounds are the same for every load: deviations from the reference are limited only by them.
: **storages**: `storage_id`, `storage_type` (`"battery"`), `storage_capacity_kWh`, `max_charging_power_kW`, `max_discharging_power_kW`, `charging_efficiency` and `discharging_efficiency` (fraction of the energy transferred that survives each conversion - a full round trip keeps eta_ch x eta_dis), `initial_state_of_charge_ratio` (fraction of `storage_capacity_kWh` in the battery at the start of the day). No final state of charge is given in the data: `BatteryModel` requires the battery to end the day at its initial level (constraint e_24 = E_0, see the report).
: **heat_pumps**: not used in Assignment 1 (`null`)

`bus_params.json` - grid and market conditions at the connection bus
: `bus_id`, `import_tariff_DKK_per_kWh`, `export_tariff_DKK_per_kWh`, `energy_price_DKK_per_kWh` (24 values), `energy_price_next_day_DKK_per_kWh` (next-day forecast, not used in the models), `max_import_kW`, `max_export_kW` (`null` = no grid limit in Assignment 1)

`load_question()` in `src/data_loader.py` shows exactly how each attribute is derived from these files.
The next-day forecasts are exposed as `energy_price_next_day`, `pv_available_next_day` and `reference_load_next_day`.