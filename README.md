# 2026 MCM — Lunar Transportation Modeling

Mathematical modeling code for the 2026 MCM, exploring transportation plans for a lunar colony using space elevators and rockets.

The repository contains scenario simulations, rocket cost modeling, multi-objective optimization, sensitivity analysis, and visualization scripts.

## Contents

- [src](src/): Models, simulations, optimization, and plotting scripts.
- [data](data/): Input spreadsheets and saved results.
- [images](images/): Generated figures.

## Main Scripts

| Script | Purpose |
| --- | --- |
| [fit_rocket_cost.py](src/fit_rocket_cost.py) | Rocket cost fitting. |
| [problem_2_simulation.py](src/problem_2_simulation.py) | Problem 2 simulation. |
| [problem_3.py](src/problem_3.py) | Problem 3 analysis. |
| [problem_4_model.py](src/problem_4_model.py) | Transportation model with cost, time, and environmental objectives. |
| [problem_4_solver.py](src/problem_4_solver.py) | NSGA-II optimization with capacity and demand constraints. |
| [problem_4_sensitivity.py](src/problem_4_sensitivity.py) | Sensitivity analysis. |

The scripts preserve the competition workflow, including alternative scenarios and plotting versions. The Problem 4 solver uses `data/harbor.xlsx` and saves optimization results under `data/`.

Dependencies vary by script. The repository's `requirements.txt` is currently empty; a complete reproducible environment has not yet been documented.
