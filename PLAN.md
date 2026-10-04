# Plan: OR-Tools Python Examples

Status: all 20 examples (16 core plus 4 extras) are done. Decisions are recorded in [Decisions](#decisions).

## Goal

Build a public GitHub repo (`christianmerkwirth/ortools-examples`) with clear,
tested examples that show how to solve real optimization problems with
Google OR-Tools in Python.

Each example must:

- Solve a problem that a reader can relate to (a factory, a delivery fleet,
  a hospital roster).
- Show one OR-Tools solver or modeling technique well.
- Explain the math model in plain words and in formulas.
- Have comments that explain *why*, not *what*.
- Have tests that prove the solution is feasible and good.

## Audience

Engineers and students who know Python but are new to OR-Tools or to
optimization. Each example stands alone. A reader can open one folder and
learn from it without reading the others.

## Solver coverage

OR-Tools has several solvers. The examples cover all main ones:

| Solver / API                   | Problem class                         |
| ------------------------------ | ------------------------------------- |
| MathOpt (GLOP, PDLP)           | Linear programming (LP)               |
| MathOpt (SCIP, HiGHS, CP-SAT)  | Mixed-integer programming (MIP)       |
| CP-SAT                         | Constraint programming, scheduling    |
| Routing library                | TSP, vehicle routing                  |
| Graph solvers                  | Max flow, min-cost flow, assignment   |
| Knapsack solver                | Knapsack                              |

## Examples

Order goes from easy to hard. Numbers are folder prefixes.

| #  | Example                          | Solver                 | Key technique                                        |
| -- | -------------------------------- | ---------------------- | ---------------------------------------------------- |
| 01 | Production planning              | MathOpt + GLOP         | LP basics, dual values (shadow prices), sensitivity  |
| 02 | Diet / feed blending             | MathOpt + GLOP         | LP with bounds, infeasibility diagnosis              |
| 03 | Knapsack & multiple knapsack     | Knapsack solver, CP-SAT| Same problem, two solvers, compare results           |
| 04 | Assignment (workers to tasks)    | `linear_sum_assignment`, CP-SAT | Specialized vs general solver               |
| 05 | Supply chain network flow        | Min-cost flow, max flow| Graph models, bottleneck (min cut) analysis          |
| 06 | Facility location                | MathOpt + SCIP/HiGHS   | MIP, big-M vs strong formulation, gap and bounds     |
| 07 | Sudoku & N-Queens                | CP-SAT                 | Pure constraint satisfaction, all-different, enumerate all solutions |
| 08 | Graph coloring / exam timetabling| CP-SAT                 | Symmetry breaking, minimize number of colors         |
| 09 | Employee shift scheduling        | CP-SAT                 | Boolean models, soft constraints, fairness           |
| 10 | Job shop scheduling              | CP-SAT                 | Interval variables, `NoOverlap`, makespan, Gantt chart |
| 11 | Project scheduling (RCPSP)       | CP-SAT                 | `Cumulative` resources, precedence                   |
| 12 | Traveling salesperson (TSP)      | Routing library        | First-solution strategy, guided local search         |
| 13 | Vehicle routing (CVRPTW)         | Routing library        | Capacity and time-window dimensions, drop penalties  |
| 14 | Bin packing & 2D packing         | CP-SAT                 | `NoOverlap2D`, lower bounds                          |
| 15 | Cutting stock                    | MathOpt (LP) + CP-SAT  | Column generation, LP relaxation + integer rounding  |
| 16 | Sports league scheduling         | CP-SAT                 | Round-robin, home/away patterns, large models        |

Optional extras, if time allows:

- 17 Portfolio selection with cardinality limits (MIP, linearized risk).
- 18 Unit commitment for power plants (MIP, time-coupled constraints).
- 19 Pickup and delivery routing (routing library, paired visits).
- 20 Planning under uncertain demand (two-stage stochastic MIP, VSS, EVPI).

## Folder layout

```
.
├── README.md                 # Overview, install, table of examples
├── PLAN.md                   # This document
├── LICENSE                   # Apache-2.0
├── pyproject.toml            # Dependencies and tool config (uv)
├── uv.lock
├── .github/workflows/ci.yml  # Lint, tests, and every example end to end
├── examples/
│   ├── common/               # Shared helpers: text tables, plot style
│   ├── ex01_production_planning/
│   │   ├── README.md         # Story, math model, how to run, results
│   │   ├── data.py           # Instance data (dataclasses)
│   │   ├── model.py          # Build and solve the model (importable)
│   │   ├── check.py          # Independent solution checker, no OR-Tools
│   │   ├── main.py           # CLI entry point: solve, verify, report, plot
│   │   └── figures/          # PNGs used in the README
│   └── ...
└── tests/
    ├── test_ex01_production_planning.py
    └── ...
```

Folder names use the `exNN_` prefix. It keeps the learning order on
GitHub, and it makes each folder a valid Python package name (a name
cannot start with a digit).

Design rules for the code:

- **Split model from data.** `model.py` takes a data object and returns a
  result object. This makes tests easy and shows good practice.
- **Typed data.** Use `dataclasses` for instances and results.
- **No global state.** Every example runs with `python -m examples.exNN_x.main`.
- **Deterministic.** Fix random seeds. Set solver time limits and worker
  counts where results can vary.
- **Small default instance, optional large one.** The default solves in a
  few seconds. A flag can generate a bigger instance to show scaling.
- **Readable first.** Prefer clear code over clever code. Keep each
  `model.py` under ~200 lines where possible.

## Documentation per example

Each `README.md` has the same sections:

1. **The problem** – a short story with a concrete case.
2. **The model** – decision variables, constraints, objective, in words and
   in LaTeX math (GitHub renders it).
3. **The OR-Tools code** – a walk-through of the key lines.
4. **Run it** – the command and sample output.
5. **Results** – what the solution means; a plot where it helps (Gantt
   chart, route map, packing layout).
6. **Try this** – ideas to change the model and learn more.
7. **References** – links to OR-Tools docs and papers.

## Verification strategy

"Good solutions" must be proven, not claimed. Each example has tests on
three levels:

1. **Feasibility check.** An independent checker in `check.py` (pure
   Python, no OR-Tools) verifies every constraint of the returned
   solution. This catches modeling bugs. `main.py` runs it on every run.
2. **Optimality check.** On small instances, compare the objective with a
   known value:
   - a brute-force enumeration (knapsack, assignment, TSP with ≤ 8 cities),
   - a second solver or second formulation (knapsack solver vs CP-SAT,
     `linear_sum_assignment` vs MIP),
   - a published optimum (classic benchmarks such as `ft06` job shop = 55,
     Solomon VRPTW instances, Sudoku with a unique solution).
3. **Quality check.** For heuristics (routing, large CP-SAT runs), check
   the solver status, and assert that the optimality gap or the distance to
   the best known value is under a set limit.

CI runs `ruff` (lint and format) and `pytest` on Linux with Python 3.11–3.13.

## Tooling

- Python ≥ 3.11.
- `uv` for environments and the lock file.
- Dependencies: `ortools`, `numpy`, `matplotlib` (plots only), `pytest`,
  `ruff`.
- Pin the OR-Tools minor version in `pyproject.toml`. Note the tested
  version in the main README.

## Work phases

1. **Scaffold** – `git init`, `pyproject.toml`, CI, README skeleton,
   `common/` helpers, one test template.
2. **Example 01** – build it fully as the reference. Review style and depth
   with the owner before going on.
3. **Examples 02–16** – build in order. Each one is done only when its
   tests pass and its README is complete.
4. **Review pass** – read all examples end to end for consistent style,
   naming, and docs. Run all tests and all `main.py` scripts.
5. **Publish** – the owner creates the GitHub repo and pushes. (I do not
   push without explicit approval.)

## Decisions

| Topic         | Decision                                                      |
| ------------- | ------------------------------------------------------------- |
| LP/MIP API    | MathOpt                                                       |
| Format        | Python scripts and a README per example; no notebooks         |
| Plots         | Yes, matplotlib; PNGs saved and shown in each README          |
| License       | Apache-2.0                                                    |
| Scope         | 16 core examples first; extras (17–19) later                  |
| Repo name     | `christianmerkwirth/ortools-examples`                         |
