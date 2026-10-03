# OR-Tools examples

[![CI](https://github.com/christianmerkwirth/ortools-examples/actions/workflows/ci.yml/badge.svg)](https://github.com/christianmerkwirth/ortools-examples/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Tested, documented examples that show how to solve real optimization
problems with [Google OR-Tools](https://developers.google.com/optimization)
in Python.

Each example:

- tells a concrete story (a workshop, a delivery fleet, a hospital roster),
- explains the math model in words and formulas,
- uses clean, commented code that keeps the model apart from the data,
- comes with a plot of the result, and
- is **verified**: an independent checker confirms every constraint, and
  tests compare the result with a known optimum, a brute-force search, or
  a second solver.

## Examples

| #  | Example | Solver | You learn |
| -- | ------- | ------ | --------- |
| 01 | [Production planning](examples/ex01_production_planning/) | MathOpt + GLOP | LP basics, shadow prices, reduced costs, what-if analysis |
| 02 | [Feed blending](examples/ex02_feed_blending/) | MathOpt + GLOP | Ranged constraints, infeasibility diagnosis (IIS, elastic model), Farkas proofs |
| 03 | [Knapsack and multiple knapsack](examples/ex03_knapsack/) | Knapsack solver, CP-SAT | Same problem, two solvers, scaling, side rules, DP referee |
| 04 | [Assignment](examples/ex04_assignment/) | `linear_sum_assignment`, CP-SAT | Specialized vs general solvers, dual potentials as proof, pricing side rules |
| 05 | [Supply chain network flow](examples/ex05_network_flow/) | SimpleMaxFlow, SimpleMinCostFlow | Max flow, min cut, bottleneck upgrades, node potentials |
| 06 | [Facility location](examples/ex06_facility_location/) | MathOpt + HiGHS/SCIP | Binary variables, weak vs strong formulations, MIP gaps and bounds |
| 07 | [Sudoku and N-Queens](examples/ex07_sudoku_nqueens/) | CP-SAT | AllDifferent, uniqueness proofs, enumerating all solutions |
| 08 | [Exam timetabling](examples/ex08_exam_timetabling/) | CP-SAT | Graph coloring, two encodings, symmetry breaking, clique/DSATUR bounds |
| 09 | [Employee shift scheduling](examples/ex09_shift_scheduling/) | CP-SAT | Boolean grid models, penalty literals, min-max fairness, lexicographic objectives |
| 10 | [Job shop scheduling](examples/ex10_job_shop/) | CP-SAT | Interval variables, `NoOverlap`, makespan vs tardiness, critical path, Gantt charts |
| 11 | [Project scheduling (RCPSP)](examples/ex11_project_scheduling/) | CP-SAT | `add_cumulative`, critical path vs resource limits, bottleneck what-if |
| 12 | [Traveling salesperson](examples/ex12_tsp/) | Routing library, CP-SAT | Index manager, search strategies, guided local search, exact `add_circuit` baseline |
| 13 | [Vehicle routing with time windows](examples/ex13_vehicle_routing/) | Routing library, CP-SAT | Capacity and time dimensions, fixed and span costs, optional visits, exact referee |
| 14 | [Bin packing and 2D packing](examples/ex14_packing/) | CP-SAT | Symmetry breaking, L1/L2 bounds, `NoOverlap2D`, optional intervals for rotation |
| 15 | [Cutting stock](examples/ex15_cutting_stock/) | MathOpt (GLOP, HiGHS) + CP-SAT | Column generation, pricing knapsack, Farley bound, price and branch |
| 16 | [Sports league scheduling](examples/ex16_sports_scheduling/) | CP-SAT | Break theory, circle method, template + draw model, redundant bounds |
| 17 | [Portfolio selection](examples/ex17_portfolio/) | MathOpt (GLOP, HiGHS) | CVaR as an LP, semi-continuous weights, cardinality limits, efficient frontiers |

See [PLAN.md](PLAN.md) for the design rules behind the examples.

## Quick start

You need Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/christianmerkwirth/ortools-examples.git
cd ortools-examples
uv sync

# Run one example (add --plot to write figures).
uv run python -m examples.ex01_production_planning.main

# Run all tests.
uv run pytest
```

Without uv: `pip install ortools numpy matplotlib pytest`, then use
`python` in place of `uv run python`.

## Layout

```text
examples/
  common/                    shared helpers: text tables, plot style
  ex01_production_planning/
    README.md                the story, the model, the results
    data.py                  the problem instance
    model.py                 the OR-Tools model (start reading here)
    check.py                 independent solution checker, no OR-Tools
    main.py                  command-line entry point
    figures/                 plots used in the README
tests/
  test_ex01_production_planning.py
```

## Tested with

OR-Tools 9.15 on Python 3.11, 3.12, and 3.13 (Linux).

## License

[Apache-2.0](LICENSE), the same license as OR-Tools.
