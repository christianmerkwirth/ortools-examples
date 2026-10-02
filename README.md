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
| 03 | Knapsack and multiple knapsack | Knapsack solver, CP-SAT | Same problem, two solvers |
| 04 | [Assignment](examples/ex04_assignment/) | `linear_sum_assignment`, CP-SAT | Specialized vs general solvers, dual potentials as proof, pricing side rules |
| 05 | [Supply chain network flow](examples/ex05_network_flow/) | SimpleMaxFlow, SimpleMinCostFlow | Max flow, min cut, bottleneck upgrades, node potentials |
| 06 | Facility location | MathOpt + SCIP/HiGHS | MIP, strong formulations, gaps and bounds |
| 07 | [Sudoku and N-Queens](examples/ex07_sudoku_nqueens/) | CP-SAT | AllDifferent, uniqueness proofs, enumerating all solutions |
| 08 | Graph coloring / exam timetabling | CP-SAT | Symmetry breaking |
| 09 | [Employee shift scheduling](examples/ex09_shift_scheduling/) | CP-SAT | Boolean grid models, penalty literals, min-max fairness, lexicographic objectives |
| 10 | [Job shop scheduling](examples/ex10_job_shop/) | CP-SAT | Interval variables, `NoOverlap`, makespan vs tardiness, critical path, Gantt charts |
| 11 | Project scheduling (RCPSP) | CP-SAT | `Cumulative` resources, precedences |
| 12 | [Traveling salesperson](examples/ex12_tsp/) | Routing library, CP-SAT | Index manager, search strategies, guided local search, exact `add_circuit` baseline |
| 13 | Vehicle routing with time windows | Routing library | Capacity and time dimensions |
| 14 | Bin packing and 2D packing | CP-SAT | `NoOverlap2D`, lower bounds |
| 15 | Cutting stock | MathOpt + CP-SAT | Column generation |
| 16 | Sports league scheduling | CP-SAT | Large combinatorial models |

Examples without a link are planned. See [PLAN.md](PLAN.md).

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
