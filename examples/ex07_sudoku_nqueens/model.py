"""Sudoku and N-Queens as constraint satisfaction problems, solved with CP-SAT.

Both puzzles have no objective. We only ask: is there an assignment that
breaks no rule? CP-SAT answers this question, and it can also:

* prove that a Sudoku has exactly one solution,
* enumerate every solution of a model (all 92 ways to place 8 queens), and
* help build new puzzles: remove clues as long as the solution stays unique.

The key tool is `add_all_different`: one constraint that says "these
variables all take different values". CP-SAT reasons about it as a whole,
which is much stronger than many separate `x != y` constraints.
"""

import random
from dataclasses import dataclass

from ortools.sat.python import cp_model

from .data import Grid

DIGITS = range(1, 10)


# ---------------------------------------------------------------- Sudoku ---


@dataclass
class SudokuModel:
    """A CP-SAT model plus its 9x9 grid of cell variables."""

    model: cp_model.CpModel
    cell: list[list[cp_model.IntVar]]


def build_sudoku(puzzle: Grid) -> SudokuModel:
    """Build the Sudoku model: one integer variable per cell, 1 to 9."""
    model = cp_model.CpModel()

    # A given digit becomes a variable with a one-value domain [v, v]. An
    # empty cell gets the domain [1, 9]. Domains are the simplest way to
    # state "this value is fixed".
    cell = [
        [
            model.new_int_var(v, v, f"cell {r},{c}")
            if v
            else model.new_int_var(1, 9, f"cell {r},{c}")
            for c, v in enumerate(row)
        ]
        for r, row in enumerate(puzzle)
    ]

    for i in range(9):
        model.add_all_different(cell[i])  # Row i.
        model.add_all_different(cell[r][i] for r in range(9))  # Column i.
        top, left = 3 * (i // 3), 3 * (i % 3)
        model.add_all_different(  # Box i, counted left to right, top to bottom.
            cell[top + dr][left + dc] for dr in range(3) for dc in range(3)
        )
    return SudokuModel(model, cell)


def solve_sudoku(puzzle: Grid, time_limit: float = 10.0) -> Grid | None:
    """Return one solution, or None if the puzzle has no solution."""
    return _solve_grid(build_sudoku(puzzle), time_limit)


def find_second_solution(
    puzzle: Grid, first: Grid, time_limit: float = 10.0
) -> Grid | None:
    """Return a solution that differs from `first`, or None if none exists.

    We add one constraint: at least one empty cell must hold a different
    digit than in `first`. If the solver proves this model infeasible,
    `first` is the only solution.
    """
    sm = build_sudoku(puzzle)
    differs = []
    for r in range(9):
        for c in range(9):
            if puzzle[r][c] == 0:
                # `d` true forces the cell away from its first value. The
                # constraint only holds when its enforcement literal is true.
                d = sm.model.new_bool_var(f"differs {r},{c}")
                sm.model.add(sm.cell[r][c] != first[r][c]).only_enforce_if(d)
                differs.append(d)
    sm.model.add_bool_or(differs)  # At least one cell must differ.
    return _solve_grid(sm, time_limit)


def is_unique(puzzle: Grid) -> bool:
    """Return True if the puzzle has exactly one solution."""
    first = solve_sudoku(puzzle)
    return first is not None and find_second_solution(puzzle, first) is None


def count_sudoku_solutions(puzzle: Grid, limit: int = 10_000) -> int:
    """Count the solutions by enumeration, and stop at `limit`."""
    sm = build_sudoku(puzzle)
    collector = _Collector([v for row in sm.cell for v in row], limit, keep=False)
    _enumerate(sm.model, collector)
    return collector.count


def make_puzzle(solution: Grid, seed: int = 0) -> Grid:
    """Turn a full grid into a puzzle with a unique solution.

    Visit the cells in random order. Blank each cell, and keep it blank
    only if the solution is still unique. The result is *minimal*: every
    remaining clue is needed.
    """
    order = [(r, c) for r in range(9) for c in range(9)]
    random.Random(seed).shuffle(order)
    grid = [list(row) for row in solution]
    for r, c in order:
        digit, grid[r][c] = grid[r][c], 0
        if not is_unique(_freeze(grid)):
            grid[r][c] = digit  # This clue is needed. Put it back.
    return _freeze(grid)


def _solve_grid(sm: SudokuModel, time_limit: float) -> Grid | None:
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 8
    status = solver.solve(sm.model)
    if status == cp_model.INFEASIBLE:
        return None
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"CP-SAT stopped early: {solver.status_name(status)}")
    return tuple(tuple(solver.value(v) for v in row) for row in sm.cell)


def _freeze(grid: list[list[int]]) -> Grid:
    return tuple(tuple(row) for row in grid)


# -------------------------------------------------------------- N-Queens ---


def build_queens(n: int) -> tuple[cp_model.CpModel, list[cp_model.IntVar]]:
    """Build the N-Queens model.

    queen[i] is the column of the queen in row i. One queen per row is thus
    built into the variables. The constraints handle columns and diagonals:

    * Columns: all queen[i] differ.
    * Diagonals going down-right: all queen[i] - i differ.
    * Diagonals going down-left: all queen[i] + i differ.
    """
    model = cp_model.CpModel()
    queen = [model.new_int_var(0, n - 1, f"queen in row {i}") for i in range(n)]
    model.add_all_different(queen)
    model.add_all_different(queen[i] - i for i in range(n))
    model.add_all_different(queen[i] + i for i in range(n))
    return model, queen


def solve_queens(n: int, time_limit: float = 10.0) -> tuple[int, ...] | None:
    """Return one placement (column per row), or None if none exists."""
    model, queen = build_queens(n)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 8
    status = solver.solve(model)
    if status == cp_model.INFEASIBLE:
        return None
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"CP-SAT stopped early: {solver.status_name(status)}")
    return tuple(solver.value(q) for q in queen)


def all_queens(n: int, limit: int | None = None) -> list[tuple[int, ...]]:
    """Return every placement of n queens (or the first `limit` of them)."""
    model, queen = build_queens(n)
    collector = _Collector(queen, limit, keep=True)
    _enumerate(model, collector)
    return collector.solutions


def count_queens(n: int) -> int:
    """Count every placement of n queens without storing them."""
    model, queen = build_queens(n)
    collector = _Collector(queen, None, keep=False)
    _enumerate(model, collector)
    return collector.count


# --------------------------------------------------------- enumeration -----


class _Collector(cp_model.CpSolverSolutionCallback):
    """Called by CP-SAT once for every solution it finds."""

    def __init__(self, variables, limit: int | None, keep: bool):
        super().__init__()
        self._variables = list(variables)
        self.limit = limit
        self._keep = keep
        self.count = 0
        self.solutions: list[tuple[int, ...]] = []

    def on_solution_callback(self) -> None:
        self.count += 1
        if self._keep:
            self.solutions.append(tuple(self.value(v) for v in self._variables))
        if self.limit is not None and self.count >= self.limit:
            self.stop_search()


def _enumerate(model: cp_model.CpModel, collector: _Collector) -> None:
    """Run CP-SAT so that it reports every solution exactly once.

    Two settings matter:

    * `enumerate_all_solutions` tells the solver not to stop at the first
      solution. It also turns off presolve steps that could remove
      solutions.
    * One worker only. With several workers, each one searches on its own
      and reports what it finds, so the same solution can come back more
      than once. (For 8 queens we saw 154 reports instead of 92.)
    """
    solver = cp_model.CpSolver()
    solver.parameters.enumerate_all_solutions = True
    solver.parameters.num_workers = 1
    status = solver.solve(model, collector)
    stopped_at_limit = (
        collector.limit is not None and collector.count >= collector.limit
    )
    if status not in (cp_model.OPTIMAL, cp_model.INFEASIBLE) and not stopped_at_limit:
        raise RuntimeError(f"Enumeration stopped early: {solver.status_name(status)}")
