"""Cutting stock: column generation with MathOpt and CP-SAT.

A *pattern* says how many narrow rolls of each width we cut from one
jumbo roll. With all patterns known, the problem is a small integer
program (Gilmore and Gomory, 1961):

    minimize   the number of jumbo rolls          sum_p x_p
    subject to every order is met                  sum_p a_ip x_p >= d_i
               x_p >= 0 and whole

But the number of patterns grows very fast with the number of widths.
Column generation never lists them all. It works with a few patterns and
asks a small helper problem (the "pricing" problem) for one new pattern
at a time:

1. Solve the LP over the known patterns (the restricted master).
2. Read the dual values y_i: what one narrow roll of width i is "worth".
3. Find the pattern with the highest total worth. That is a knapsack.
4. If its worth is above 1 (one jumbo roll), it lowers the LP value: add
   it and go to step 1. If not, no pattern can help: the LP is optimal.

Then we need whole numbers of rolls. We compare four ways to get them.
"""

import math
from dataclasses import dataclass

from ortools.math_opt.python import mathopt
from ortools.sat.python import cp_model

from .data import CuttingData

Pattern = tuple[int, ...]  # Number of narrow rolls of each order's width.

# CP-SAT needs whole-number objective coefficients. We scale the dual
# values by this factor and round. The true worth of the chosen pattern is
# then recomputed with the exact duals.
DUAL_SCALE = 1_000_000
TOL = 1e-9


@dataclass(frozen=True)
class Iteration:
    """What one round of column generation saw."""

    lp_value: float  # Value of the restricted master LP.
    best_worth: float  # Highest pattern worth sum_i y_i a_i at the duals.
    lower_bound: float  # lp_value / max(1, best_worth); see check.dual_bound.
    n_patterns: int


@dataclass(frozen=True)
class LpResult:
    """The optimal LP over all patterns, found by column generation."""

    value: float
    patterns: list[Pattern]  # Every pattern generated, in order.
    usage: list[float]  # LP value x_p of each pattern.
    duals: list[float]  # y_i of each demand row at the end.
    history: list[Iteration]


@dataclass(frozen=True)
class CuttingPlan:
    """A whole-number cutting plan: pattern -> number of jumbo rolls."""

    method: str
    cuts: dict[Pattern, int]

    @property
    def rolls(self) -> int:
        """Return the number of jumbo rolls used."""
        return sum(self.cuts.values())


def max_copies(data: CuttingData, i: int) -> int:
    """Return the most rolls of order i that one pattern may hold.

    Never more than fit, never more than the knives allow, and never more
    than the customer ordered. The last limit is not physical, but it makes
    the LP bound stronger at no cost: a pattern with extra copies is never
    needed in a whole-number plan.
    """
    o = data.orders[i]
    return min(data.roll_width // o.width, data.max_pieces, o.quantity)


def initial_patterns(data: CuttingData) -> list[Pattern]:
    """Return one simple pattern per width: as many copies as allowed."""
    n = len(data.orders)
    return [
        tuple(max_copies(data, i) if k == i else 0 for k in range(n)) for i in range(n)
    ]


def price_pattern(data: CuttingData, duals: list[float]) -> tuple[Pattern, float]:
    """Return the pattern of highest worth at these duals, and its worth.

    This is a bounded integer knapsack with one extra rule (the number of
    knives). A dynamic program would also do; check.py has one. CP-SAT
    keeps the code short, and any extra cutting rule is one more line.
    """
    model = cp_model.CpModel()
    a = [
        model.new_int_var(0, max_copies(data, i), f"a{i}")
        for i in range(len(data.orders))
    ]
    model.add(sum(o.width * a[i] for i, o in enumerate(data.orders)) <= data.roll_width)
    model.add(sum(a) <= data.max_pieces)
    model.maximize(sum(round(y * DUAL_SCALE) * a[i] for i, y in enumerate(duals)))

    solver = cp_model.CpSolver()
    solver.parameters.num_workers = 1  # Tiny model: one worker is fastest.
    solver.parameters.max_time_in_seconds = 10.0
    status = solver.solve(model)
    if status != cp_model.OPTIMAL:
        raise RuntimeError(f"Pricing failed with status {solver.status_name(status)}")
    pattern = tuple(solver.value(x) for x in a)
    return pattern, sum(y * k for y, k in zip(duals, pattern, strict=True))


def column_generation(data: CuttingData, max_iterations: int = 500) -> LpResult:
    """Solve the LP over all patterns without listing them."""
    model = mathopt.Model(name="restricted master")
    demand_row = [
        model.add_linear_constraint(lb=o.quantity, name=f"width {o.width}")
        for o in data.orders
    ]
    patterns: list[Pattern] = []
    x: list[mathopt.Variable] = []

    def add_pattern(pattern: Pattern) -> None:
        # Add a column to the existing model: a new variable, its
        # coefficients in the demand rows, and its cost of 1 roll.
        var = model.add_variable(lb=0.0, name=f"pattern {len(x)}")
        for row, count in zip(demand_row, pattern, strict=True):
            if count:
                row.set_coefficient(var, count)
        model.objective.set_linear_coefficient(var, 1.0)
        patterns.append(pattern)
        x.append(var)

    model.objective.is_maximize = False
    for p in initial_patterns(data):
        add_pattern(p)

    # The incremental solver keeps GLOP alive between solves, so each new
    # solve starts from the last optimal basis instead of from scratch.
    solver = mathopt.IncrementalSolver(model, mathopt.SolverType.GLOP)
    history = []
    for _ in range(max_iterations):
        result = solver.solve()
        if result.termination.reason != mathopt.TerminationReason.OPTIMAL:
            raise RuntimeError(f"Master LP failed: {result.termination}")
        duals = [result.dual_values(row) for row in demand_row]
        lp_value = result.objective_value()
        pattern, worth = price_pattern(data, duals)
        history.append(Iteration(lp_value, worth, lp_value / max(1.0, worth), len(x)))
        if worth <= 1.0 + TOL:
            break  # No pattern has negative reduced cost 1 - worth.
        add_pattern(pattern)
    else:
        raise RuntimeError("Column generation did not converge.")

    usage = [result.variable_values(v) for v in x]
    return LpResult(lp_value, patterns, usage, duals, history)


def solve_pattern_ip(
    data: CuttingData, patterns: list[Pattern], time_limit: float = 10.0
) -> CuttingPlan:
    """Solve the integer master over a fixed set of patterns with HiGHS.

    With the patterns from column generation this is "price and branch". It
    is optimal for these patterns, but a missing pattern could be better.
    The LP bound tells us if that can happen.
    """
    model = mathopt.Model(name="integer master")
    x = [
        model.add_integer_variable(lb=0, name=f"pattern {k}")
        for k in range(len(patterns))
    ]
    for i, o in enumerate(data.orders):
        model.add_linear_constraint(
            mathopt.fast_sum(p[i] * x[k] for k, p in enumerate(patterns) if p[i])
            >= o.quantity
        )
    model.minimize(mathopt.fast_sum(x))
    params = mathopt.SolveParameters(time_limit=_seconds(time_limit))
    result = mathopt.solve(model, mathopt.SolverType.HIGHS, params=params)
    if not result.has_primal_feasible_solution():
        raise RuntimeError(f"Integer master failed: {result.termination}")
    cuts = {}
    for p, var in zip(patterns, x, strict=True):
        n = round(result.variable_values(var))
        if n:
            cuts[p] = n
    return CuttingPlan("integer master over generated patterns", cuts)


def round_up(lp: LpResult) -> CuttingPlan:
    """Round every LP value up. Always feasible, often wasteful."""
    cuts = {p: math.ceil(u - 1e-6) for p, u in zip(lp.patterns, lp.usage, strict=True)}
    return CuttingPlan("LP rounded up", {p: n for p, n in cuts.items() if n})


def round_down_and_repair(data: CuttingData, lp: LpResult) -> CuttingPlan:
    """Round every LP value down, then cut what is missing by first fit."""
    cuts = {p: math.floor(u + 1e-6) for p, u in zip(lp.patterns, lp.usage, strict=True)}
    cuts = {p: n for p, n in cuts.items() if n}
    made = [sum(p[i] * n for p, n in cuts.items()) for i in range(len(data.orders))]
    missing = [max(0, o.quantity - m) for o, m in zip(data.orders, made, strict=True)]
    for p in first_fit_decreasing(data, missing):
        cuts[p] = cuts.get(p, 0) + 1
    return CuttingPlan("LP rounded down + first fit", cuts)


def first_fit_plan(data: CuttingData) -> CuttingPlan:
    """Plan by the classic shop-floor rule: first fit decreasing, no LP."""
    cuts: dict[Pattern, int] = {}
    for p in first_fit_decreasing(data, data.demand):
        cuts[p] = cuts.get(p, 0) + 1
    return CuttingPlan("first fit decreasing", cuts)


def first_fit_decreasing(data: CuttingData, demand: list[int]) -> list[Pattern]:
    """Cut the widest rolls first; put each into the first jumbo it fits."""
    n = len(data.orders)
    rolls: list[list[int]] = []  # Pieces per order, per jumbo roll.
    room: list[int] = []
    for i in sorted(range(n), key=lambda k: -data.orders[k].width):
        for _ in range(demand[i]):
            w = data.orders[i].width
            for r, counts in enumerate(rolls):
                if room[r] >= w and sum(counts) < data.max_pieces:
                    counts[i] += 1
                    room[r] -= w
                    break
            else:
                rolls.append([1 if k == i else 0 for k in range(n)])
                room.append(data.roll_width - w)
    return [tuple(c) for c in rolls]


def kantorovich(
    data: CuttingData, n_rolls: int, time_limit: float = 10.0, relax: bool = False
) -> tuple[float, bool]:
    """Solve the direct model: which piece goes on which jumbo roll.

    y_k = 1 if jumbo roll k is used; z_ik = rolls of width i cut from it.
    `n_rolls` must be an upper bound on the rolls needed (for example from
    first fit). This model is easy to write but hard to solve: its LP bound
    is weak, and the rolls are interchangeable, so the solver sees many
    copies of every solution.

    Returns:
        (objective, proven optimal). With relax=True, the LP value.

    """
    model = mathopt.Model(name="kantorovich")
    rolls = range(n_rolls)
    if relax:
        y = [model.add_variable(lb=0, ub=1) for _ in rolls]
        z = {
            (i, k): model.add_variable(lb=0)
            for i in range(len(data.orders))
            for k in rolls
        }
    else:
        y = [model.add_binary_variable() for _ in rolls]
        z = {
            (i, k): model.add_integer_variable(lb=0, ub=max_copies(data, i))
            for i in range(len(data.orders))
            for k in rolls
        }
    for k in rolls:
        model.add_linear_constraint(
            mathopt.fast_sum(o.width * z[i, k] for i, o in enumerate(data.orders))
            <= data.roll_width * y[k]
        )
        model.add_linear_constraint(
            mathopt.fast_sum(z[i, k] for i in range(len(data.orders)))
            <= data.max_pieces * y[k]
        )
        if k > 0:
            model.add_linear_constraint(y[k] <= y[k - 1])  # Some symmetry breaking.
    for i, o in enumerate(data.orders):
        model.add_linear_constraint(
            mathopt.fast_sum(z[i, k] for k in rolls) >= o.quantity
        )
    model.minimize(mathopt.fast_sum(y))
    params = mathopt.SolveParameters(time_limit=_seconds(time_limit))
    solver = mathopt.SolverType.GLOP if relax else mathopt.SolverType.HIGHS
    result = mathopt.solve(model, solver, params=params)
    if not result.has_primal_feasible_solution():
        raise RuntimeError(f"Kantorovich model failed: {result.termination}")
    optimal = result.termination.reason == mathopt.TerminationReason.OPTIMAL
    return result.objective_value(), optimal


def _seconds(value: float):
    import datetime

    return datetime.timedelta(seconds=value)
