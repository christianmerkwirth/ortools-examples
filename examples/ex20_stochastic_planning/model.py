"""Season planning under uncertain demand: a two-stage stochastic MIP.

Stage 1, in summer (here and now): rent n sewing lines (integer) and make
x_p units of each product. Sewing hours must fit on the lines.

Stage 2, in winter (wait and see): demand d_sp and express price r_sp are
known in scenario s. Buy y_sp express units, sell z_sp units, and send the
rest to the outlet. Each scenario has its own y and z: we may react to
what we see, but n and x are fixed before we see anything.

    max  -F n - sum_p c_p x_p
         + 1/S sum_s sum_p [ q_p z_sp + v_p (x_p + y_sp - z_sp) - r_sp y_sp ]

    s.t. sum_p a_p x_p <= H n
         z_sp <= x_p + y_sp     (sell only what we have)
         z_sp <= d_sp           (sell only what customers want)
         y_sp <= u_p            (express capacity)

With one scenario (the point forecast) this is a plain deterministic MIP.
With S scenarios it is the "extensive form" of the stochastic program:
the same model, S copies of the second stage, one shared first stage.
"""

import datetime
from dataclasses import dataclass

import numpy as np
from ortools.math_opt.python import mathopt

from .data import PlanningData, Scenarios

OPTIMAL = mathopt.TerminationReason.OPTIMAL


@dataclass(frozen=True)
class Plan:
    """The stage 1 decision: what we commit to in summer."""

    lines: int
    make: np.ndarray  # Units made in-house, per product.


@dataclass(frozen=True)
class Solution:
    """A plan and the profit the model expects from it."""

    plan: Plan
    # Average profit over the scenarios the model was given. This is the
    # model's own promise; it is not what the plan earns in the real future.
    promised: float
    proven_optimal: bool


def solve_plan(
    data: PlanningData, scenarios: Scenarios, time_limit: float = 60.0
) -> Solution:
    """Build and solve the extensive form for the given scenarios."""
    model = mathopt.Model(name="season plan")
    plant, P, S = data.plant, range(len(data.products)), scenarios.count
    lines = model.add_integer_variable(lb=0, ub=plant.max_lines, name="lines")
    make = [model.add_variable(lb=0.0, name=f"make {p.name}") for p in data.products]
    model.add_linear_constraint(
        mathopt.fast_sum(p.hours * x for p, x in zip(data.products, make, strict=True))
        <= plant.line_hours * lines,
        name="sewing hours",
    )

    # Every unit we own is worth at least its outlet price, so write the
    # stage 2 profit as  v x + (q - v) z - (r - v) y.  This form shows the
    # margins at stake: (q - v) for a sale, (r - v) for an express unit.
    stage2 = []
    for s in range(S):
        for i in P:
            prod = data.products[i]
            sell = model.add_variable(lb=0.0, ub=float(scenarios.demand[s, i]))
            rush = model.add_variable(lb=0.0, ub=prod.rush_cap)
            model.add_linear_constraint(sell <= make[i] + rush)
            margin_rush = float(scenarios.rush_price[s, i]) - prod.salvage
            stage2.append((prod.price - prod.salvage) * sell - margin_rush * rush)

    first = -plant.line_cost * lines - mathopt.fast_sum(
        (p.cost - p.salvage) * x for p, x in zip(data.products, make, strict=True)
    )
    model.maximize(first + (1.0 / S) * mathopt.fast_sum(stage2))

    params = mathopt.SolveParameters(
        time_limit=datetime.timedelta(seconds=time_limit),
        relative_gap_tolerance=1e-7,
    )
    result = mathopt.solve(model, mathopt.SolverType.HIGHS, params=params)
    if not result.has_primal_feasible_solution():
        raise RuntimeError(f"No plan found: {result.termination}")
    x = np.array([result.variable_values(v) for v in make])
    x[np.abs(x) < 1e-7] = 0.0  # Clean solver round-off.
    plan = Plan(round(result.variable_values(lines)), x)
    return Solution(
        plan, result.objective_value(), result.termination.reason == OPTIMAL
    )


def solve_each(data: PlanningData, scenarios: Scenarios) -> list[Solution]:
    """Solve the deterministic model once per scenario (the "what-if" approach).

    Each run sees one future as if it were certain. The runs disagree, and
    none of them is a plan we can carry out: in summer we do not know which
    run will turn out right.
    """
    return [solve_plan(data, scenarios.subset([s])) for s in range(scenarios.count)]


def average_plan(data: PlanningData, solutions: list[Solution]) -> Plan:
    """Average the what-if plans into one plan (a common heuristic).

    Round the lines up, so the averaged production always fits.
    """
    make = np.mean([s.plan.make for s in solutions], axis=0)
    hours = float(data.column("hours") @ make)
    return Plan(int(np.ceil(hours / data.plant.line_hours - 1e-9)), make)
