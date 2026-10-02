"""Capacitated facility location as a mixed-integer program (MIP).

Decisions: which sites to open (yes/no), and which open site serves each
customer (yes/no).
Goal: minimize weekly fixed costs plus transport costs.
Limits: every customer gets exactly one site, and no site ships more than
its capacity.

The module builds two versions of the same model:

* "weak": the capacity row is the only link between y (open) and x
  (assign): sum_i d_i x_ij <= C_j y_j.
* "strong": the same, plus one extra row per pair: x_ij <= y_j.

Both have the same integer solutions, so the same optimum. But the strong
version has a much tighter LP relaxation, and that makes the MIP solver's
job far easier. Choosing a good formulation is often worth more than a
faster solver.
"""

from dataclasses import dataclass

from ortools.math_opt.python import mathopt

from .data import LocationData

WEAK, STRONG = "weak", "strong"


@dataclass
class LocationModel:
    """A MathOpt model plus handles to its variables."""

    model: mathopt.Model
    open: dict[str, mathopt.Variable]  # Site name -> 1 if open.
    assign: dict[tuple[str, str], mathopt.Variable]  # (customer, site) -> 1 if served.


@dataclass(frozen=True)
class Plan:
    """The result of one solve."""

    open_sites: list[str]
    assignment: dict[str, str]  # Customer name -> site name.
    cost: float  # Best solution found (primal bound).
    bound: float  # Proven lower bound on the optimum (dual bound).
    proven_optimal: bool
    solve_time: float  # Seconds.
    nodes: int  # Branch-and-bound nodes.

    @property
    def gap(self) -> float:
        """Relative gap between solution and bound; 0.0 means optimal."""
        return (self.cost - self.bound) / abs(self.cost) if self.cost else 0.0


def build_model(
    data: LocationData, formulation: str = STRONG, relax: bool = False
) -> LocationModel:
    """Build the MIP. With relax=True, build its LP relaxation instead."""
    if formulation not in (WEAK, STRONG):
        raise ValueError(f"unknown formulation {formulation!r}")
    model = mathopt.Model(name=f"{data.name} ({formulation})")
    # is_integer=False with bounds [0, 1] gives the LP relaxation: the
    # solver may then open "40% of a DC" and pay 40% of its fixed cost.
    integer = not relax
    open_ = {
        s.name: model.add_variable(
            lb=0, ub=1, is_integer=integer, name=f"open {s.name}"
        )
        for s in data.sites
    }
    assign = {
        (c.name, s.name): model.add_variable(
            lb=0, ub=1, is_integer=integer, name=f"{c.name} <- {s.name}"
        )
        for c in data.customers
        for s in data.sites
    }

    # Every customer is served by exactly one site.
    for c in data.customers:
        model.add_linear_constraint(
            mathopt.fast_sum(assign[c.name, s.name] for s in data.sites) == 1
        )

    # Capacity. This row also links x and y: a closed site (y = 0) ships 0.
    for s in data.sites:
        load = mathopt.fast_sum(
            c.demand * assign[c.name, s.name] for c in data.customers
        )
        model.add_linear_constraint(load <= s.capacity * open_[s.name])

    if formulation == STRONG:
        # Redundant for integer solutions, but they cut off fractional
        # ones: a customer cannot be served 100% by a site that is 40% open.
        for c in data.customers:
            for s in data.sites:
                model.add_linear_constraint(assign[c.name, s.name] <= open_[s.name])

    model.minimize(
        mathopt.fast_sum(s.fixed_cost * open_[s.name] for s in data.sites)
        + mathopt.fast_sum(
            data.transport_cost(c, s) * assign[c.name, s.name]
            for c in data.customers
            for s in data.sites
        )
    )
    return LocationModel(model, open_, assign)


def solve_location(
    data: LocationData,
    formulation: str = STRONG,
    solver_type: mathopt.SolverType = mathopt.SolverType.HIGHS,
    time_limit: float = 60.0,
    relative_gap: float = 1e-6,
    force_open_count: int | None = None,
) -> Plan:
    """Solve the MIP and return the plan with its bounds.

    The solver stops at the time limit or once it proves the solution is
    within `relative_gap` of the optimum, whichever comes first. If
    `force_open_count` is set, exactly that many sites must open.
    """
    lm = build_model(data, formulation)
    if force_open_count is not None:
        lm.model.add_linear_constraint(
            mathopt.fast_sum(lm.open.values()) == force_open_count
        )
    params = mathopt.SolveParameters(
        time_limit=_seconds(time_limit),
        relative_gap_tolerance=relative_gap,
        random_seed=1,
    )
    result = mathopt.solve(lm.model, solver_type, params=params)
    reason = result.termination.reason
    if reason not in (
        mathopt.TerminationReason.OPTIMAL,
        mathopt.TerminationReason.FEASIBLE,
    ):
        raise RuntimeError(f"No solution: {result.termination}")

    # Binary values come back as floats such as 0.9999999; round them.
    open_sites = [n for n, v in lm.open.items() if result.variable_values(v) > 0.5]
    assignment = {
        c: s for (c, s), v in lm.assign.items() if result.variable_values(v) > 0.5
    }
    bounds = result.termination.objective_bounds
    return Plan(
        open_sites=open_sites,
        assignment=assignment,
        cost=bounds.primal_bound,
        bound=bounds.dual_bound,
        proven_optimal=reason == mathopt.TerminationReason.OPTIMAL,
        solve_time=result.solve_stats.solve_time.total_seconds(),
        nodes=result.solve_stats.node_count,
    )


def lp_bound(data: LocationData, formulation: str = STRONG) -> float:
    """Return the optimal value of the LP relaxation (a lower bound)."""
    lm = build_model(data, formulation, relax=True)
    result = mathopt.solve(lm.model, mathopt.SolverType.GLOP)
    if result.termination.reason != mathopt.TerminationReason.OPTIMAL:
        raise RuntimeError(f"LP relaxation failed: {result.termination}")
    return result.objective_value()


def _seconds(value: float):
    from datetime import timedelta

    return timedelta(seconds=value)
