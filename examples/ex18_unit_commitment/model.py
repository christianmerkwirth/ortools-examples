"""Unit commitment as a mixed-integer program (MIP), solved with MathOpt.

Decisions, for every plant and every hour:
* on[g, t]: is the plant running? (binary)
* power[g, t]: how many MW does it produce? (continuous)
Goal: minimize fuel, no-load, and startup costs for the whole day.
Limits: demand met every hour, plant output limits, ramp rates, minimum
up and down times, spinning reserve, and the dam's daily water budget.

Every hour depends on the hour before it (a plant that starts at 14:00
must still run at 15:00), so the hours cannot be planned one by one. This
"time coupling" is what makes the problem a real MIP.

The module builds two versions of the minimum up/down rules:

* "weak": if a plant starts at t, then all of the next L hours are on,
  written as one aggregated row: L * start[t] <= sum of on[t .. t+L-1].
* "strong": the turn-on/turn-off inequalities of Rajan and Takriti
  (2005): the starts in the last L hours are at most on[t].

Both versions have exactly the same integer solutions. The strong one has
a much tighter LP relaxation, so the MIP solver has less work to do.
"""

import datetime
from dataclasses import dataclass

from ortools.math_opt.python import mathopt

from .data import GridData

WEAK, STRONG = "weak", "strong"
OPTIMAL = mathopt.TerminationReason.OPTIMAL
FEASIBLE = mathopt.TerminationReason.FEASIBLE


class NoScheduleError(RuntimeError):
    """Raised when the solver finds no schedule."""


@dataclass
class CommitmentModel:
    """A MathOpt model plus handles to its variables and key rows."""

    model: mathopt.Model
    on: dict[tuple[str, int], mathopt.Variable]
    start: dict[tuple[str, int], mathopt.Variable]
    stop: dict[tuple[str, int], mathopt.Variable]
    power: dict[tuple[str, int], mathopt.Variable]
    solar: list[mathopt.Variable]  # Solar power used (the rest is curtailed).
    shed: list[mathopt.Variable]  # Demand not served; should stay zero.
    balance: list[mathopt.LinearConstraint]  # Supply = demand, per hour.
    reserve: list[mathopt.LinearConstraint]  # Spare capacity, per hour.
    water: dict[str, mathopt.LinearConstraint]  # Daily energy limits.


@dataclass(frozen=True)
class Schedule:
    """The result of one solve."""

    on: dict[str, tuple[int, ...]]  # Unit -> 0/1 per hour.
    power: dict[str, tuple[float, ...]]  # Unit -> MW per hour.
    solar_used: tuple[float, ...]
    shed: tuple[float, ...]
    cost: float  # Best solution found (primal bound).
    bound: float  # Proven lower bound on the optimum (dual bound).
    proven_optimal: bool
    solve_time: float  # Seconds.

    @property
    def gap(self) -> float:
        """Relative gap between solution and bound; 0.0 means optimal."""
        if not self.cost:
            return 0.0
        return max(0.0, (self.cost - self.bound) / abs(self.cost))  # No "-0.000%".


@dataclass(frozen=True)
class Prices:
    """Approximate marginal prices from the LP with fixed on/off decisions."""

    energy: tuple[float, ...]  # $ per MWh, per hour.
    reserve: tuple[float, ...]  # $ per MW of reserve, per hour.
    water_value: dict[str, float]  # $ per MWh of stored water, per dam.
    dispatch_cost: float  # Optimal cost with the on/off plan fixed.


def build_model(
    data: GridData,
    formulation: str = STRONG,
    relax: bool = False,
    fixed_on: dict[str, tuple[int, ...]] | None = None,
) -> CommitmentModel:
    """Build the MIP.

    Args:
        data: the instance.
        formulation: WEAK or STRONG minimum up/down rows.
        relax: build the LP relaxation (on/start/stop in [0, 1]).
        fixed_on: fix every on/off decision to this schedule. The model is
            then an LP, which gives prices (see `hourly_prices`).

    """
    if formulation not in (WEAK, STRONG):
        raise ValueError(f"unknown formulation {formulation!r}")
    model = mathopt.Model(name=f"{data.name} ({formulation})")
    hours = range(data.hours)
    integer = not relax and fixed_on is None

    on, start, stop, power = {}, {}, {}, {}
    for g in data.units:
        for t in hours:
            key = (g.name, t)
            on[key] = model.add_variable(
                lb=0, ub=1, is_integer=integer, name=f"on {key}"
            )
            # start and stop are 0/1 whenever `on` is 0/1 (they follow from
            # the transition row below), so they can stay continuous.
            start[key] = model.add_variable(lb=0, ub=1, name=f"start {key}")
            stop[key] = model.add_variable(lb=0, ub=1, name=f"stop {key}")
            power[key] = model.add_variable(lb=0, ub=g.p_max, name=f"power {key}")
    solar = [
        model.add_variable(lb=0, ub=data.solar[t], name=f"solar {t}") for t in hours
    ]
    shed = [
        model.add_variable(lb=0, ub=data.demand[t], name=f"shed {t}") for t in hours
    ]

    for g in data.units:
        u = [on[g.name, t] for t in hours]
        v = [start[g.name, t] for t in hours]
        w = [stop[g.name, t] for t in hours]
        p = [power[g.name, t] for t in hours]
        u_before = 1.0 if g.initial_on else 0.0
        p_before = g.initial_output

        for t in hours:
            u_prev = u[t - 1] if t > 0 else u_before
            # Transition: a change of state is a start or a stop.
            model.add_linear_constraint(u[t] - u_prev == v[t] - w[t])
            model.add_linear_constraint(v[t] + w[t] <= 1)
            # Output limits: zero when off, between p_min and p_max when on.
            model.add_linear_constraint(p[t] >= g.p_min * u[t])
            model.add_linear_constraint(p[t] <= g.p_max * u[t])

            # Ramp limits. A starting unit may jump to at most its start-up
            # level, and a stopping unit must come down to it first.
            if g.ramp is not None:
                p_prev = p[t - 1] if t > 0 else p_before
                jump = max(g.p_min, g.ramp)
                model.add_linear_constraint(
                    p[t] - p_prev <= g.ramp * u_prev + jump * v[t]
                )
                model.add_linear_constraint(
                    p_prev - p[t] <= g.ramp * u[t] + jump * w[t]
                )

        # The plant may have to keep its state from before the plan starts.
        if g.initial_on:
            for t in range(min(data.hours, max(0, g.min_up - g.initial_hours))):
                model.add_linear_constraint(u[t] == 1)
        else:
            for t in range(min(data.hours, max(0, g.min_down - g.initial_hours))):
                model.add_linear_constraint(u[t] == 0)

        if formulation == STRONG:
            _strong_min_up_down(model, g, u, v, w, data.hours)
        else:
            _weak_min_up_down(model, g, u, v, w, data.hours)

        if fixed_on is not None:
            for t in hours:
                u[t].lower_bound = u[t].upper_bound = fixed_on[g.name][t]

    # Supply meets demand. Shedding load costs so much that it is a last
    # resort; it keeps the model feasible and shows where the grid is short.
    balance = [
        model.add_linear_constraint(
            mathopt.fast_sum(power[g.name, t] for g in data.units) + solar[t] + shed[t]
            == data.demand[t],
            name=f"balance {t}",
        )
        for t in hours
    ]
    # Spinning reserve: running plants must have spare room to cover the
    # sudden loss of a plant or a forecast error.
    reserve = [
        model.add_linear_constraint(
            mathopt.fast_sum(
                g.p_max * on[g.name, t] - power[g.name, t] for g in data.units
            )
            >= data.reserve(t),
            name=f"reserve {t}",
        )
        for t in hours
    ]
    water = {
        g.name: model.add_linear_constraint(
            mathopt.fast_sum(power[g.name, t] for t in hours) <= g.energy_limit,
            name=f"water {g.name}",
        )
        for g in data.units
        if g.energy_limit is not None
    }

    model.minimize(
        mathopt.fast_sum(
            g.marginal_cost * power[g.name, t]
            + g.no_load_cost * on[g.name, t]
            + g.startup_cost * start[g.name, t]
            for g in data.units
            for t in hours
        )
        + data.shed_cost * mathopt.fast_sum(shed)
    )
    return CommitmentModel(
        model, on, start, stop, power, solar, shed, balance, reserve, water
    )


def _strong_min_up_down(model, g, u, v, w, hours: int) -> None:
    """Turn-on/turn-off inequalities.

    If the unit started in any of the last `min_up` hours, it must be on
    now: sum(start[t-min_up+1 .. t]) <= on[t]. Likewise for stops and
    `min_down`: sum(stop[t-min_down+1 .. t]) <= 1 - on[t].

    In the LP relaxation, a fractional start of 0.3 forces 0.3 of "on" in
    each of the next hours, and starts add up. This is the convex hull of
    the min up/down rules for one unit, so no LP bound can be tighter.
    """
    for t in range(hours):
        if g.min_up > 1:
            model.add_linear_constraint(
                mathopt.fast_sum(v[s] for s in range(max(0, t - g.min_up + 1), t + 1))
                <= u[t]
            )
        if g.min_down > 1:
            model.add_linear_constraint(
                mathopt.fast_sum(w[s] for s in range(max(0, t - g.min_down + 1), t + 1))
                <= 1 - u[t]
            )


def _weak_min_up_down(model, g, u, v, w, hours: int) -> None:
    """Add aggregated min up/down rows, as found in many older models.

    If the unit starts at t, the next L hours are all on:
    L * start[t] <= sum(on[t .. t+L-1]). For 0/1 values this is the same
    rule. But in the LP relaxation the sum may spread "on" thinly over the
    window, so the LP can cheat much more.
    """
    for t in range(hours):
        if g.min_up > 1:
            window = range(t, min(hours, t + g.min_up))
            model.add_linear_constraint(
                len(window) * v[t] <= mathopt.fast_sum(u[s] for s in window)
            )
        if g.min_down > 1:
            window = range(t, min(hours, t + g.min_down))
            model.add_linear_constraint(
                len(window) * w[t] <= mathopt.fast_sum(1 - u[s] for s in window)
            )


def solve_commitment(
    data: GridData,
    formulation: str = STRONG,
    solver: mathopt.SolverType = mathopt.SolverType.HIGHS,
    time_limit: float = 30.0,
    relative_gap: float = 1e-6,
) -> Schedule:
    """Solve the MIP and return the best schedule found."""
    cm = build_model(data, formulation)
    params = mathopt.SolveParameters(
        time_limit=datetime.timedelta(seconds=time_limit),
        relative_gap_tolerance=relative_gap,
    )
    result = mathopt.solve(cm.model, solver, params=params)
    if result.termination.reason not in (OPTIMAL, FEASIBLE):
        raise NoScheduleError(f"No schedule found: {result.termination}")

    names = [g.name for g in data.units]
    hours = range(data.hours)
    return Schedule(
        on={
            n: tuple(round(result.variable_values(cm.on[n, t])) for t in hours)
            for n in names
        },
        power={
            n: tuple(result.variable_values(cm.power[n, t]) for t in hours)
            for n in names
        },
        solar_used=tuple(result.variable_values(s) for s in cm.solar),
        shed=tuple(result.variable_values(s) for s in cm.shed),
        cost=result.objective_value(),
        bound=result.termination.objective_bounds.dual_bound,
        proven_optimal=result.termination.reason == OPTIMAL,
        solve_time=result.solve_time().total_seconds(),
    )


def _clean(value: float, tol: float = 1e-6) -> float:
    """Map tiny round-off values (and -0.0) to 0.0 for readable output."""
    return 0.0 if abs(value) < tol else value


def lp_bound(data: GridData, formulation: str = STRONG) -> float:
    """Return the optimal value of the LP relaxation: a lower bound."""
    cm = build_model(data, formulation, relax=True)
    result = mathopt.solve(cm.model, mathopt.SolverType.GLOP)
    if result.termination.reason != OPTIMAL:
        raise NoScheduleError(f"LP relaxation failed: {result.termination}")
    return result.objective_value()


def hourly_prices(data: GridData, schedule: Schedule) -> Prices:
    """Return marginal prices: fix the on/off plan, solve the LP, read duals.

    A MIP has no dual values (example 01 needed an LP for its shadow
    prices). Grid operators therefore fix the integer decisions at their
    optimal values and solve the remaining LP. Its dual values say what
    one more MW of demand would cost in each hour, given the plants that
    run. This is how many electricity markets set their hourly prices.
    """
    cm = build_model(data, fixed_on=schedule.on)
    result = mathopt.solve(cm.model, mathopt.SolverType.GLOP)
    if result.termination.reason != OPTIMAL:
        raise NoScheduleError(f"Pricing LP failed: {result.termination}")
    return Prices(
        energy=tuple(result.dual_values(c) for c in cm.balance),
        reserve=tuple(result.dual_values(c) for c in cm.reserve),
        # The water row is "<=", so its dual is <= 0; flip the sign to get
        # the value of one more MWh of water in the dam.
        water_value={n: -result.dual_values(c) for n, c in cm.water.items()},
        dispatch_cost=result.objective_value(),
    )
