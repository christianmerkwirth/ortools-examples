"""Nurse rostering with CP-SAT.

Decision: for each nurse and day, which shift they work, if any.
Hard rules: coverage, a senior on every shift, one shift per day, rest
time between shifts, at most N days in a row, contract limits, leave.
Soft rules: personal requests, no isolated days off, and a fair share of
nights and weekends. Each broken soft rule costs penalty points, and the
solver minimizes the total.

The model uses one Boolean variable per (nurse, day, shift). Every rule
is a small linear or logical constraint on these variables. This "grid of
Booleans" pattern fits most rostering and timetabling problems.
"""

from dataclasses import dataclass

from ortools.sat.python import cp_model

from .data import RosterData

# Penalty categories, in the order the report prints them.
CATEGORIES = ("requests", "isolated days off", "night spread", "weekend spread")


class NoRosterError(RuntimeError):
    """Raised when CP-SAT proves that no roster meets the hard rules."""


@dataclass
class RosterModel:
    """A CP-SAT model plus handles to its variables and penalty terms."""

    model: cp_model.CpModel
    x: dict[tuple[str, int, str], cp_model.IntVar]  # (nurse, day, shift) -> bool.
    # Weighted penalty expression per category (see CATEGORIES).
    penalty: dict[str, cp_model.LinearExprT]
    # Unweighted spreads: "nights" and "weekends" -> most minus fewest.
    spread: dict[str, cp_model.LinearExprT]


@dataclass(frozen=True)
class Roster:
    """A solved roster."""

    # (nurse, day) -> shift code, or None for a day off.
    assignment: dict[tuple[str, int], str | None]
    objective: int  # Total penalty points (all categories).
    penalty: dict[str, int]  # Points per category.
    # Bound and optimality refer to the objective that was solved. For the
    # weighted model, that is `objective`.
    best_bound: float  # No roster can score below this.
    optimal: bool
    wall_time: float  # Seconds.

    @property
    def gap(self) -> float:
        """Relative distance between the roster and the best bound."""
        if self.objective == 0:
            return 0.0
        return (self.objective - self.best_bound) / self.objective


def build_model(data: RosterData) -> RosterModel:
    """Build the CP-SAT model with all hard rules and soft-rule penalties."""
    model = cp_model.CpModel()
    days = range(data.num_days)
    codes = [s.code for s in data.shifts]

    # x[n, d, s] is true if nurse n works shift s on day d.
    x = {
        (n.name, d, s): model.new_bool_var(f"{n.name}_{d}_{s}")
        for n in data.nurses
        for d in days
        for s in codes
    }

    # work[n, d] is true if nurse n works any shift on day d. Together with
    # add_exactly_one, it also enforces "at most one shift per day".
    work = {}
    for n in data.nurses:
        for d in days:
            off = model.new_bool_var(f"{n.name}_{d}_off")
            model.add_exactly_one([off, *(x[n.name, d, s] for s in codes)])
            work[n.name, d] = ~off

    # Coverage and skill mix: enough nurses, and at least one senior.
    seniors = [n.name for n in data.nurses if n.senior]
    for d in days:
        for s in codes:
            need = data.demand[d].get(s, 0)
            model.add(sum(x[n.name, d, s] for n in data.nurses) >= need)
            if need > 0:
                model.add_at_least_one(x[n, d, s] for n in seniors)

    for n in data.nurses:
        # Rest time: shift a today forbids shift b tomorrow.
        for d in range(data.num_days - 1):
            for a, b in data.forbidden_sequences:
                model.add_implication(x[n.name, d, a], ~x[n.name, d + 1, b])

        # At most k working days in any window of k + 1 days.
        k = data.max_consecutive_days
        for start in range(data.num_days - k):
            model.add(sum(work[n.name, d] for d in range(start, start + k + 1)) <= k)

        # Contract: total shifts in the period.
        total = sum(work[n.name, d] for d in days)
        model.add_linear_constraint(total, n.min_shifts, n.max_shifts)

        # Approved leave is a hard rule.
        for d in n.leave:
            model.add(work[n.name, d] == 0)

    w = data.weights

    # Soft rule 1: requests. The penalty literal is the very event the nurse
    # wants to avoid, so no extra variable is needed.
    request_terms = []
    for r in data.requests:
        lit = work[r.nurse, r.day] if r.shift is None else x[r.nurse, r.day, r.shift]
        request_terms.append(r.weight * lit)

    # Soft rule 2: isolated days off (work, off, work). The clause
    # "not work[d-1] or work[d] or not work[d+1] or iso" forces iso to be
    # true exactly when the pattern occurs (the objective keeps it false
    # otherwise).
    isolated = []
    for n in data.nurses:
        for d in range(1, data.num_days - 1):
            iso = model.new_bool_var(f"{n.name}_{d}_isolated_off")
            model.add_bool_or(
                [~work[n.name, d - 1], work[n.name, d], ~work[n.name, d + 1], iso]
            )
            isolated.append(iso)

    # Soft rules 3 and 4: fairness. Minimize the spread (most minus fewest)
    # of nights and of weekend shifts among full-time nurses.
    full_time = [n for n in data.nurses if n.full_time]
    nights = [sum(x[n.name, d, data.night_shift] for d in days) for n in full_time]
    weekends = [sum(work[n.name, d] for d in data.weekend_days) for n in full_time]

    spread = {
        "nights": _spread(model, nights, data.num_days, "nights"),
        "weekends": _spread(model, weekends, len(data.weekend_days), "weekends"),
    }
    penalty = {
        "requests": sum(request_terms),
        "isolated days off": w.isolated_day_off * sum(isolated),
        "night spread": w.night_spread * spread["nights"],
        "weekend spread": w.weekend_spread * spread["weekends"],
    }
    model.minimize(sum(penalty.values()))
    return RosterModel(model, x, penalty, spread)


def _spread(model: cp_model.CpModel, counts, upper: int, name: str):
    """Return max(counts) - min(counts) as a new expression."""
    if not counts:
        return 0
    most = model.new_int_var(0, upper, f"most_{name}")
    fewest = model.new_int_var(0, upper, f"fewest_{name}")
    model.add_max_equality(most, counts)
    model.add_min_equality(fewest, counts)
    return most - fewest


def solve_model(rm: RosterModel, data: RosterData, time_limit: float = 20.0) -> Roster:
    """Solve a built model and read back the roster."""
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 8
    status = solver.solve(rm.model)
    if status == cp_model.INFEASIBLE:
        raise NoRosterError(f"No roster meets the hard rules of '{data.name}'.")
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(
            f"CP-SAT stopped without a roster: {solver.status_name(status)}"
        )

    assignment = {}
    for n in data.nurses:
        for d in range(data.num_days):
            on = [
                s.code
                for s in data.shifts
                if solver.boolean_value(rm.x[n.name, d, s.code])
            ]
            assignment[n.name, d] = on[0] if on else None
    penalty = {c: round(solver.value(e)) for c, e in rm.penalty.items()}
    return Roster(
        assignment=assignment,
        objective=sum(penalty.values()),
        penalty=penalty,
        best_bound=solver.best_objective_bound,
        optimal=status == cp_model.OPTIMAL,
        wall_time=solver.wall_time,
    )


def solve_roster(data: RosterData, time_limit: float = 20.0) -> Roster:
    """Build and solve the weighted model in one call."""
    return solve_model(build_model(data), data, time_limit)


def solve_lexicographic(
    data: RosterData, first: tuple[str, ...], time_limit: float = 20.0
) -> tuple[Roster, Roster]:
    """Optimize in two stages instead of with one weighted sum.

    Stage 1 minimizes only the `first` categories (for example fairness).
    Stage 2 keeps that result fixed and minimizes everything else. The
    stage 1 roster is passed to stage 2 as a hint, so stage 2 starts from
    a good roster instead of from scratch.

    Returns:
        The stage 1 and stage 2 rosters.

    """
    rm = build_model(data)
    first_expr = sum(rm.penalty[c] for c in first)
    rest_expr = sum(rm.penalty[c] for c in CATEGORIES if c not in first)

    rm.model.minimize(first_expr)
    stage1 = solve_model(rm, data, time_limit)

    # Lock in stage 1. If it was not proven optimal, its value is still a
    # valid (if loose) limit.
    rm.model.add(first_expr <= sum(stage1.penalty[c] for c in first))
    rm.model.minimize(rest_expr)
    for (n, d, s), var in rm.x.items():
        rm.model.add_hint(var, stage1.assignment[n, d] == s)
    stage2 = solve_model(rm, data, time_limit)
    return stage1, stage2


def night_fairness_tradeoff(
    data: RosterData, max_spreads: list[int], time_limit: float = 20.0
) -> list[tuple[int, Roster]]:
    """Trace the price of fair nights (the epsilon-constraint method).

    For each limit k, require "most nights minus fewest nights <= k" as a
    hard rule. Then minimize all other penalties. The result shows how many
    wishes each extra step of fairness costs.
    """
    points = []
    for k in max_spreads:
        rm = build_model(data)
        rm.model.add(rm.spread["nights"] <= k)
        rm.model.minimize(sum(rm.penalty[c] for c in CATEGORIES if c != "night spread"))
        points.append((k, solve_model(rm, data, time_limit)))
    return points
