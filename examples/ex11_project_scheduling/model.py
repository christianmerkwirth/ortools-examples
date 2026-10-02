"""Resource-constrained project scheduling (RCPSP) with CP-SAT.

Decision: the start day of every activity.
Goal: finish the project as early as possible (minimize the makespan).
Limits: an activity starts only after all its predecessors end, and on no
day may the activities in progress need more of a resource than exists.

CP-SAT has two building blocks made for this:

* An interval variable ties start, duration, and end together.
* `add_cumulative` keeps the summed demand of all intervals that run at
  the same time below a capacity, on every day.
"""

from dataclasses import dataclass

from ortools.sat.python import cp_model

from .data import ProjectData


@dataclass(frozen=True)
class ProjectSchedule:
    """A solved schedule."""

    start: dict[str, int]  # Activity name -> start day.
    makespan: int
    lower_bound: int  # CP-SAT's proven bound; equal to makespan if optimal.
    proven_optimal: bool


def _build(data: ProjectData, horizon: int):
    """Build the CP-SAT model. Return it with the start, end, and makespan vars."""
    model = cp_model.CpModel()
    start, end, interval = {}, {}, {}
    for a in data.activities:
        start[a.name] = model.new_int_var(0, horizon - a.duration, f"start {a.name}")
        end[a.name] = model.new_int_var(a.duration, horizon, f"end {a.name}")
        # The interval enforces end == start + duration.
        interval[a.name] = model.new_interval_var(
            start[a.name], a.duration, end[a.name], a.name
        )

    # Precedences: a successor starts no earlier than its predecessor ends.
    for a in data.activities:
        for p in a.predecessors:
            model.add(start[a.name] >= end[p])

    # Renewable resources: one cumulative constraint per resource. Only
    # the activities that use the resource take part.
    for r in data.resources:
        users = [a for a in data.activities if a.needs(r.name) > 0]
        model.add_cumulative(
            [interval[a.name] for a in users],
            [a.needs(r.name) for a in users],
            r.capacity,
        )

    makespan = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(makespan, list(end.values()))
    return model, start, makespan


def solve_schedule(
    data: ProjectData,
    time_limit: float = 20.0,
    max_makespan: int | None = None,
    tidy: bool = True,
) -> ProjectSchedule:
    """Find a shortest schedule.

    Args:
        data: The project.
        time_limit: Seconds before CP-SAT stops and returns its best schedule.
        max_makespan: Optional known upper bound (for example from a quick
            heuristic). A smaller horizon gives a smaller model.
        tidy: Run a second, quick solve that keeps the shortest makespan and
            starts every activity as early as possible.

    """
    # The horizon is the latest day anything can end. Doing all activities
    # one after another always works, so the sum of durations is safe.
    horizon = max_makespan or sum(a.duration for a in data.activities)

    # Phase 1: minimize the makespan with 8 parallel workers.
    model, start, makespan = _build(data, horizon)
    model.minimize(makespan)
    solver = _solver(time_limit, workers=8)
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"No schedule found: {solver.status_name(status)}")
    best = round(solver.objective_value)
    result = ProjectSchedule(
        start={name: solver.value(var) for name, var in start.items()},
        makespan=best,
        lower_bound=round(solver.best_objective_bound),
        proven_optimal=status == cp_model.OPTIMAL,
    )
    if not tidy:
        return result

    # Phase 2: keep that makespan, and minimize the sum of start days. Many
    # schedules share the shortest makespan, and phase 1 may return one
    # with idle gaps. This picks a tidy one. One worker makes it
    # repeatable: the same input always gives the same schedule.
    model, start, makespan = _build(data, best)
    model.minimize(sum(start.values()))
    solver = _solver(time_limit, workers=1)
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return result  # Out of time: keep the phase 1 schedule.
    return ProjectSchedule(
        start={name: solver.value(var) for name, var in start.items()},
        makespan=best,
        lower_bound=result.lower_bound,
        proven_optimal=result.proven_optimal,
    )


def _solver(time_limit: float, workers: int) -> cp_model.CpSolver:
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = workers
    return solver


def extra_unit_gains(
    data: ProjectData, base_makespan: int, time_limit: float = 10.0
) -> dict[str, int]:
    """Return the days saved by one extra unit of each resource.

    This is the "what if we hire one more crew?" question. Each answer is a
    full re-solve, so it accounts for all knock-on effects.
    """
    gains = {}
    for r in data.resources:
        bigger = data.with_capacity(r.name, r.capacity + 1)
        result = solve_schedule(
            bigger, time_limit, max_makespan=base_makespan, tidy=False
        )
        gains[r.name] = base_makespan - result.makespan
    return gains
