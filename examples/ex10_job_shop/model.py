"""Job shop scheduling with CP-SAT interval variables.

Decision: the start time of every operation.
Goal: finish all jobs as early as possible (minimize the makespan), or,
in a second variant, minimize the weighted lateness against due dates.
Limits: each job follows its route in order, and each machine runs one
operation at a time.

CP-SAT has a special variable type for this: the interval variable. It
ties a start, a duration, and an end together. `add_no_overlap` then
says "these intervals must not overlap in time". That one constraint
replaces the many either-or constraints a MIP model would need.
"""

from dataclasses import dataclass, field

from ortools.sat.python import cp_model

from .data import JobShopData

OpKey = tuple[str, int]  # (job name, position in the route).


@dataclass(frozen=True)
class Schedule:
    """A solved schedule and what the solver knows about its quality."""

    start: dict[OpKey, int]  # Start time of every operation.
    makespan: int  # End time of the last operation.
    objective: int  # Makespan or weighted tardiness, per the objective.
    best_bound: int  # Proven lower bound on the objective.
    optimal: bool
    wall_time: float  # Seconds.
    # (seconds, objective, bound) for every improving solution found.
    progress: list[tuple[float, int, int]] = field(default_factory=list)


@dataclass
class JobShopModel:
    """A CP-SAT model plus handles to its variables."""

    model: cp_model.CpModel
    start: dict[OpKey, cp_model.IntVar]
    end: dict[OpKey, cp_model.IntVar]
    makespan: cp_model.IntVar
    tardiness: dict[str, cp_model.IntVar]


def build_model(
    data: JobShopData,
    objective: str = "makespan",
    max_makespan: int | None = None,
    max_tardiness: int | None = None,
) -> JobShopModel:
    """Build the CP-SAT model.

    Args:
        data: the instance.
        objective: "makespan" or "tardiness" (weighted sum of lateness).
        max_makespan: optional cap on the makespan.
        max_tardiness: optional cap on the weighted tardiness.

    """
    model = cp_model.CpModel()

    # A safe horizon: run every operation one after another. No optimal
    # schedule is longer than that. A tight horizon keeps domains small.
    horizon = sum(op.duration for job in data.jobs for op in job.operations)

    start, end = {}, {}
    on_machine: dict[str, list[cp_model.IntervalVar]] = {m: [] for m in data.machines}
    for job in data.jobs:
        for k, op in enumerate(job.operations):
            key = (job.name, k)
            start[key] = model.new_int_var(0, horizon, f"start {job.name} #{k}")
            end[key] = model.new_int_var(0, horizon, f"end {job.name} #{k}")
            # The interval links the three: start + duration == end.
            interval = model.new_interval_var(
                start[key], op.duration, end[key], f"{job.name} #{k} on {op.machine}"
            )
            on_machine[op.machine].append(interval)

    # Route order: an operation starts after the previous one of its job ends.
    for job in data.jobs:
        for k in range(1, len(job.operations)):
            model.add(start[job.name, k] >= end[job.name, k - 1])

    # One operation at a time per machine.
    for intervals in on_machine.values():
        model.add_no_overlap(intervals)

    # The makespan is the latest end of any job's last operation.
    makespan = model.new_int_var(0, horizon, "makespan")
    last_end = {job.name: end[job.name, len(job.operations) - 1] for job in data.jobs}
    model.add_max_equality(makespan, list(last_end.values()))
    if max_makespan is not None:
        model.add(makespan <= max_makespan)

    # Weighted tardiness, only if every job has a due date.
    tardiness = {}
    total_tardiness = 0
    if all(job.due is not None for job in data.jobs):
        for job in data.jobs:
            # t >= 0 and t >= end - due. Minimizing (or capping the sum)
            # pushes t down to max(0, end - due).
            t = model.new_int_var(0, horizon, f"tardiness {job.name}")
            model.add(t >= last_end[job.name] - job.due)
            tardiness[job.name] = t
        total_tardiness = sum(job.weight * tardiness[job.name] for job in data.jobs)
        if max_tardiness is not None:
            model.add(total_tardiness <= max_tardiness)

    if objective == "makespan":
        model.minimize(makespan)
    elif objective == "tardiness":
        if not tardiness:
            raise ValueError("the tardiness objective needs a due date for every job")
        model.minimize(total_tardiness)
    else:
        raise ValueError(f"unknown objective: {objective}")

    return JobShopModel(model, start, end, makespan, tardiness)


class _ProgressRecorder(cp_model.CpSolverSolutionCallback):
    """Record every improving solution with its time and the best bound."""

    def __init__(self):
        super().__init__()
        self.points: list[tuple[float, int, int]] = []

    def on_solution_callback(self) -> None:
        self.points.append(
            (
                self.wall_time,
                round(self.objective_value),
                round(self.best_objective_bound),
            )
        )


def solve(
    data: JobShopData,
    objective: str = "makespan",
    time_limit: float = 10.0,
    workers: int = 8,
    record_progress: bool = False,
    max_makespan: int | None = None,
    max_tardiness: int | None = None,
) -> Schedule:
    """Build and solve the model. Raise RuntimeError if no schedule is found."""
    jsm = build_model(data, objective, max_makespan, max_tardiness)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = workers

    recorder = _ProgressRecorder() if record_progress else None
    status = solver.solve(jsm.model, recorder)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"No schedule found: {solver.status_name(status)}")
    if recorder:
        # The bound can still rise after the last solution. Add a final point.
        recorder.points.append(
            (
                solver.wall_time,
                round(solver.objective_value),
                round(solver.best_objective_bound),
            )
        )

    return Schedule(
        start={key: solver.value(var) for key, var in jsm.start.items()},
        makespan=solver.value(jsm.makespan),
        objective=round(solver.objective_value),
        best_bound=round(solver.best_objective_bound),
        optimal=status == cp_model.OPTIMAL,
        wall_time=solver.wall_time,
        progress=recorder.points if recorder else [],
    )


def solve_lexicographic(
    data: JobShopData, first: str, second: str, time_limit: float = 10.0
) -> Schedule:
    """Optimize `first`, then optimize `second` without making `first` worse.

    Many schedules often share the best makespan. A second pass picks the
    one among them that is best for the second goal. This also makes the
    reported numbers stable from run to run.
    """
    best = solve(data, first, time_limit)
    cap = {f"max_{first}": best.objective}
    second_best = solve(data, second, time_limit, **cap)
    # Optimal only if both passes proved optimality.
    return Schedule(
        start=second_best.start,
        makespan=second_best.makespan,
        objective=second_best.objective,
        best_bound=second_best.best_bound,
        optimal=best.optimal and second_best.optimal,
        wall_time=best.wall_time + second_best.wall_time,
    )
