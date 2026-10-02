"""Assignment models: a specialized solver first, then CP-SAT for real rules.

Part A is the classic linear assignment problem: n technicians, n jobs,
each technician takes exactly one job, minimize total minutes.

* `solve_morning` uses OR-Tools' `SimpleLinearSumAssignment`. It solves
  this one problem class very fast and always to proven optimality.
* `solve_morning_cp_sat` writes the same problem in CP-SAT, as a cross-check
  and as the starting point for Part B.

Part B (`solve_day`) breaks the classic structure: a technician may take
several jobs within their shift, and some jobs need two people. This is a
generalized assignment problem. It is NP-hard, and the specialized solver
cannot express it. CP-SAT can.
"""

from dataclasses import dataclass

from ortools.graph.python import linear_sum_assignment
from ortools.sat.python import cp_model

from .data import AssignmentData


class NoAssignmentError(RuntimeError):
    """Raised when no assignment meets all rules."""


@dataclass(frozen=True)
class Plan:
    """A solution: who does which job."""

    crew: dict[str, tuple[str, ...]]  # Job name -> technician names.
    minutes: int  # Total working minutes, travel included.
    proven_optimal: bool

    def jobs_of(self, tech: str) -> list[str]:
        """Return the jobs of one technician, in data order."""
        return [j for j, crew in self.crew.items() if tech in crew]


# ---------------------------------------------------------------- Part A ---


def solve_morning(data: AssignmentData) -> Plan:
    """Solve a square instance with the linear sum assignment solver."""
    if len(data.technicians) != len(data.jobs):
        raise ValueError("The linear assignment solver needs a square instance.")

    solver = linear_sum_assignment.SimpleLinearSumAssignment()
    # The solver works on a bipartite graph: left nodes are technicians,
    # right nodes are jobs. We add an arc only where the technician is
    # qualified. A missing arc means "not allowed"; no big-M cost needed.
    for t, tech in enumerate(data.technicians):
        for j, job in enumerate(data.jobs):
            cost = data.cost(tech, job)
            if cost is not None:
                solver.add_arc_with_cost(t, j, cost)

    status = solver.solve()
    if status == solver.INFEASIBLE:
        raise NoAssignmentError("No way to give every technician a job they can do.")
    if status != solver.OPTIMAL:
        raise RuntimeError(f"Assignment solver failed with status {status}.")

    crew = {
        data.jobs[solver.right_mate(t)].name: (data.technicians[t].name,)
        for t in range(solver.num_nodes())
    }
    return Plan(crew, solver.optimal_cost(), proven_optimal=True)


def solve_morning_cp_sat(data: AssignmentData, time_limit: float = 10.0) -> Plan:
    """Solve the same square instance with CP-SAT."""
    model = cp_model.CpModel()
    x = _pair_variables(model, data)
    for pairs in _group(x, by=0).values():  # Each technician: one job.
        model.add_exactly_one(pairs.values())
    for pairs in _group(x, by=1).values():  # Each job: one technician.
        model.add_exactly_one(pairs.values())
    return _solve(model, x, data, time_limit)


# ---------------------------------------------------------------- Part B ---


def solve_day(
    data: AssignmentData,
    use_shifts: bool = True,
    use_teams: bool = True,
    time_limit: float = 10.0,
) -> Plan:
    """Plan a full day: several jobs per technician, shifts, and team jobs.

    The two switches turn rules off. This lets us measure what each rule
    costs in minutes.
    """
    model = cp_model.CpModel()
    x = _pair_variables(model, data)

    by_job = _group(x, by=1)
    by_tech = _group(x, by=0)

    # Every job gets exactly as many people as it needs.
    for job in data.jobs:
        need = job.team_size if use_teams else 1
        model.add(sum(by_job[job.name].values()) == need)

    # Nobody works longer than their shift. On a team job, each member
    # spends the full time, so each member's own minutes count.
    if use_shifts:
        cost = _costs(data)
        for tech in data.technicians:
            pairs = by_tech.get(tech.name, {})
            model.add(sum(cost[p] * v for p, v in pairs.items()) <= tech.shift)
    return _solve(model, x, data, time_limit)


# --------------------------------------------------------------- helpers ---


def _pair_variables(model: cp_model.CpModel, data: AssignmentData):
    """Create one Boolean per qualified (technician, job) pair.

    Unqualified pairs get no variable at all. This is both smaller and
    clearer than a variable that we then force to zero.
    """
    return {(t, j): model.new_bool_var(f"{t} does {j}") for t, j in _costs(data)}


def _group(x: dict, by: int) -> dict[str, dict]:
    """Group pair variables by technician (by=0) or by job (by=1)."""
    groups: dict[str, dict] = {}
    for pair, var in x.items():
        groups.setdefault(pair[by], {})[pair] = var
    return groups


def _costs(data: AssignmentData) -> dict[tuple[str, str], int]:
    """Return minutes for every qualified (technician, job) pair."""
    return {
        (t.name, j.name): data.cost(t, j)
        for t in data.technicians
        for j in data.jobs
        if data.cost(t, j) is not None
    }


def _solve(model, x, data: AssignmentData, time_limit: float) -> Plan:
    cost = _costs(data)
    model.minimize(sum(cost[pair] * v for pair, v in x.items()))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 8
    status = solver.solve(model)
    if status == cp_model.INFEASIBLE:
        raise NoAssignmentError("No plan meets all rules.")
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(
            f"CP-SAT found no plan (status {solver.status_name(status)})."
        )

    crew: dict[str, tuple[str, ...]] = {j.name: () for j in data.jobs}
    for (t, j), v in x.items():
        if solver.boolean_value(v):
            crew[j] += (t,)
    return Plan(crew, round(solver.objective_value), status == cp_model.OPTIMAL)
