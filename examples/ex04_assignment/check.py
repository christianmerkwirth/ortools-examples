"""Independent checks for assignment plans. Uses no OR-Tools code.

* `feasibility_errors` checks every rule of a plan.
* `hungarian` is a small textbook implementation of the Hungarian method.
  Besides the optimal cost, it returns dual potentials u (technicians) and
  v (jobs).
* `certificate_errors` uses those potentials as a proof of optimality:
  if u_t + v_j <= cost[t][j] for every allowed pair, then every complete
  assignment costs at least sum(u) + sum(v). If our plan costs exactly
  that, no plan is cheaper.
* `brute_force_*` try every plan. Only for small instances.
"""

import itertools
import math

from .data import AssignmentData


def plan_minutes(data: AssignmentData, crew: dict[str, tuple[str, ...]]) -> int:
    """Return the total minutes of a plan, recomputed from the data."""
    techs = {t.name: t for t in data.technicians}
    return sum(
        data.cost(techs[name], job)
        for job in data.jobs
        for name in crew.get(job.name, ())
    )


def feasibility_errors(
    data: AssignmentData,
    crew: dict[str, tuple[str, ...]],
    one_job_each: bool = False,
    use_shifts: bool = True,
    use_teams: bool = True,
) -> list[str]:
    """Return a list of broken rules. An empty list means feasible.

    With `one_job_each`, check Part A's rule: every technician does exactly
    one job and every job has exactly one technician. Otherwise check Part
    B's rules: team sizes and shift limits.
    """
    errors = []
    techs = {t.name: t for t in data.technicians}
    jobs = {j.name: j for j in data.jobs}
    if set(crew) != set(jobs):
        errors.append(f"jobs in plan {sorted(crew)} != jobs in data {sorted(jobs)}")
        return errors

    for job_name, names in crew.items():
        job = jobs[job_name]
        if len(set(names)) != len(names):
            errors.append(f"{job_name}: a technician is listed twice")
        for name in names:
            if name not in techs:
                errors.append(f"{job_name}: unknown technician {name}")
            elif data.cost(techs[name], job) is None:
                errors.append(f"{name} is not qualified for {job_name} ({job.trade})")
        need = 1 if one_job_each or not use_teams else job.team_size
        if len(names) != need:
            errors.append(f"{job_name}: has {len(names)} technicians, needs {need}")

    load = dict.fromkeys(techs, 0)
    count = dict.fromkeys(techs, 0)
    for job_name, names in crew.items():
        for name in names:
            if name in techs and data.cost(techs[name], jobs[job_name]) is not None:
                load[name] += data.cost(techs[name], jobs[job_name])
                count[name] += 1
    for name, tech in techs.items():
        if one_job_each and count[name] != 1:
            errors.append(f"{name}: has {count[name]} jobs, needs exactly 1")
        if not one_job_each and use_shifts and load[name] > tech.shift:
            errors.append(f"{name}: works {load[name]} min, shift is {tech.shift}")
    return errors


def hungarian(
    cost: list[list[int | None]],
) -> tuple[int, list[int], list[int], list[int]]:
    """Solve a square assignment problem with the Hungarian method.

    `cost[t][j]` is None where the pair is not allowed. Returns
    (total cost, job index per row, row potentials u, column potentials v).

    This is the O(n³) shortest augmenting path version (Kuhn-Munkres with
    potentials). It keeps u[t] + v[j] <= cost[t][j] for every pair at all
    times, and it ends with equality on the chosen pairs.
    """
    n = len(cost)
    finite = [c for row in cost for c in row if c is not None]
    big = (sum(finite) + 1) * (n + 1)  # Larger than any real plan.
    a = [[big if c is None else c for c in row] for row in cost]

    # 1-based arrays; index 0 is a helper "virtual" column.
    u = [0] * (n + 1)
    v = [0] * (n + 1)
    row_of = [0] * (n + 1)  # row_of[j]: the row assigned to column j.
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        row_of[0] = i
        j0 = 0
        min_slack = [math.inf] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = row_of[j0], math.inf, 0
            for j in range(1, n + 1):
                if not used[j]:
                    slack = a[i0 - 1][j - 1] - u[i0] - v[j]
                    if slack < min_slack[j]:
                        min_slack[j], way[j] = slack, j0
                    if min_slack[j] < delta:
                        delta, j1 = min_slack[j], j
            for j in range(n + 1):
                if used[j]:
                    u[row_of[j]] += delta
                    v[j] -= delta
                else:
                    min_slack[j] -= delta
            j0 = j1
            if row_of[j0] == 0:
                break
        while j0:  # Flip the augmenting path.
            j1 = way[j0]
            row_of[j0] = row_of[j1]
            j0 = j1

    job_of = [0] * n
    for j in range(1, n + 1):
        job_of[row_of[j] - 1] = j - 1
    total = sum(a[t][job_of[t]] for t in range(n))
    if total >= big:
        raise ValueError("No complete assignment uses only allowed pairs.")
    return total, job_of, u[1:], v[1:]


def certificate_errors(
    cost: list[list[int | None]], plan_cost: int, u: list[float], v: list[float]
) -> list[str]:
    """Check that potentials (u, v) prove `plan_cost` is optimal.

    For any complete assignment, add up u_t + v_j <= cost[t][j] over its n
    pairs. Each u_t and each v_j appears exactly once, so the assignment
    costs at least sum(u) + sum(v). Pairs that are not allowed never appear
    in an assignment, so they need no check.
    """
    errors = []
    for t, row in enumerate(cost):
        for j, c in enumerate(row):
            if c is not None and u[t] + v[j] > c + 1e-9:
                errors.append(f"pair ({t}, {j}): u + v = {u[t] + v[j]} > cost {c}")
    bound = sum(u) + sum(v)
    if abs(bound - plan_cost) > 1e-9:
        errors.append(f"dual bound {bound} != plan cost {plan_cost}")
    return errors


def brute_force_morning(data: AssignmentData) -> int:
    """Try every permutation (n! plans). Use for n <= 9."""
    cost = data.cost_matrix()
    n = len(cost)
    best = math.inf
    for perm in itertools.permutations(range(n)):
        if all(cost[t][perm[t]] is not None for t in range(n)):
            best = min(best, sum(cost[t][perm[t]] for t in range(n)))
    return best


def brute_force_day(data: AssignmentData) -> int:
    """Try every crew for every job. Tiny instances only."""
    techs = [t.name for t in data.technicians]
    options = [list(itertools.combinations(techs, job.team_size)) for job in data.jobs]
    best = math.inf
    for choice in itertools.product(*options):
        crew = {job.name: names for job, names in zip(data.jobs, choice, strict=True)}
        if not feasibility_errors(data, crew):
            best = min(best, plan_minutes(data, crew))
    return best
