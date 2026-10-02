"""Independent checks for job shop schedules. Uses no OR-Tools code.

* `feasibility_errors` checks every rule of a schedule.
* `lower_bound` gives simple bounds that no schedule can beat. If the
  makespan equals a bound, the schedule is optimal by itself.
* `critical_path` finds the chain of operations that sets the makespan.
* `brute_force_makespan` tries every machine order. Tiny instances only.
"""

import itertools
import math

from .data import JobShopData

OpKey = tuple[str, int]


def end_time(data: JobShopData, start: dict[OpKey, int], key: OpKey) -> int:
    """Return the end time of one operation."""
    job, k = key
    return start[key] + data.job(job).operations[k].duration


def makespan(data: JobShopData, start: dict[OpKey, int]) -> int:
    """Return the end time of the last operation."""
    return max(end_time(data, start, key) for key in start)


def weighted_tardiness(data: JobShopData, start: dict[OpKey, int]) -> int:
    """Return the sum over jobs of weight x minutes late."""
    total = 0
    for job in data.jobs:
        finish = end_time(data, start, (job.name, len(job.operations) - 1))
        total += job.weight * max(0, finish - job.due)
    return total


def feasibility_errors(data: JobShopData, start: dict[OpKey, int]) -> list[str]:
    """Return a list of broken rules. An empty list means feasible."""
    errors = []
    expected = {(j.name, k) for j in data.jobs for k in range(len(j.operations))}
    if set(start) != expected:
        errors.append("the schedule does not cover exactly the operations of the data")
        return errors

    for job in data.jobs:
        for k in range(len(job.operations)):
            if start[job.name, k] < 0:
                errors.append(f"{job.name} #{k} starts before time 0")
            if k > 0 and start[job.name, k] < end_time(data, start, (job.name, k - 1)):
                errors.append(f"{job.name} #{k} starts before #{k - 1} ends")

    for machine in data.machines:
        ops = sorted(
            (start[j.name, k], end_time(data, start, (j.name, k)), j.name, k)
            for j in data.jobs
            for k, op in enumerate(j.operations)
            if op.machine == machine
        )
        for (_, end_a, ja, ka), (start_b, _, jb, kb) in itertools.pairwise(ops):
            if start_b < end_a:
                errors.append(f"{machine}: {ja} #{ka} overlaps {jb} #{kb}")
    return errors


def lower_bound(data: JobShopData) -> dict[str, int]:
    """Return three makespan lower bounds, from weak to strong.

    * job: the longest route. A job cannot finish faster than its own
      operations back to back.
    * machine: the busiest machine. It needs at least its total load.
    * machine + head + tail: before a machine's first operation, some job
      must reach it (head). After its last, some job must finish its route
      (tail). Each machine needs at least min head + load + min tail.
    """
    job_bound = max(sum(op.duration for op in j.operations) for j in data.jobs)
    machine_bound = 0
    head_tail_bound = 0
    for m in data.machines:
        load, heads, tails = 0, [], []
        for j in data.jobs:
            for k, op in enumerate(j.operations):
                if op.machine == m:
                    load += op.duration
                    heads.append(sum(o.duration for o in j.operations[:k]))
                    tails.append(sum(o.duration for o in j.operations[k + 1 :]))
        if heads:
            machine_bound = max(machine_bound, load)
            head_tail_bound = max(head_tail_bound, min(heads) + load + min(tails))
    return {
        "longest job": job_bound,
        "busiest machine": machine_bound,
        "machine + head + tail": head_tail_bound,
    }


def critical_path(data: JobShopData, start: dict[OpKey, int]) -> list[OpKey]:
    """Return a chain of operations with no idle time that ends at the makespan.

    Walk back from an operation that ends at the makespan. At each step,
    find a predecessor (the previous operation of the same job, or any
    operation on the same machine) that ends exactly when the current one
    starts. Delaying any operation on this chain delays the whole schedule.
    """
    machine_of = {
        (j.name, k): op.machine for j in data.jobs for k, op in enumerate(j.operations)
    }
    span = makespan(data, start)
    current = next(key for key in start if end_time(data, start, key) == span)
    path = [current]
    while start[current] > 0:
        job, k = current
        candidates = [(job, k - 1)] if k > 0 else []
        candidates += [key for key in start if machine_of[key] == machine_of[current]]
        previous = next(
            (
                key
                for key in candidates
                if key != current and end_time(data, start, key) == start[current]
            ),
            None,
        )
        if previous is None:
            break  # Idle time before this operation: the chain ends here.
        path.append(previous)
        current = previous
    return path[::-1]


def brute_force_makespan(data: JobShopData) -> int:
    """Return the optimal makespan by trying every order on every machine.

    For fixed machine orders, the earliest start of each operation follows
    from its two predecessors (job and machine). Some order combinations
    contain a cycle and have no schedule; we skip them. The work grows as
    (jobs!)^machines, so use this only for tiny instances.
    """
    ops_on = {
        m: [
            (j.name, k)
            for j in data.jobs
            for k, op in enumerate(j.operations)
            if op.machine == m
        ]
        for m in data.machines
    }
    duration = {
        (j.name, k): op.duration for j in data.jobs for k, op in enumerate(j.operations)
    }
    best = math.inf
    for orders in itertools.product(
        *(itertools.permutations(ops) for ops in ops_on.values())
    ):
        preds = {key: [] for key in duration}
        for order in orders:
            for a, b in itertools.pairwise(order):
                preds[b].append(a)
        for job, k in duration:
            if k > 0:
                preds[job, k].append((job, k - 1))
        start = _earliest_starts(preds, duration)
        if start is not None:
            best = min(best, max(start[key] + duration[key] for key in duration))
    return int(best)


def _earliest_starts(preds, duration):
    """Longest-path start times in a precedence graph, or None on a cycle."""
    start: dict[OpKey, int] = {}
    remaining = set(duration)
    while remaining:
        ready = [key for key in remaining if all(p in start for p in preds[key])]
        if not ready:
            return None  # Cycle: these machine orders contradict the routes.
        for key in ready:
            start[key] = max((start[p] + duration[p] for p in preds[key]), default=0)
            remaining.discard(key)
    return start
