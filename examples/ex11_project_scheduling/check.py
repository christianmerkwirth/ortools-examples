"""Independent checks and bounds for project schedules. Uses no OR-Tools code.

* `feasibility_errors` checks durations, precedences, and resource limits
  on every single day.
* `critical_path_length` and `energy_bound` are lower bounds: no schedule
  can be shorter than either of them.
* `serial_schedule` is the classic serial schedule generation scheme. It
  turns any priority list of activities into a feasible schedule. A
  simple priority rule gives a quick heuristic, and trying every list
  solves tiny instances exactly.
"""

import itertools
import math

from .data import ProjectData

Schedule = dict[str, int]  # Activity name -> start day (day 0 is the first).


def makespan(data: ProjectData, start: Schedule) -> int:
    """Return the day the last activity ends."""
    return max(start[a.name] + a.duration for a in data.activities)


def feasibility_errors(data: ProjectData, start: Schedule) -> list[str]:
    """Return a list of broken rules. An empty list means feasible."""
    errors = []
    names = {a.name for a in data.activities}
    if set(start) != names:
        return [f"schedule covers {sorted(set(start) ^ names)} wrongly"]
    end = {a.name: start[a.name] + a.duration for a in data.activities}
    for a in data.activities:
        if start[a.name] < 0:
            errors.append(f"{a.name} starts before day 0")
        for p in a.predecessors:
            if start[a.name] < end[p]:
                errors.append(
                    f"{a.name} starts on day {start[a.name]}, before {p} ends"
                )
    for r in data.resources:
        usage = resource_profile(data, start, r.name)
        for day, used in enumerate(usage):
            if used > r.capacity:
                errors.append(f"{r.name}: {used} used on day {day}, only {r.capacity}")
    return errors


def resource_profile(data: ProjectData, start: Schedule, resource: str) -> list[int]:
    """Return the units of one resource in use on each day of the schedule."""
    usage = [0] * makespan(data, start)
    for a in data.activities:
        for day in range(start[a.name], start[a.name] + a.duration):
            usage[day] += a.needs(resource)
    return usage


def topological_order(data: ProjectData) -> list[str]:
    """Return activity names so that every predecessor comes first."""
    order, done = [], set()
    pending = list(data.activities)
    while pending:
        ready = [a for a in pending if set(a.predecessors) <= done]
        if not ready:
            raise ValueError("the precedence graph has a cycle")
        for a in ready:
            order.append(a.name)
            done.add(a.name)
        pending = [a for a in pending if a.name not in done]
    return order


def earliest_starts(data: ProjectData) -> Schedule:
    """Return the earliest start of each activity if resources were unlimited."""
    es: Schedule = {}
    for name in topological_order(data):
        a = data.activity(name)
        es[name] = max(
            (es[p] + data.activity(p).duration for p in a.predecessors), default=0
        )
    return es


def critical_path_length(data: ProjectData) -> int:
    """Return the project length with unlimited resources (a lower bound)."""
    return makespan(data, earliest_starts(data))


def critical_activities(data: ProjectData) -> list[str]:
    """Return the activities that have no slack when resources are unlimited."""
    es = earliest_starts(data)
    length = makespan(data, es)
    # Latest finish: go backward from the end.
    lf = {a.name: length for a in data.activities}
    for name in reversed(topological_order(data)):
        a = data.activity(name)
        for p in a.predecessors:
            lf[p] = min(lf[p], lf[name] - a.duration)
    return [
        name
        for name in topological_order(data)
        if lf[name] - data.activity(name).duration == es[name]
    ]


def energy_bound(data: ProjectData) -> int:
    """Return a lower bound from resource work alone.

    A resource with capacity c can deliver at most c unit-days per day. If
    the activities need W unit-days of it in total, the project lasts at
    least ceil(W / c) days.
    """
    return max(
        math.ceil(
            sum(a.duration * a.needs(r.name) for a in data.activities) / r.capacity
        )
        for r in data.resources
    )


def serial_schedule(data: ProjectData, priority: list[str]) -> Schedule:
    """Turn a priority list into a feasible schedule (serial SGS).

    Take the eligible activity (all predecessors scheduled) that comes first
    in the list. Start it on the earliest day that respects its
    predecessors and leaves enough of every resource for its whole
    duration. Repeat until all are scheduled.
    """
    for a in data.activities:
        for r in data.resources:
            if a.needs(r.name) > r.capacity:
                raise ValueError(f"{a.name} needs more {r.name} than exist")
    rank = {name: k for k, name in enumerate(priority)}
    # Doing everything one after another always fits in this many days.
    horizon = sum(a.duration for a in data.activities)
    free = {r.name: [r.capacity] * horizon for r in data.resources}
    start: Schedule = {}
    while len(start) < len(data.activities):
        eligible = [
            a
            for a in data.activities
            if a.name not in start and all(p in start for p in a.predecessors)
        ]
        a = min(eligible, key=lambda x: rank[x.name])
        t = max(
            (start[p] + data.activity(p).duration for p in a.predecessors), default=0
        )
        while not all(
            free[r][day] >= a.needs(r) for r in free for day in range(t, t + a.duration)
        ):
            t += 1
        for r in free:
            for day in range(t, t + a.duration):
                free[r][day] -= a.needs(r)
        start[a.name] = t
    return start


def latest_finish_rule(data: ProjectData) -> Schedule:
    """Return a quick heuristic schedule: serial SGS, smallest slack first.

    The priority is the latest finish time from the critical path
    analysis, a classic and robust rule.
    """
    es = earliest_starts(data)
    length = makespan(data, es)
    lf = {a.name: length for a in data.activities}
    for name in reversed(topological_order(data)):
        a = data.activity(name)
        for p in a.predecessors:
            lf[p] = min(lf[p], lf[name] - a.duration)
    order = sorted(topological_order(data), key=lambda n: (lf[n], es[n]))
    return serial_schedule(data, order)


def brute_force_makespan(data: ProjectData) -> int:
    """Return the optimal makespan by trying every priority list.

    For makespan, some optimal schedule is "active": no activity can start
    earlier without delaying another. The serial SGS creates exactly the
    active schedules, so trying all lists finds the optimum. Only for tiny
    instances: n activities give n! lists.
    """
    names = [a.name for a in data.activities]
    return min(
        makespan(data, serial_schedule(data, list(order)))
        for order in itertools.permutations(names)
    )
