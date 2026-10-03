"""Independent checks for delivery plans. Uses no OR-Tools code.

* `plan_errors` checks every rule. It replays each route minute by minute:
  drive, wait if the window is not open yet, unload, drive on.
* `plan_cost` recomputes the cost from the map coordinates.
* `exact_optimum` finds the true optimum of a tiny instance by exhaustive
  search. It is slow, but it shares no code with either solver.
* `simple_lower_bound` gives a bound that no plan can beat.

Distances and driving times follow the same rounding rules as the model
(whole meters, whole minutes), but this module computes them on its own.
"""

import itertools
import math

from .data import VrpData

ROAD_FACTOR = 1.3
METERS_PER_KM = 1000


def legs(data: VrpData) -> tuple[list[list[int]], list[list[int]]]:
    """Return (meters, minutes) between all stops, computed from coordinates."""
    meters, minutes = [], []
    per_minute = data.speed_kmh * METERS_PER_KM / 60
    for a in data.stops:
        row_m, row_t = [], []
        for b in data.stops:
            d = round(METERS_PER_KM * ROAD_FACTOR * math.dist((a.x, a.y), (b.x, b.y)))
            row_m.append(d)
            row_t.append(round(d / per_minute))
        meters.append(row_m)
        minutes.append(row_t)
    return meters, minutes


def replay(
    data: VrpData, route: list[int], minutes, leave: int | None = None
) -> tuple[list[int], int] | None:
    """Drive a route as early as possible. Return (service starts, return time).

    The van leaves at `leave` (default: the shift start), waits whenever it
    is early, and unloads. Return None if a window closes before the van
    gets there. Leaving at the shift start is never worse for later
    windows, so if that replay fails, no departure time can work.
    """
    now = data.shift_start if leave is None else leave
    starts = []
    prev = 0
    for stop in route[1:-1]:
        s = data.stops[stop]
        now = max(now + minutes[prev][stop], s.open)
        if now > s.close:
            return None
        starts.append(now)
        now += s.service
        prev = stop
    return starts, now + minutes[prev][0]


def shortest_span(data: VrpData, route: list[int], minutes) -> int | None:
    """Return the shortest working time (leave to return) of a route.

    If the van leaves one minute later, it comes back at most one minute
    later (it may simply wait one minute less). So the working time never
    grows when the van leaves later, and the best plan leaves as late as
    possible. Leaving later only ever breaks a route, never repairs it, so
    a binary search finds the latest departure that still works.
    Return None if the route does not work at all.
    """

    def works(leave: int) -> bool:
        result = replay(data, route, minutes, leave)
        return result is not None and result[1] <= data.shift_end

    if not works(data.shift_start):
        return None
    lo, hi = data.shift_start, data.shift_end  # works(lo) is True.
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if works(mid):
            lo = mid
        else:
            hi = mid - 1
    return replay(data, route, minutes, lo)[1] - lo


def plan_errors(data: VrpData, plan) -> list[str]:
    """Return a list of broken rules. An empty list means feasible."""
    errors = []
    meters, minutes = legs(data)
    seen = []
    if len(plan.routes) > data.num_vehicles:
        errors.append(f"{len(plan.routes)} routes but only {data.num_vehicles} vans")
    for k, (route, times) in enumerate(zip(plan.routes, plan.start_times, strict=True)):
        name = f"route {k + 1}"
        if route[0] != 0 or route[-1] != 0 or 0 in route[1:-1]:
            errors.append(f"{name}: must start and end at the depot only")
            continue
        seen += route[1:-1]
        load = sum(data.stops[c].demand for c in route[1:-1])
        if load > data.capacity:
            errors.append(f"{name}: carries {load} crates > {data.capacity}")

        # 1. The route must be feasible at all (earliest replay).
        result = replay(data, route, minutes)
        if result is None:
            errors.append(f"{name}: a time window closes before the van arrives")
        elif result[1] > data.shift_end:
            errors.append(f"{name}: back at {result[1]} after the shift end")

        # 2. The solver's own timetable must also hold.
        if times[0] < data.shift_start or times[-1] > data.shift_end:
            errors.append(f"{name}: timetable leaves the shift")
        for (a, ta), (b, tb) in itertools.pairwise(zip(route, times, strict=True)):
            if b != 0:
                s = data.stops[b]
                if not s.open <= tb <= s.close:
                    errors.append(f"{name}: {s.name} starts at {tb}, outside window")
            if tb < ta + data.stops[a].service + minutes[a][b]:
                errors.append(f"{name}: cannot reach stop {b} by {tb}")

    if len(seen) != len(set(seen)):
        errors.append("a customer is visited twice")
    missing = set(data.customers) - set(seen)
    if missing != set(plan.dropped):
        errors.append(f"unserved {sorted(missing)} != reported {sorted(plan.dropped)}")
    if missing and data.drop_penalty_per_crate is None:
        errors.append(
            f"customers {sorted(missing)} unserved, but drops are not allowed"
        )
    if plan_cost(data, plan) != plan.cost:
        errors.append(f"reported cost {plan.cost} != computed {plan_cost(data, plan)}")
    return errors


def plan_cost(data: VrpData, plan) -> int:
    """Return fixed costs + meters driven + drop penalties + paid time.

    Paid time uses the plan's own timetable (checked by `plan_errors`).
    """
    meters, _ = legs(data)
    driven = sum(meters[a][b] for r in plan.routes for a, b in itertools.pairwise(r))
    penalty = sum(data.drop_penalty(c) or 0 for c in plan.dropped)
    working = sum(times[-1] - times[0] for times in plan.start_times)
    return (
        data.fixed_cost * len(plan.routes)
        + driven
        + penalty
        + data.minute_cost * working
    )


def exact_optimum(data: VrpData) -> int:
    """Return the optimal cost of a tiny instance by exhaustive search.

    Step 1: depth-first search over every feasible route (respecting
    windows, shift, and capacity). For each set of customers, keep the
    cheapest route that serves exactly that set (distance, plus paid time
    at the best departure time).

    Step 2: dynamic programming over sets. best[k][S] is the cheapest way
    to serve the set S with at most k vans. Then add drop penalties for the
    customers outside S, and take the best S.

    Use it for up to about 9 customers.
    """
    n = len(data.stops) - 1
    meters, minutes = legs(data)
    route_cost: dict[int, int] = {}

    def extend(last, now, mask, load, dist, path) -> None:
        back = now + minutes[last][0]
        if mask and back <= data.shift_end:
            total = dist + meters[last][0]
            if data.minute_cost:
                span = shortest_span(data, [0, *path, 0], minutes)
                total += data.minute_cost * span
            if total < route_cost.get(mask, math.inf):
                route_cost[mask] = total
        for c in range(1, n + 1):
            bit = 1 << (c - 1)
            s = data.stops[c]
            if mask & bit or load + s.demand > data.capacity:
                continue
            start = max(now + minutes[last][c], s.open)
            if start <= s.close:
                extend(
                    c,
                    start + s.service,
                    mask | bit,
                    load + s.demand,
                    dist + meters[last][c],
                    [*path, c],
                )

    extend(0, data.shift_start, 0, 0, 0, [])

    full = (1 << n) - 1
    best = {0: 0}  # With 0 vans, only the empty set is possible.
    for _ in range(data.num_vehicles):
        nxt = dict(best)
        for served, cost in best.items():
            rest = full ^ served
            sub = rest
            while sub:
                if sub in route_cost:
                    new = served | sub
                    value = cost + data.fixed_cost + route_cost[sub]
                    if value < nxt.get(new, math.inf):
                        nxt[new] = value
                sub = (sub - 1) & rest
        best = nxt

    answer = math.inf
    for served, cost in best.items():
        unserved = [c for c in range(1, n + 1) if not served & (1 << (c - 1))]
        if unserved and data.drop_penalty_per_crate is None:
            continue
        answer = min(answer, cost + sum(data.drop_penalty(c) for c in unserved))
    if answer == math.inf:
        raise ValueError("instance is infeasible")
    return int(answer)


def simple_lower_bound(data: VrpData) -> tuple[int, int]:
    """Return (vans needed, cost bound) for an instance where all are served.

    Vans: the crates must fit, so at least ceil(total demand / capacity).
    Distance: every customer is entered exactly once, so the total is at
    least the sum of each customer's shortest incoming leg. Every van also
    drives back to the depot once, over at least the shortest leg into it.
    """
    meters, _ = legs(data)
    demand = sum(s.demand for s in data.stops)
    vans = math.ceil(demand / data.capacity)
    into = sum(
        min(meters[i][j] for i in range(len(data.stops)) if i != j)
        for j in data.customers
    )
    home = min(meters[j][0] for j in data.customers)
    return vans, vans * data.fixed_cost + into + vans * home
