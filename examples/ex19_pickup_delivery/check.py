"""Independent checks for patient transport plans. Uses no OR-Tools code.

* `plan_errors` checks every rule of a plan: pairs on one bus, pickup
  first, seats, time windows, shift, and ride times.
* `plan_cost` recomputes the objective from the map.
* `schedule_exists` decides if a bus can drive a given stop order on time.
  With ride-time limits, "drive as early as possible" is not enough: an
  early pickup can make a passenger ride too long. So we solve the timing
  as a small system of difference constraints.
* `exact_optimum` finds the true optimum of a tiny instance by exhaustive
  search over all stop orders.

Distances and times follow the same rounding as the model (whole meters,
whole minutes), but this module computes them on its own.
"""

import itertools
import math

from .data import PdpData

ROAD_FACTOR = 1.3
METERS_PER_KM = 1000


def legs(data: PdpData) -> tuple[list[list[int]], list[list[int]]]:
    """Return (meters, minutes) between all nodes, from the coordinates."""
    nodes = data.stops()
    per_minute = data.speed_kmh * METERS_PER_KM / 60
    meters = [
        [
            round(METERS_PER_KM * ROAD_FACTOR * math.dist((a.x, a.y), (b.x, b.y)))
            for b in nodes
        ]
        for a in nodes
    ]
    minutes = [[round(m / per_minute) for m in row] for row in meters]
    return meters, minutes


def ride_limits(data: PdpData, minutes) -> dict[int, int]:
    """Return the ride-time limit per request (empty if there is none)."""
    if data.max_detour is None:
        return {}
    limits = {}
    for r, req in enumerate(data.requests):
        p, d = 2 * r + 1, 2 * r + 2
        limits[r] = (2 + 2 * req.seats) + minutes[p][d] + data.max_detour
    return limits


def plan_errors(data: PdpData, plan) -> list[str]:
    """Return a list of broken rules. An empty list means the plan is valid."""
    nodes = data.stops()
    meters, minutes = legs(data)
    limits = ride_limits(data, minutes)
    errors = []
    if len(plan.routes) > data.num_vehicles:
        errors.append(f"{len(plan.routes)} routes but only {data.num_vehicles} buses")

    where: dict[int, tuple[int, int]] = {}  # node -> (route, position)
    for k, route in enumerate(plan.routes):
        if route[0] != 0 or route[-1] != 0:
            errors.append(f"route {k} does not start and end at the garage")
        for pos, node in enumerate(route[1:-1], start=1):
            if node in where:
                errors.append(f"{nodes[node].name} is visited twice")
            where[node] = (k, pos)

    for r, req in enumerate(data.requests):
        p, d = 2 * r + 1, 2 * r + 2
        if r in plan.declined:
            if data.decline_penalty is None:
                errors.append(f"{req.name} declined, but declining is not allowed")
            if p in where or d in where:
                errors.append(f"{req.name} is declined but still visited")
            continue
        if p not in where or d not in where:
            errors.append(f"{req.name} is neither fully served nor declined")
            continue
        if where[p][0] != where[d][0]:
            errors.append(f"{req.name}: pickup and drop-off on different buses")
        elif where[p][1] > where[d][1]:
            errors.append(f"{req.name}: drop-off before pickup")

    for k, (route, times) in enumerate(zip(plan.routes, plan.times, strict=True)):
        if len(times) != len(route):
            errors.append(f"route {k}: {len(times)} times for {len(route)} stops")
            continue
        on_board = 0
        for node in route[1:-1]:
            on_board += nodes[node].load
            if not 0 <= on_board <= data.capacity:
                errors.append(
                    f"route {k}: {on_board} people on board at {nodes[node].name}"
                )
        if times[0] < data.shift_start or times[-1] > data.shift_end:
            errors.append(f"route {k}: outside the shift")
        for pos in range(1, len(route)):
            a, b = route[pos - 1], route[pos]
            earliest = times[pos - 1] + nodes[a].service + minutes[a][b]
            if times[pos] < earliest:
                errors.append(
                    f"route {k}: cannot reach {nodes[b].name} by {times[pos]}"
                )
        for pos, node in enumerate(route[1:-1], start=1):
            if not nodes[node].open <= times[pos] <= nodes[node].close:
                errors.append(f"route {k}: {nodes[node].name} outside its window")
        for r, limit in limits.items():
            p, d = 2 * r + 1, 2 * r + 2
            if p in route and d in route:
                ride = times[route.index(d)] - times[route.index(p)]
                if ride > limit:
                    errors.append(
                        f"{data.requests[r].name}: rides {ride} min, limit {limit}"
                    )
    return errors


def plan_cost(data: PdpData, plan) -> int:
    """Return fixed costs + meters driven + taxi penalties."""
    meters, _ = legs(data)
    driven = sum(
        meters[a][b] for route in plan.routes for a, b in itertools.pairwise(route)
    )
    taxis = len(plan.declined) * (data.decline_penalty or 0)
    return driven + data.fixed_cost * len(plan.routes) + taxis


def extra_ride(data: PdpData, plan) -> dict[int, int]:
    """Return, per served request, the minutes beyond a direct trip.

    Ride time runs from the start of boarding to arrival at the drop-off.
    A direct trip takes the boarding time plus the direct drive.
    """
    _, minutes = legs(data)
    extra = {}
    for route, times in zip(plan.routes, plan.times, strict=True):
        position = {node: pos for pos, node in enumerate(route)}
        for r, req in enumerate(data.requests):
            p, d = 2 * r + 1, 2 * r + 2
            if p in position:
                ride = times[position[d]] - times[position[p]]
                extra[r] = ride - (2 + 2 * req.seats) - minutes[p][d]
    return extra


def lifo_errors(data: PdpData, plan) -> list[str]:
    """Return breaches of last-on, first-off order (a stack of parcels)."""
    errors = []
    for k, route in enumerate(plan.routes):
        stack = []
        for node in route[1:-1]:
            r = (node - 1) // 2
            if node % 2 == 1:
                stack.append(r)
            elif not stack or stack.pop() != r:
                errors.append(f"route {k}: {data.requests[r].name} is not on top")
    return errors


# ----------------------------------------------------- timing of one route


def schedule_exists(data: PdpData, route: list[int], minutes) -> bool:
    """Return True if a bus can drive this stop order without breaking a rule.

    Every rule on the timing is a difference constraint of the form
    t_b - t_a <= w: travel (t_next >= t_prev + gap), windows (via a fixed
    zero node), the shift, and ride times (t_drop - t_pick <= limit). Such a
    system has a solution exactly when its constraint graph has no negative
    cycle (a classic result). Bellman-Ford finds out.
    """
    nodes = data.stops()
    limits = ride_limits(data, minutes)
    m = len(route)  # Positions 0..m-1; position m is the zero node.
    zero = m
    edges = []  # (a, b, w) means t_b - t_a <= w.

    def at_most(a, b, w):
        edges.append((a, b, w))

    # Shift: leave no earlier than the start, return no later than the end.
    at_most(0, zero, -data.shift_start)  # t_zero - t_0 <= -start.
    at_most(zero, m - 1, data.shift_end)  # t_end - t_zero <= end.
    for pos in range(1, m):
        a, b = route[pos - 1], route[pos]
        gap = nodes[a].service + minutes[a][b]
        at_most(pos, pos - 1, -gap)  # t_prev - t_next <= -gap.
    for pos in range(1, m - 1):
        node = nodes[route[pos]]
        at_most(zero, pos, node.close)
        at_most(pos, zero, -node.open)
    position = {node: pos for pos, node in enumerate(route)}
    for r, limit in limits.items():
        p, d = 2 * r + 1, 2 * r + 2
        if p in position and d in position:
            at_most(position[p], position[d], limit)

    dist = [0] * (m + 1)  # A virtual source reaches every node at 0.
    for _ in range(m + 1):
        changed = False
        for a, b, w in edges:
            if dist[a] + w < dist[b]:
                dist[b] = dist[a] + w
                changed = True
        if not changed:
            return True
    return False


def best_route(data: PdpData, requests: tuple[int, ...], minutes, meters) -> int | None:
    """Return the shortest feasible route (meters) serving exactly these requests.

    Depth-first search over all stop orders with each pickup before its
    drop-off. It prunes orders that overfill the bus or miss a window even
    when driving as early as possible. Complete orders get the exact
    timing test.
    """
    nodes = data.stops()
    best = math.inf
    todo = {2 * r + 1 for r in requests}

    def search(route, available, on_board, earliest, length):
        nonlocal best
        if length >= best:
            return
        if not available:
            full = [*route, 0]
            total = length + meters[route[-1]][0]
            if total < best and schedule_exists(data, full, minutes):
                best = total
            return
        last = route[-1]
        for node in sorted(available):
            load = on_board + nodes[node].load
            if load > data.capacity:
                continue
            t = max(
                earliest + nodes[last].service + minutes[last][node], nodes[node].open
            )
            if t > nodes[node].close:
                continue
            nxt = set(available) - {node}
            if node % 2 == 1:
                nxt.add(node + 1)  # The drop-off becomes possible.
            search([*route, node], nxt, load, t, length + meters[last][node])

    search([0], todo, 0, data.shift_start, 0)
    return None if best == math.inf else best


def exact_optimum(data: PdpData) -> int:
    """Return the optimal objective of a tiny instance by exhaustive search.

    1. For every set of requests, find the shortest single-bus route.
    2. Split the served requests into at most `num_vehicles` such sets, in
       the cheapest way (dynamic programming over subsets).
    3. If declining is allowed, also try every set of declined requests.
    """
    meters, minutes = legs(data)
    count = len(data.requests)
    route_cost = {}
    for mask in range(1, 1 << count):
        reqs = tuple(r for r in range(count) if mask >> r & 1)
        length = best_route(data, reqs, minutes, meters)
        if length is not None:
            route_cost[mask] = length + data.fixed_cost

    # cover[k][mask]: cheapest way to serve `mask` with at most k buses.
    cover = [{0: 0}]
    for _ in range(data.num_vehicles):
        prev, cur = cover[-1], dict(cover[-1])
        for mask in range(1, 1 << count):
            low = mask & -mask
            sub = mask
            while sub:
                if sub & low and sub in route_cost and (mask ^ sub) in prev:
                    value = route_cost[sub] + prev[mask ^ sub]
                    if value < cur.get(mask, math.inf):
                        cur[mask] = value
                sub = (sub - 1) & mask
        cover.append(cur)
    best = cover[-1]

    everyone = (1 << count) - 1
    if data.decline_penalty is None:
        if everyone not in best:
            raise ValueError("no feasible plan serves every request")
        return best[everyone]
    return min(
        value + data.decline_penalty * (count - bin(mask).count("1"))
        for mask, value in best.items()
    )
