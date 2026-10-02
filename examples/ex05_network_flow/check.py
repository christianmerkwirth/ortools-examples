"""Independent checks for network flows. Uses no OR-Tools code.

* `flow_errors` checks capacities, flow conservation, and the totals.
* `cut_capacity` adds up the capacity of the arcs that cross a cut. No flow
  can be larger than any cut. So if a flow equals a cut, both are optimal
  (the max-flow min-cut theorem).
* `has_negative_cycle` looks for a cycle of negative cost in the residual
  network. A flow is cheapest among flows of its size if and only if no
  such cycle exists. (A negative cycle would be a way to reroute pallets
  and save money.)
* `marginal_costs` gives node potentials: the cost of one more pallet at
  each node.
"""

import math

from .data import NetworkData

SOURCE, SINK = "SOURCE", "SINK"

Arc = tuple[str, str, int, int]  # (tail, head, capacity, unit cost)


def arcs(data: NetworkData) -> list[Arc]:
    """Return the arcs of the network with the virtual SOURCE and SINK."""
    out = [(SOURCE, p.name, p.capacity, p.unit_cost) for p in data.plants]
    out += [(x.origin, x.destination, x.capacity, x.unit_cost) for x in data.lanes]
    out += [(s.name, SINK, s.demand, 0) for s in data.stores]
    return out


def nodes(data: NetworkData) -> list[str]:
    """Return every node name, virtual ones included."""
    return [
        SOURCE,
        *(p.name for p in data.plants),
        *data.warehouses,
        *(s.name for s in data.stores),
        SINK,
    ]


def flow_errors(
    data: NetworkData, arc_flow: dict[tuple[str, str], int], value: int, cost: int
) -> list[str]:
    """Return a list of broken rules. An empty list means a valid flow."""
    errors = []
    balance = dict.fromkeys(nodes(data), 0)
    total_cost = 0
    for tail, head, cap, unit_cost in arcs(data):
        f = arc_flow[tail, head]
        if not isinstance(f, int):
            errors.append(f"{tail}->{head}: flow {f!r} is not an integer")
        if f < 0 or f > cap:
            errors.append(f"{tail}->{head}: flow {f} outside [0, {cap}]")
        balance[tail] -= f
        balance[head] += f
        total_cost += unit_cost * f
    for node, b in balance.items():
        if node not in (SOURCE, SINK) and b != 0:
            errors.append(f"{node}: inflow and outflow differ by {b}")
    if balance[SINK] != value:
        errors.append(f"reported value {value}, but SINK receives {balance[SINK]}")
    if total_cost != cost:
        errors.append(f"reported cost {cost}, computed {total_cost}")
    return errors


def cut_capacity(data: NetworkData, source_side: set[str]) -> int:
    """Return the capacity of the cut (source_side, everything else)."""
    if SOURCE not in source_side or SINK in source_side:
        raise ValueError("A cut must put SOURCE and SINK on different sides.")
    return sum(
        cap
        for tail, head, cap, _ in arcs(data)
        if tail in source_side and head not in source_side
    )


def residual_arcs(
    data: NetworkData, arc_flow: dict[tuple[str, str], int]
) -> list[tuple[str, str, int]]:
    """Return the residual network as (tail, head, cost) arcs.

    An arc with room left gives a forward arc (send more, pay the cost).
    An arc with flow gives a backward arc (send less, get the cost back).
    """
    out = []
    for tail, head, cap, cost in arcs(data):
        f = arc_flow[tail, head]
        if f < cap:
            out.append((tail, head, cost))
        if f > 0:
            out.append((head, tail, -cost))
    return out


def _bellman_ford(names, edges, start: dict[str, float]):
    """Relax all edges |V| times. Return (distances, found_negative_cycle)."""
    dist = dict(start)
    for _ in range(len(names)):
        changed = False
        for u, v, c in edges:
            if dist[u] + c < dist[v]:
                dist[v] = dist[u] + c
                changed = True
        if not changed:
            return dist, False
    return dist, True  # Still improving after |V| rounds: a negative cycle.


def has_negative_cycle(data: NetworkData, arc_flow) -> bool:
    """Return True if the flow can be made cheaper without changing its size."""
    names = nodes(data)
    # Start every node at 0. This is like a virtual root with a free arc to
    # every node, so we find negative cycles anywhere in the network.
    _, negative = _bellman_ford(
        names, residual_arcs(data, arc_flow), dict.fromkeys(names, 0)
    )
    return negative


def marginal_costs(data: NetworkData, arc_flow) -> dict[str, float]:
    """Return the cost of one more pallet at each node, in dollars.

    This is the shortest-path distance from SOURCE in the residual network:
    the cheapest way to push one more pallet from some plant to the node,
    maybe by rerouting other pallets on the way. It is infinite where no
    more pallets can arrive. These distances are node potentials: the
    reduced cost c_uv + pi_u - pi_v of every residual arc is >= 0.
    """
    names = nodes(data)
    start = dict.fromkeys(names, math.inf)
    start[SOURCE] = 0.0
    dist, negative = _bellman_ford(names, residual_arcs(data, arc_flow), start)
    if negative:
        raise ValueError("The flow has a negative cycle; it is not optimal.")
    return dist
