"""Pickup and delivery routing (dial-a-ride) with the routing library.

Decision: which bus serves which ride request, and in which order each bus
visits its pickups and drop-offs.
Goal: minimize fixed bus costs plus driving distance (plus taxi costs for
declined requests, when declining is allowed).
Limits: seats per bus, time windows, the drivers' shift, and a maximum
ride time per passenger.

Compared with example 13, a pickup and delivery problem adds three rules
for every request:

* the same bus must do the pickup and the drop-off,
* the pickup comes first, and
* (here) the passenger must not ride much longer than a direct trip.

The routing library knows the first two rules through
`AddPickupAndDelivery`. We state them again as explicit constraints,
because that helps the search and shows what the rules mean. The third
rule is one more constraint on the time dimension.

As in examples 12 and 13, the routing library is a heuristic.
`solve_exact` builds the same problem in CP-SAT. For small instances it
proves the optimum, so we can measure the routing result.
"""

import itertools
import math
import time
from dataclasses import dataclass

from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from ortools.sat.python import cp_model

from .data import (
    PdpData,
    distance_matrix,
    dropoff_node,
    max_ride,
    pickup_node,
    travel_minutes,
)

# Tight time windows make it hard to find ANY plan that serves everyone.
# The first-solution heuristics then often fail, even when a plan exists.
# The safety net: every request may be skipped, but at a penalty far above
# the cost of any real plan (here 10,000 km of driving). The search always
# has a starting point, and it serves everyone as soon as it can. If a
# request is still skipped at the end, we report that no plan was found.
SAFETY_NET_PENALTY = 10_000_000

POLICIES = {
    None: pywrapcp.RoutingModel.PICKUP_AND_DELIVERY_NO_ORDER,
    "LIFO": pywrapcp.RoutingModel.PICKUP_AND_DELIVERY_LIFO,
    "FIFO": pywrapcp.RoutingModel.PICKUP_AND_DELIVERY_FIFO,
}


@dataclass(frozen=True)
class Plan:
    """A day plan: one route per used bus, plus declined requests."""

    routes: list[list[int]]  # Node indices; each route starts and ends at 0.
    times: list[list[int]]  # Start of service per route node (depot: leave/return).
    declined: list[int]  # Request indices handed to a taxi.
    cost: int  # Objective: fixed costs + meters + taxi penalties.
    distance: int  # Meters driven.
    seconds: float
    method: str
    proven_optimal: bool = False
    lower_bound: int | None = None


# ---------------------------------------------------------- routing library


def search_parameters(
    first_solution: str = "PARALLEL_CHEAPEST_INSERTION",
    metaheuristic: str | None = "GUIDED_LOCAL_SEARCH",
    time_limit: float = 3.0,
):
    """Return routing search parameters from plain names (as in example 12).

    PARALLEL_CHEAPEST_INSERTION is the strategy of choice for pickup and
    delivery: it inserts a pickup and its drop-off together, so every
    partial plan already respects the pairs.
    """
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = getattr(
        routing_enums_pb2.FirstSolutionStrategy, first_solution
    )
    if metaheuristic is not None:
        params.local_search_metaheuristic = getattr(
            routing_enums_pb2.LocalSearchMetaheuristic, metaheuristic
        )
    params.time_limit.FromMilliseconds(round(1000 * time_limit))
    return params


def solve_routing(
    data: PdpData,
    time_limit: float = 3.0,
    policy: str | None = None,
    first_solution: str = "PARALLEL_CHEAPEST_INSERTION",
    metaheuristic: str | None = "GUIDED_LOCAL_SEARCH",
) -> Plan:
    """Solve a pickup and delivery instance with the routing library.

    `policy` may be None (any order), "LIFO" (last on, first off, as in a
    van stacked from the back), or "FIFO" (first on, first off).
    """
    nodes = data.stops()
    n, buses = len(nodes), data.num_vehicles
    manager = pywrapcp.RoutingIndexManager(n, buses, 0)  # 0 = garage.
    routing = pywrapcp.RoutingModel(manager)
    solver = routing.solver()

    distance = distance_matrix(data)
    routing.SetArcCostEvaluatorOfAllVehicles(routing.RegisterTransitMatrix(distance))
    routing.SetFixedCostOfAllVehicles(data.fixed_cost)

    # Seats. A pickup adds the party's seats, a drop-off frees them. The
    # cumul is the number of people on board after a stop. It must stay in
    # [0, capacity] everywhere, so the bus is never overfull.
    seats = routing.RegisterUnaryTransitVector([s.load for s in nodes])
    routing.AddDimensionWithVehicleCapacity(
        seats, 0, [data.capacity] * buses, True, "Seats"
    )

    # Time: service at i plus the drive from i to j. Slack is waiting time.
    drive = travel_minutes(data)
    transit = [[nodes[i].service + drive[i][j] for j in range(n)] for i in range(n)]
    routing.AddDimension(
        routing.RegisterTransitMatrix(transit),
        data.shift_end - data.shift_start,  # Any waiting is allowed.
        data.shift_end,
        False,
        "Time",
    )
    time_dim = routing.GetDimensionOrDie("Time")
    for node in range(1, n):
        index = manager.NodeToIndex(node)
        time_dim.CumulVar(index).SetRange(nodes[node].open, nodes[node].close)
    for v in range(buses):
        for index in (routing.Start(v), routing.End(v)):
            time_dim.CumulVar(index).SetRange(data.shift_start, data.shift_end)
        routing.AddVariableMaximizedByFinalizer(time_dim.CumulVar(routing.Start(v)))
        routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(routing.End(v)))

    for r in range(len(data.requests)):
        p = manager.NodeToIndex(pickup_node(r))
        d = manager.NodeToIndex(dropoff_node(r))
        # Tell the library that p and d form a pair. Its local search then
        # moves them together, and its insertion heuristics keep them
        # together.
        routing.AddPickupAndDelivery(p, d)
        # The same bus does both stops...
        solver.Add(routing.VehicleVar(p) == routing.VehicleVar(d))
        # ...and the pickup comes first.
        solver.Add(time_dim.CumulVar(p) <= time_dim.CumulVar(d))
        # Ride time: from the start of boarding to arrival at the drop-off.
        limit = max_ride(data, r)
        if limit is not None:
            solver.Add(time_dim.CumulVar(d) - time_dim.CumulVar(p) <= limit)
        # A route fixes the order of stops, not the exact times. After the
        # search, board each passenger as late as possible and drop them off
        # as early as possible: same route, shorter rides.
        routing.AddVariableMaximizedByFinalizer(time_dim.CumulVar(p))
        routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(d))
        # Declining a request drops both of its stops. PENALIZE_ONCE charges
        # the penalty once, not once per stop. If declining is not allowed,
        # we still add the disjunction, but with a huge penalty: see
        # SAFETY_NET_PENALTY.
        penalty = data.decline_penalty
        routing.AddDisjunction(
            [p, d],
            SAFETY_NET_PENALTY if penalty is None else penalty,
            2,
            routing.PENALIZE_ONCE,
        )

    routing.SetPickupAndDeliveryPolicyOfAllVehicles(POLICIES[policy])

    params = search_parameters(first_solution, metaheuristic, time_limit)
    start = time.perf_counter()
    solution = routing.SolveWithParameters(params)
    seconds = time.perf_counter() - start
    if solution is None:
        raise RuntimeError(f"No feasible plan found (status {routing.status()}).")

    routes, times = [], []
    for v in range(buses):
        index = routing.Start(v)
        if routing.IsEnd(solution.Value(routing.NextVar(index))):
            continue  # This bus stays in the garage.
        route, stamps = [], []
        while True:
            route.append(manager.IndexToNode(index))
            stamps.append(solution.Min(time_dim.CumulVar(index)))
            if routing.IsEnd(index):
                break
            index = solution.Value(routing.NextVar(index))
        routes.append(route)
        times.append(stamps)

    served = {node for route in routes for node in route[1:-1]}
    if data.decline_penalty is None and len(served) < n - 1:
        raise RuntimeError("No plan found that serves every request.")
    method = first_solution + (f" + {metaheuristic}" if metaheuristic else "")
    if policy:
        method += f", {policy}"
    return Plan(
        routes=routes,
        times=times,
        declined=[r for r in range(len(data.requests)) if pickup_node(r) not in served],
        cost=solution.ObjectiveValue(),
        distance=sum(distance[a][b] for r in routes for a, b in itertools.pairwise(r)),
        seconds=seconds,
        method=method,
    )


# ------------------------------------------------------------ exact baseline


def solve_exact(data: PdpData, time_limit: float = 30.0) -> Plan:
    """Solve a small pickup and delivery instance exactly with CP-SAT.

    As in example 13, `add_multiple_circuit` makes the chosen arcs form
    routes through the garage (node 0). A self-loop (i, i) means "node i is
    not visited", which models declined requests.

    The new part is "same bus". The buses are identical, so we do not need
    to know *which* bus serves a stop, only which stops share a route. Each
    node gets a route label: the first stop after the garage labels the
    route with its own number, and every arc passes the label on. A pickup
    and its drop-off must carry the same label.
    """
    nodes = data.stops()
    n = len(nodes)
    dist, drive = distance_matrix(data), travel_minutes(data)
    stops = range(1, n)
    model = cp_model.CpModel()

    arc = {
        (i, j): model.new_bool_var(f"{i}->{j}")
        for i in range(n)
        for j in range(n)
        if i != j
    }
    skip = {}
    if data.decline_penalty is not None:
        for r in range(len(data.requests)):
            lit = model.new_bool_var(f"decline {r}")
            skip[pickup_node(r)] = skip[dropoff_node(r)] = lit  # Both or none.
    arcs = [(i, j, lit) for (i, j), lit in arc.items()]
    arcs += [(c, c, lit) for c, lit in skip.items()]
    model.add_multiple_circuit(arcs)

    buses_used = sum(arc[0, j] for j in stops)
    model.add(buses_used <= data.num_vehicles)

    on_board = {c: model.new_int_var(0, data.capacity, f"seats {c}") for c in stops}
    start = {
        c: model.new_int_var(nodes[c].open, nodes[c].close, f"t {c}") for c in stops
    }
    label = {c: model.new_int_var(1, n - 1, f"route {c}") for c in stops}
    for (i, j), lit in arc.items():
        if i == 0:
            model.add(on_board[j] == nodes[j].load).only_enforce_if(lit)
            model.add(start[j] >= data.shift_start + drive[0][j]).only_enforce_if(lit)
            model.add(label[j] == j).only_enforce_if(lit)
        elif j == 0:
            done = start[i] + nodes[i].service + drive[i][0]
            model.add(done <= data.shift_end).only_enforce_if(lit)
        else:
            model.add(on_board[j] == on_board[i] + nodes[j].load).only_enforce_if(lit)
            gap = nodes[i].service + drive[i][j]
            model.add(start[j] >= start[i] + gap).only_enforce_if(lit)
            model.add(label[j] == label[i]).only_enforce_if(lit)

    for r in range(len(data.requests)):
        p, d = pickup_node(r), dropoff_node(r)
        served = [skip[p].Not()] if p in skip else []
        model.add(label[p] == label[d]).only_enforce_if(served)
        model.add(start[d] >= start[p] + nodes[p].service).only_enforce_if(served)
        limit = max_ride(data, r)
        if limit is not None:
            model.add(start[d] - start[p] <= limit).only_enforce_if(served)

    declined = [skip[pickup_node(r)] for r in range(len(data.requests))] if skip else []
    model.minimize(
        sum(dist[i][j] * lit for (i, j), lit in arc.items())
        + data.fixed_cost * buses_used
        + (data.decline_penalty or 0) * sum(declined)
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 8
    t0 = time.perf_counter()
    status = solver.solve(model)
    seconds = time.perf_counter() - t0
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"CP-SAT found no plan ({solver.status_name(status)}).")

    successor = {i: j for (i, j), lit in arc.items() if solver.boolean_value(lit)}
    routes, times = [], []
    for first in stops:
        if not solver.boolean_value(arc[0, first]):
            continue
        route = [0, first]
        while route[-1] != 0:
            route.append(successor[route[-1]])
        stamps = [solver.value(start[c]) for c in route[1:-1]]
        last = route[-2]
        leave = stamps[0] - drive[0][first]
        back = stamps[-1] + nodes[last].service + drive[last][0]
        routes.append(route)
        times.append([leave, *stamps, back])

    return Plan(
        routes=routes,
        times=times,
        declined=[r for r, lit in enumerate(declined) if solver.boolean_value(lit)],
        cost=round(solver.objective_value),
        distance=sum(dist[a][b] for r in routes for a, b in itertools.pairwise(r)),
        seconds=seconds,
        method="CP-SAT add_multiple_circuit",
        proven_optimal=status == cp_model.OPTIMAL,
        lower_bound=math.ceil(solver.best_objective_bound - 1e-6),
    )
