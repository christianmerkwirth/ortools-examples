"""Vehicle routing with capacities and time windows (CVRPTW).

Decision: which van serves which customers, and in which order.
Goal: minimize fixed van costs plus driving distance (plus penalties for
customers left unserved, when that is allowed).
Limits: van capacity, delivery time windows, and the drivers' shift.

The routing library models such rules as *dimensions*. A dimension is a
quantity that accumulates along a route:

* The capacity dimension adds each stop's demand. Its value at the end of
  a route must stay within the van's capacity.
* The time dimension adds service time plus driving time. Its value at a
  stop is the start of service there, and it must fall in the stop's time
  window. Slack lets a van wait when it arrives early.

Like in example 12, the routing library is a heuristic. For small
instances, `solve_exact` builds the same problem in CP-SAT and proves the
optimum, so we can measure how good the routing solution is.
"""

import itertools
import math
import time
from dataclasses import dataclass

from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from ortools.sat.python import cp_model

from .data import VrpData, distance_matrix, travel_minutes


@dataclass(frozen=True)
class Plan:
    """A delivery plan: one route per used van, plus unserved customers."""

    routes: list[list[int]]  # Stop indices, each starts and ends at 0.
    start_times: list[list[int]]  # Start of service (minutes) per route stop.
    dropped: list[int]  # Customers left unserved.
    cost: int  # Objective: fixed costs + meters + penalties + paid time.
    distance: int  # Meters driven.
    seconds: float
    method: str
    proven_optimal: bool = False
    lower_bound: int | None = None


# ---------------------------------------------------------- routing library


def search_parameters(
    first_solution: str = "PARALLEL_CHEAPEST_INSERTION",
    metaheuristic: str | None = "GUIDED_LOCAL_SEARCH",
    time_limit: float = 5.0,
):
    """Return routing search parameters from plain names (as in example 12)."""
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
    data: VrpData,
    time_limit: float = 5.0,
    first_solution: str = "PARALLEL_CHEAPEST_INSERTION",
    metaheuristic: str | None = "GUIDED_LOCAL_SEARCH",
) -> Plan:
    """Solve a CVRPTW instance with the routing library."""
    n, vans = len(data.stops), data.num_vehicles
    manager = pywrapcp.RoutingIndexManager(n, vans, 0)  # 0 = depot.
    routing = pywrapcp.RoutingModel(manager)

    # Cost of each arc: road distance in meters.
    distance = distance_matrix(data)
    routing.SetArcCostEvaluatorOfAllVehicles(routing.RegisterTransitMatrix(distance))

    # Every van that leaves the depot costs a fixed amount. This makes the
    # solver use as few vans as is worth it.
    routing.SetFixedCostOfAllVehicles(data.fixed_cost)

    # Capacity dimension. A unary callback depends on one stop only.
    demand = routing.RegisterUnaryTransitVector([s.demand for s in data.stops])
    routing.AddDimensionWithVehicleCapacity(
        demand,
        0,  # No slack: load only changes by demand.
        [data.capacity] * vans,  # One capacity per van (fleets may differ).
        True,  # Every van starts empty.
        "Load",
    )

    # Time dimension. The transit from i to j is the service time at i
    # plus the drive from i to j, so the cumul at j is the moment the van
    # can start serving j. Slack is waiting time, for vans that arrive
    # before a window opens.
    drive = travel_minutes(data)
    transit_time = [
        [data.stops[i].service + drive[i][j] for j in range(n)] for i in range(n)
    ]
    shift = data.shift_end - data.shift_start
    routing.AddDimension(
        routing.RegisterTransitMatrix(transit_time),
        shift,  # Max waiting at one stop: we do not limit it.
        data.shift_end,  # No cumul may pass the end of the shift.
        False,  # Do not force vans to leave at time 0.
        "Time",
    )
    time_dim = routing.GetDimensionOrDie("Time")
    for c in data.customers:
        stop = data.stops[c]
        time_dim.CumulVar(manager.NodeToIndex(c)).SetRange(stop.open, stop.close)
    for v in range(vans):
        time_dim.CumulVar(routing.Start(v)).SetRange(data.shift_start, data.shift_end)
        time_dim.CumulVar(routing.End(v)).SetRange(data.shift_start, data.shift_end)
        # After the search: leave as late as possible and return as early as
        # possible, so the plan shows no needless waiting at the start or end.
        routing.AddVariableMaximizedByFinalizer(time_dim.CumulVar(routing.Start(v)))
        routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(routing.End(v)))

    # Paid working time. The span of a van is the time from leaving the
    # depot to coming back. A span cost charges every minute of it, waiting
    # included, so the solver now avoids long idle waits.
    if data.minute_cost:
        time_dim.SetSpanCostCoefficientForAllVehicles(data.minute_cost)

    # Optional visits. A disjunction with one node and a penalty means
    # "visit this node, or pay the penalty". Without it, every node must be
    # visited, and the solver fails if that is impossible.
    if data.drop_penalty_per_crate is not None:
        for c in data.customers:
            routing.AddDisjunction([manager.NodeToIndex(c)], data.drop_penalty(c))

    params = search_parameters(first_solution, metaheuristic, time_limit)
    start = time.perf_counter()
    solution = routing.SolveWithParameters(params)
    seconds = time.perf_counter() - start
    if solution is None:
        raise RuntimeError(f"No feasible plan found (status {routing.status()}).")

    routes, start_times = [], []
    for v in range(vans):
        index = routing.Start(v)
        if routing.IsEnd(solution.Value(routing.NextVar(index))):
            continue  # This van stays at the depot.
        route, times = [], []
        while True:
            route.append(manager.IndexToNode(index))
            times.append(solution.Min(time_dim.CumulVar(index)))
            if routing.IsEnd(index):
                break
            index = solution.Value(routing.NextVar(index))
        routes.append(route)
        start_times.append(times)

    served = {c for r in routes for c in r[1:-1]}
    method = first_solution + (f" + {metaheuristic}" if metaheuristic else "")
    return Plan(
        routes=routes,
        start_times=start_times,
        dropped=[c for c in data.customers if c not in served],
        cost=solution.ObjectiveValue(),
        distance=sum(distance[a][b] for r in routes for a, b in itertools.pairwise(r)),
        seconds=seconds,
        method=method,
    )


# ------------------------------------------------------------ exact baseline


def solve_exact(data: VrpData, time_limit: float = 30.0) -> Plan:
    """Solve a small CVRPTW instance exactly with CP-SAT.

    `add_multiple_circuit` is CP-SAT's routing constraint: the chosen arcs
    must form cycles through the depot (node 0), and every other node gets
    exactly one arc in and one arc out. A self-loop (i, i) means "node i is
    not visited", which models dropped customers.

    Load and time are integer variables per node, linked by implications:
    if the van drives from i to j, then j's load and start time must be at
    least i's value plus what happens on the way. This is the classic MTZ
    idea, and it also rules out sub-tours that miss the depot.
    """
    n = len(data.stops)
    dist, drive = distance_matrix(data), travel_minutes(data)
    model = cp_model.CpModel()

    arc = {
        (i, j): model.new_bool_var(f"{i}->{j}")
        for i in range(n)
        for j in range(n)
        if i != j
    }
    skip = {}
    if data.drop_penalty_per_crate is not None:
        skip = {c: model.new_bool_var(f"skip {c}") for c in data.customers}
    arcs = [(i, j, lit) for (i, j), lit in arc.items()]
    arcs += [(c, c, lit) for c, lit in skip.items()]
    model.add_multiple_circuit(arcs)

    # Each arc out of the depot starts one van's route.
    vans_used = sum(arc[0, j] for j in data.customers)
    model.add(vans_used <= data.num_vehicles)

    load = {
        c: model.new_int_var(data.stops[c].demand, data.capacity, f"load {c}")
        for c in data.customers
    }
    start = {
        c: model.new_int_var(data.stops[c].open, data.stops[c].close, f"t {c}")
        for c in data.customers
    }
    for (i, j), lit in arc.items():
        if i == 0:
            # Leave the depot no earlier than the shift start.
            model.add(start[j] >= data.shift_start + drive[0][j]).only_enforce_if(lit)
        elif j == 0:
            # Be back before the shift ends.
            done = start[i] + data.stops[i].service + drive[i][0]
            model.add(done <= data.shift_end).only_enforce_if(lit)
        else:
            model.add(load[j] >= load[i] + data.stops[j].demand).only_enforce_if(lit)
            gap = data.stops[i].service + drive[i][j]
            model.add(start[j] >= start[i] + gap).only_enforce_if(lit)

    # Paid time: every customer remembers when its van left the depot. The
    # last customer of a route then knows the route's working time.
    span_cost = 0
    if data.minute_cost:
        shift = data.shift_end - data.shift_start
        left = {
            c: model.new_int_var(data.shift_start, data.shift_end, f"left {c}")
            for c in data.customers
        }
        span = {c: model.new_int_var(0, shift, f"span {c}") for c in data.customers}
        for (i, j), lit in arc.items():
            if i == 0:
                model.add(left[j] == start[j] - drive[0][j]).only_enforce_if(lit)
            elif j == 0:
                back = start[i] + data.stops[i].service + drive[i][0]
                model.add(span[i] >= back - left[i]).only_enforce_if(lit)
            else:
                model.add(left[j] == left[i]).only_enforce_if(lit)
        span_cost = data.minute_cost * sum(span.values())

    model.minimize(
        sum(dist[i][j] * lit for (i, j), lit in arc.items())
        + data.fixed_cost * vans_used
        + sum(data.drop_penalty(c) * lit for c, lit in skip.items())
        + span_cost
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
    routes, start_times = [], []
    for first in data.customers:
        if not solver.boolean_value(arc[0, first]):
            continue
        route, times = [0, first], [None, solver.value(start[first])]
        while route[-1] != 0:
            nxt = successor[route[-1]]
            route.append(nxt)
            times.append(None if nxt == 0 else solver.value(start[nxt]))
        # Depot times: leave just in time for the first stop; return when done.
        times[0] = times[1] - drive[0][first]
        last = route[-2]
        times[-1] = times[-2] + data.stops[last].service + drive[last][0]
        routes.append(route)
        start_times.append(times)

    return Plan(
        routes=routes,
        start_times=start_times,
        dropped=[c for c, lit in skip.items() if solver.boolean_value(lit)],
        cost=round(solver.objective_value),
        distance=sum(dist[a][b] for r in routes for a, b in itertools.pairwise(r)),
        seconds=seconds,
        method="CP-SAT add_multiple_circuit",
        proven_optimal=status == cp_model.OPTIMAL,
        lower_bound=math.ceil(solver.best_objective_bound - 1e-6),
    )
