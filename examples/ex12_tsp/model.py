"""Traveling salesperson: the routing library, and an exact CP-SAT baseline.

Decision: the order in which to visit the sites.
Goal: minimize the total distance of the round trip.
Limits: visit every site exactly once; start and end at the depot.

The routing library is a heuristic. It builds a first tour fast and then
improves it by local search. It does not prove that a tour is optimal. To
know how good its tours are, we also solve the problem exactly with CP-SAT
and its `add_circuit` constraint. That works well up to about a hundred
sites. Beyond that, the routing library is the practical tool.
"""

import math
import time
from dataclasses import dataclass

from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from ortools.sat.python import cp_model

from .data import TspData, distance_matrix


@dataclass(frozen=True)
class Tour:
    """A round trip and how it was found."""

    route: list[int]  # Site indices; starts and ends at the depot.
    length: int  # Meters.
    seconds: float  # Wall-clock solve time.
    method: str
    proven_optimal: bool = False
    lower_bound: int | None = None  # Only from the exact solver.


# --------------------------------------------------------- routing library ---
#
# These helpers are small on purpose: example 13 (vehicle routing) builds on
# them. The routing library has its own "index" for each stop. With one
# vehicle, index and site number are often equal, but not always (vehicles
# with several start or end depots add extra indices). Always convert with
# the index manager.


def build_routing(
    matrix: list[list[int]], depot: int = 0, num_vehicles: int = 1
) -> tuple[pywrapcp.RoutingIndexManager, pywrapcp.RoutingModel, int]:
    """Create the index manager, the routing model, and the distance callback.

    Returns:
        The manager, the model, and the index of the registered distance
        callback (use it to add dimensions later).

    """
    manager = pywrapcp.RoutingIndexManager(len(matrix), num_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)
    # RegisterTransitMatrix keeps the whole matrix in C++. The solver then
    # never calls back into Python, which is much faster than a Python
    # callback (see `register_python_callback`).
    transit = routing.RegisterTransitMatrix(matrix)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)
    return manager, routing, transit


def register_python_callback(
    manager: pywrapcp.RoutingIndexManager,
    routing: pywrapcp.RoutingModel,
    matrix: list[list[int]],
) -> int:
    """Register the same distances as a Python function (the classic way).

    The callback gets routing *indices*, not site numbers. Convert them
    with the manager before you look up the matrix.
    """

    def distance(from_index: int, to_index: int) -> int:
        return matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

    return routing.RegisterTransitCallback(distance)


def search_parameters(
    first_solution: str = "PATH_CHEAPEST_ARC",
    metaheuristic: str | None = None,
    time_limit: float = 1.0,
    first_solution_only: bool = False,
):
    """Return routing search parameters from plain names.

    Args:
        first_solution: A `FirstSolutionStrategy` name, for example
            "PATH_CHEAPEST_ARC" or "CHRISTOFIDES".
        metaheuristic: A `LocalSearchMetaheuristic` name, for example
            "GUIDED_LOCAL_SEARCH". None keeps the default, a greedy descent
            that stops at the first local optimum.
        time_limit: Seconds. Metaheuristics never stop on their own, so
            they need a limit. Greedy descent usually stops much earlier.
        first_solution_only: Stop after the first tour (no local search).

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
    if first_solution_only:
        params.solution_limit = 1
    return params


def extract_route(
    manager: pywrapcp.RoutingIndexManager,
    routing: pywrapcp.RoutingModel,
    solution: pywrapcp.Assignment,
    vehicle: int = 0,
) -> list[int]:
    """Follow one vehicle's `NextVar` chain and return its site numbers."""
    index = routing.Start(vehicle)
    route = [manager.IndexToNode(index)]
    while not routing.IsEnd(index):
        index = solution.Value(routing.NextVar(index))
        route.append(manager.IndexToNode(index))
    return route


def solve_with_routing(
    data: TspData,
    first_solution: str = "PATH_CHEAPEST_ARC",
    metaheuristic: str | None = None,
    time_limit: float = 1.0,
    first_solution_only: bool = False,
    python_callback: bool = False,
) -> Tour:
    """Solve the TSP with the routing library and return the tour."""
    matrix = distance_matrix(data)
    manager, routing, _ = build_routing(matrix, data.depot)
    if python_callback:
        routing.SetArcCostEvaluatorOfAllVehicles(
            register_python_callback(manager, routing, matrix)
        )
    params = search_parameters(
        first_solution, metaheuristic, time_limit, first_solution_only
    )

    start = time.perf_counter()
    solution = routing.SolveWithParameters(params)
    seconds = time.perf_counter() - start
    if solution is None:
        raise RuntimeError(f"No tour found (status {routing.status()}).")

    method = first_solution + (f" + {metaheuristic}" if metaheuristic else "")
    if first_solution_only:
        method += " (no local search)"
    elif metaheuristic is None:
        method += " + greedy descent"
    return Tour(
        route=extract_route(manager, routing, solution),
        length=solution.ObjectiveValue(),
        seconds=seconds,
        method=method,
    )


# ---------------------------------------------------------- exact baseline ---


def solve_exact(data: TspData, time_limit: float = 60.0) -> Tour:
    """Solve the TSP exactly with CP-SAT's circuit constraint.

    One Boolean literal per arc (i, j) says "drive from i straight to j".
    `add_circuit` forces the chosen arcs to form one single cycle through
    all sites. That rules out sub-tours, the classic difficulty of TSP
    models, without any extra constraints.

    If the time limit stops the search, the tour may not be optimal. Then
    `lower_bound` tells how far from optimal it can be at most.
    """
    matrix = distance_matrix(data)
    n = data.size
    model = cp_model.CpModel()
    arc = {
        (i, j): model.new_bool_var(f"{i}->{j}")
        for i in range(n)
        for j in range(n)
        if i != j
    }
    model.add_circuit([(i, j, lit) for (i, j), lit in arc.items()])
    model.minimize(sum(matrix[i][j] * lit for (i, j), lit in arc.items()))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 8
    start = time.perf_counter()
    status = solver.solve(model)
    seconds = time.perf_counter() - start
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"CP-SAT found no tour ({solver.status_name(status)}).")

    # Walk the chosen arcs from the depot to get the visiting order.
    successor = {i: j for (i, j), lit in arc.items() if solver.boolean_value(lit)}
    route = [data.depot]
    for _ in range(n):
        route.append(successor[route[-1]])
    return Tour(
        route=route,
        length=round(solver.objective_value),
        seconds=seconds,
        method="CP-SAT add_circuit",
        proven_optimal=status == cp_model.OPTIMAL,
        # Tour lengths are whole meters, so we may round the bound up.
        lower_bound=math.ceil(solver.best_objective_bound - 1e-6),
    )
