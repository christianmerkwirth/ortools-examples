"""Knapsack models: one vehicle with two solvers, then several vehicles.

Decision: which items to load (and, in Part B, on which vehicle).
Goal: maximize the total priority score.
Limits: weight and volume of every vehicle, plus side rules in Part B.

Part A solves the same problem twice:

* `solve_with_knapsack_solver` uses OR-Tools' specialized branch-and-bound
  knapsack solver. It is fast, but it only knows knapsack constraints.
* `solve_with_cp_sat` uses CP-SAT, a general solver. It is a few lines
  longer, but you can add any rule.

Part B (`solve_convoy`) needs rules that the knapsack solver cannot
express, so it uses CP-SAT only.
"""

from dataclasses import dataclass

from ortools.algorithms.python import knapsack_solver
from ortools.sat.python import cp_model

from .data import Item, LoadingData

# Both solvers need integer data. Volumes have one decimal, so we count
# volume in units of 0.1 m³.
VOLUME_SCALE = 10


@dataclass(frozen=True)
class Loading:
    """A solution: which items go on which vehicle."""

    assignment: dict[str, str]  # Item name -> vehicle name (loaded items only).
    value: int
    proven_optimal: bool

    def items_on(self, vehicle: str) -> list[str]:
        """Return the names of the items loaded on one vehicle."""
        return [i for i, v in self.assignment.items() if v == vehicle]


def _volume_units(volume: float) -> int:
    return round(volume * VOLUME_SCALE)


# ---------------------------------------------------------------- Part A ---


def solve_with_knapsack_solver(data: LoadingData, time_limit: float = 10.0) -> Loading:
    """Solve a single-vehicle instance with the dedicated knapsack solver."""
    (vehicle,) = data.vehicles
    items = _allowed_items(data.items, vehicle.refrigerated)

    solver = knapsack_solver.KnapsackSolver(
        knapsack_solver.SolverType.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER,
        "relief plane",
    )
    # init(values, weights, capacities): `weights` has one row per
    # dimension. Here the two dimensions are weight and volume.
    solver.init(
        [i.value for i in items],
        [[i.weight for i in items], [_volume_units(i.volume) for i in items]],
        [vehicle.max_weight, _volume_units(vehicle.max_volume)],
    )
    solver.set_time_limit(time_limit)
    value = solver.solve()
    chosen = {
        item.name: vehicle.name
        for k, item in enumerate(items)
        if solver.best_solution_contains(k)
    }
    return Loading(chosen, value, solver.is_solution_optimal())


def solve_with_cp_sat(data: LoadingData, time_limit: float = 10.0) -> Loading:
    """Solve a single-vehicle instance with CP-SAT."""
    (vehicle,) = data.vehicles
    items = _allowed_items(data.items, vehicle.refrigerated)

    model = cp_model.CpModel()
    take = {i.name: model.new_bool_var(f"take {i.name}") for i in items}
    model.add(sum(i.weight * take[i.name] for i in items) <= vehicle.max_weight)
    model.add(
        sum(_volume_units(i.volume) * take[i.name] for i in items)
        <= _volume_units(vehicle.max_volume)
    )
    model.maximize(sum(i.value * take[i.name] for i in items))

    solver = _cp_solver(time_limit)
    status = solver.solve(model)
    _require_solution(status)
    chosen = {n: vehicle.name for n, var in take.items() if solver.boolean_value(var)}
    return Loading(chosen, round(solver.objective_value), status == cp_model.OPTIMAL)


def _allowed_items(items, refrigerated: bool) -> list[Item]:
    """Drop cold-chain items if the vehicle cannot keep them cold."""
    return [i for i in items if refrigerated or not i.cold_chain]


# ---------------------------------------------------------------- Part B ---


def solve_convoy(data: LoadingData, time_limit: float = 10.0) -> Loading:
    """Load several vehicles at once (multiple knapsack) with side rules."""
    model = cp_model.CpModel()

    # on[i, v] is true if item i travels on vehicle v. We only create the
    # variable if the pair is allowed, so cold-chain items simply have no
    # variable for vehicles without a fridge.
    on = {
        (i.name, v.name): model.new_bool_var(f"{i.name} on {v.name}")
        for i in data.items
        for v in data.vehicles
        if v.refrigerated or not i.cold_chain
    }

    # Each item travels at most once. (Leaving it behind is allowed.)
    for i in data.items:
        model.add_at_most_one(
            on[i.name, v.name] for v in data.vehicles if (i.name, v.name) in on
        )

    # Weight and volume limits of each vehicle.
    for v in data.vehicles:
        loaded = [(i, on[i.name, v.name]) for i in data.items if (i.name, v.name) in on]
        model.add(sum(i.weight * x for i, x in loaded) <= v.max_weight)
        model.add(
            sum(_volume_units(i.volume) * x for i, x in loaded)
            <= _volume_units(v.max_volume)
        )

    # Hazardous pairs must not share a vehicle.
    for a, b in data.incompatible:
        for v in data.vehicles:
            if (a, v.name) in on and (b, v.name) in on:
                model.add_at_most_one(on[a, v.name], on[b, v.name])

    item_value = {i.name: i.value for i in data.items}
    model.maximize(sum(item_value[i] * x for (i, _), x in on.items()))

    solver = _cp_solver(time_limit)
    status = solver.solve(model)
    _require_solution(status)
    chosen = {i: v for (i, v), x in on.items() if solver.boolean_value(x)}
    return Loading(chosen, round(solver.objective_value), status == cp_model.OPTIMAL)


# --------------------------------------------------------------- helpers ---


def _cp_solver(time_limit: float) -> cp_model.CpSolver:
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 8  # Parallel portfolio of strategies.
    return solver


def _require_solution(status) -> None:
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"CP-SAT found no solution (status {status}).")
