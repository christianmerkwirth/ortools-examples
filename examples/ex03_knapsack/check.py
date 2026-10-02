"""Independent checks for loadings. Uses no OR-Tools code.

* `feasibility_errors` checks every rule of a loading.
* `best_single_vehicle_value` solves a one-vehicle instance exactly by
  dynamic programming over (weight, volume). It is a different algorithm
  from both solvers, so it is a fair referee.
* `brute_force_value` tries every assignment. Only for tiny instances.
"""

import itertools

import numpy as np

from .data import LoadingData, Vehicle

VOLUME_SCALE = 10  # Same unit as the model: 0.1 m³.


def feasibility_errors(data: LoadingData, assignment: dict[str, str]) -> list[str]:
    """Return a list of broken rules. An empty list means feasible."""
    errors = []
    items = {i.name: i for i in data.items}
    vehicles = {v.name: v for v in data.vehicles}
    for item, vehicle in assignment.items():
        if item not in items or vehicle not in vehicles:
            errors.append(f"unknown item or vehicle: {item} -> {vehicle}")
        elif items[item].cold_chain and not vehicles[vehicle].refrigerated:
            errors.append(f"{item} needs a fridge but is on {vehicle}")
    for v in data.vehicles:
        load = [items[i] for i, w in assignment.items() if w == v.name]
        weight = sum(i.weight for i in load)
        volume = sum(round(i.volume * VOLUME_SCALE) for i in load) / VOLUME_SCALE
        if weight > v.max_weight:
            errors.append(f"{v.name}: {weight} kg > {v.max_weight} kg")
        if volume > v.max_volume + 1e-9:
            errors.append(f"{v.name}: {volume} m³ > {v.max_volume} m³")
        names = {i.name for i in load}
        for a, b in data.incompatible:
            if a in names and b in names:
                errors.append(f"{v.name}: carries both {a} and {b}")
    return errors


def loading_value(data: LoadingData, assignment: dict[str, str]) -> int:
    """Return the total priority score of the loaded items."""
    return sum(i.value for i in data.items if i.name in assignment)


def best_single_vehicle_value(data: LoadingData) -> int:
    """Return the optimal value of a one-vehicle instance by dynamic programming.

    best[w, v] is the highest value that fits in weight w and volume v. We
    add the items one by one. For each item, either skip it, or take it and
    use the best value of the remaining space. numpy does each step for the
    whole (w, v) table at once.
    """
    (vehicle,) = data.vehicles
    cap_w = vehicle.max_weight
    cap_v = round(vehicle.max_volume * VOLUME_SCALE)
    best = np.zeros((cap_w + 1, cap_v + 1), dtype=np.int64)
    for item in data.items:
        if item.cold_chain and not vehicle.refrigerated:
            continue
        w, v = item.weight, round(item.volume * VOLUME_SCALE)
        if w > cap_w or v > cap_v:
            continue
        take = best[: cap_w + 1 - w, : cap_v + 1 - v] + item.value
        # Compute the new values from the old table before writing.
        best[w:, v:] = np.maximum(best[w:, v:], take)
    return int(best[cap_w, cap_v])


def brute_force_value(data: LoadingData) -> int:
    """Try every way to place each item (or leave it). Tiny instances only."""
    choices = [None, *(v.name for v in data.vehicles)]
    best = 0
    for combo in itertools.product(choices, repeat=len(data.items)):
        assignment = {i.name: v for i, v in zip(data.items, combo, strict=True) if v}
        if not feasibility_errors(data, assignment):
            best = max(best, loading_value(data, assignment))
    return best


def pooled_bound(data: LoadingData) -> int:
    """Return an upper bound for a multi-vehicle instance.

    Merge all vehicles into one refrigerated vehicle with their summed
    limits, and drop the hazardous-goods rule. Every real loading also fits
    this pooled vehicle, so its optimum can only be higher or equal. If a
    real loading reaches the bound, that loading is optimal.
    """
    pooled = Vehicle(
        "pooled",
        sum(v.max_weight for v in data.vehicles),
        round(sum(v.max_volume for v in data.vehicles), 1),
        refrigerated=True,
    )
    return best_single_vehicle_value(LoadingData(data.items, (pooled,)))
