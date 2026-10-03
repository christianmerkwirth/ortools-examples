"""Independent checks for packings. Uses no OR-Tools code.

* `bin_packing_errors` and `strip_packing_errors` check every rule.
* The lower bounds in `bounds.py` prove optimality when a solution
  reaches them.
* `min_bins_brute_force` finds the optimum of tiny 1D instances by
  exhaustive search.
"""

import itertools

from .bounds import heights_that_fit
from .data import BinPackingData, StripData


def bin_packing_errors(data: BinPackingData, bins: list[list[str]]) -> list[str]:
    """Return a list of broken rules. An empty list means feasible."""
    errors = []
    packed = [name for b in bins for name in b]
    missing = set(data.sizes) - set(packed)
    if missing:
        errors.append(f"not packed: {sorted(missing)}")
    for name in {n for n in packed if packed.count(n) > 1}:
        errors.append(f"{name} packed {packed.count(name)} times")
    for name in set(packed) - set(data.sizes):
        errors.append(f"unknown item {name}")
    for k, b in enumerate(bins):
        load = sum(data.sizes.get(n, 0) for n in b)
        if load > data.capacity:
            errors.append(f"bin {k}: load {load} > capacity {data.capacity}")
        if not b:
            errors.append(f"bin {k} is empty")
    return errors


def min_bins_brute_force(sizes: dict[str, int], capacity: int) -> int:
    """Return the fewest bins by exhaustive search. Tiny instances only.

    Place the items one by one, into any open bin with room or into a new
    bin. Only open a new bin if it can beat the best count so far.
    """
    values = sorted(sizes.values(), reverse=True)
    best = len(values)

    def place(k: int, loads: list[int]) -> None:
        nonlocal best
        if len(loads) >= best:
            return
        if k == len(values):
            best = len(loads)
            return
        for b in range(len(loads)):
            if loads[b] + values[k] <= capacity:
                loads[b] += values[k]
                place(k + 1, loads)
                loads[b] -= values[k]
        place(k + 1, [*loads, values[k]])

    place(0, [])
    return best


def strip_packing_errors(
    data: StripData, placements: dict, length: int, allow_rotation: bool = True
) -> list[str]:
    """Return a list of broken rules. An empty list means feasible.

    `placements` maps panel names to objects with x, y, width, height, and
    rotated attributes.
    """
    errors = []
    panels = {p.name: p for p in data.panels}
    if set(placements) != set(panels):
        errors.append("placed panels differ from the order")
        return errors
    for name, q in placements.items():
        p = panels[name]
        expected = (p.height, p.width) if q.rotated else (p.width, p.height)
        if (q.width, q.height) != expected:
            errors.append(f"{name}: size {q.width}x{q.height} != {expected}")
        if q.rotated and not (allow_rotation and p.can_rotate):
            errors.append(f"{name}: rotated but rotation is not allowed")
        if q.x < 0 or q.y < 0 or q.x + q.width > data.strip_width:
            errors.append(f"{name}: outside the strip")
        if q.y + q.height > length:
            errors.append(f"{name}: ends at {q.y + q.height}, beyond length {length}")
    for (a, qa), (b, qb) in itertools.combinations(placements.items(), 2):
        overlap_x = qa.x < qb.x + qb.width and qb.x < qa.x + qa.width
        overlap_y = qa.y < qb.y + qb.height and qb.y < qa.y + qa.height
        if overlap_x and overlap_y:
            errors.append(f"{a} overlaps {b}")
    used = max(q.y + q.height for q in placements.values())
    if used != length:
        errors.append(f"reported length {length} != used length {used}")
    return errors


def waste(data: StripData, length: int) -> float:
    """Return the share of the used roll area that is not covered by panels."""
    area = sum(p.width * p.height for p in data.panels)
    return 1 - area / (length * data.strip_width)


def every_panel_fits(data: StripData, allow_rotation: bool = True) -> bool:
    """Return True if every panel fits across the strip in some orientation."""
    try:
        for p in data.panels:
            heights_that_fit(p, data.strip_width, allow_rotation)
    except ValueError:
        return False
    return True
