"""Fast bounds and a heuristic for packing. Pure Python, no OR-Tools.

Both `model.py` and `check.py` use this file. The model needs an upper
bound on the number of bins. The checker uses the lower bounds to prove
that a solution is optimal without trusting the solver.
"""

import math
from functools import reduce


def first_fit_decreasing(sizes: dict[str, int], capacity: int) -> list[list[str]]:
    """Pack items by the first-fit-decreasing rule. Return the bins.

    Take the items from largest to smallest. Put each item into the first
    bin that still has room. Open a new bin if none has room. This simple
    rule never needs more than 11/9 of the optimum plus 6/9 bins.
    """
    bins: list[list[str]] = []
    loads: list[int] = []
    for name in sorted(sizes, key=lambda n: -sizes[n]):
        for k, load in enumerate(loads):
            if load + sizes[name] <= capacity:
                bins[k].append(name)
                loads[k] += sizes[name]
                break
        else:
            bins.append([name])
            loads.append(sizes[name])
    return bins


def l1_bound(sizes: dict[str, int], capacity: int) -> int:
    """Return the simplest lower bound: total size / capacity, rounded up."""
    return math.ceil(sum(sizes.values()) / capacity)


def l2_bound(sizes: dict[str, int], capacity: int) -> int:
    """Return the Martello-Toth lower bound L2 (1990).

    Pick a threshold a <= capacity/2. Items larger than capacity - a need a
    bin of their own. Items larger than capacity/2 cannot share a bin with
    each other. Items between a and capacity/2 can only use the space that
    the large items leave free, plus new bins. The best threshold gives the
    bound. L2 is never weaker than L1.
    """
    values = list(sizes.values())
    best = l1_bound(sizes, capacity)
    half = capacity / 2
    for a in sorted({0, *(v for v in values if v <= half)}):
        own = [v for v in values if v > capacity - a]
        big = [v for v in values if half < v <= capacity - a]
        mid = [v for v in values if a <= v <= half]
        free = len(big) * capacity - sum(big)
        extra = max(0, math.ceil((sum(mid) - free) / capacity))
        best = max(best, len(own) + len(big) + extra)
    return best


def grid_size(dims: list[int]) -> int:
    """Return the greatest common divisor of all dimensions."""
    return reduce(math.gcd, dims)


def strip_lower_bound(panels, strip_width: int, allow_rotation: bool = True) -> int:
    """Return a lower bound on the strip length for 2D strip packing.

    Two simple bounds:

    * Area: the strip must hold the total panel area.
    * Tallest piece: each panel needs at least its smallest possible height.

    If all sizes are multiples of g, we can push every panel down and left
    until it touches an edge or another panel. Then every coordinate is a
    sum of panel sizes, so the best length is a multiple of g too. That
    lets us round the bound up to the next multiple of g.
    """
    area = sum(p.width * p.height for p in panels)
    bound = math.ceil(area / strip_width)
    for p in panels:
        bound = max(bound, min(heights_that_fit(p, strip_width, allow_rotation)))
    g = grid_size([strip_width, *(d for p in panels for d in (p.width, p.height))])
    return math.ceil(bound / g) * g


def heights_that_fit(panel, strip_width: int, allow_rotation: bool) -> list[int]:
    """Return the heights this panel can take while it fits the strip width."""
    heights = []
    if panel.width <= strip_width:
        heights.append(panel.height)
    if allow_rotation and panel.can_rotate and panel.height <= strip_width:
        heights.append(panel.width)
    if not heights:
        raise ValueError(f"{panel.name} does not fit the strip in any orientation")
    return heights
