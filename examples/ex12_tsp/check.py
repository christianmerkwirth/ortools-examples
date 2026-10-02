"""Independent checks for tours. Uses no OR-Tools code.

* `tour_errors` checks that a tour is a valid round trip.
* `tour_length` recomputes the length from the map coordinates.
* `held_karp` finds the exact optimum by dynamic programming. It needs
  O(2^n * n^2) steps, so it is only for small instances (up to ~13 sites).
* `brute_force` tries every order. Only for up to ~9 sites.

Both exact methods use the same integer distances as the model, so their
results must match the solvers' tour lengths to the meter.
"""

import itertools
import math

import numpy as np

from .data import TspData

METERS_PER_KM = 1000  # Same unit as the model.


def leg_matrix(data: TspData) -> np.ndarray:
    """Return integer distances in meters, computed with numpy."""
    xy = np.array([(s.x, s.y) for s in data.sites])
    diff = xy[:, None, :] - xy[None, :, :]
    return np.rint(METERS_PER_KM * np.sqrt((diff**2).sum(axis=2))).astype(np.int64)


def tour_errors(data: TspData, route: list[int]) -> list[str]:
    """Return a list of problems with a tour. An empty list means valid."""
    errors = []
    if len(route) != data.size + 1:
        errors.append(f"tour has {len(route)} stops, expected {data.size + 1}")
    if not route or route[0] != data.depot or route[-1] != data.depot:
        errors.append("tour must start and end at the depot")
    inner = route[1:-1]
    missing = set(range(data.size)) - {data.depot} - set(inner)
    if missing:
        errors.append(f"sites never visited: {sorted(missing)}")
    if len(inner) != len(set(inner)) or data.depot in inner:
        errors.append("a site is visited more than once")
    return errors


def tour_length(data: TspData, route: list[int]) -> int:
    """Return the length of a tour in meters."""
    legs = leg_matrix(data)
    return int(sum(legs[a, b] for a, b in itertools.pairwise(route)))


def held_karp(data: TspData) -> tuple[int, list[int]]:
    """Return the optimal tour length and one optimal tour.

    best[S][j] is the shortest path that starts at the depot, visits every
    site in the set S exactly once, and ends at site j (j in S). Sets are
    bit masks over the non-depot sites. We grow the sets one site at a
    time; the optimal tour closes the best full path back to the depot.
    """
    legs = leg_matrix(data).tolist()  # Plain lists are faster in loops.
    others = [k for k in range(data.size) if k != data.depot]
    m = len(others)
    if m == 0:
        return 0, [data.depot, data.depot]
    # d[a][b]: distance between the a-th and b-th "other" site.
    d = [[legs[a][b] for b in others] for a in others]
    from_depot = [legs[data.depot][a] for a in others]

    full = (1 << m) - 1
    best = [[math.inf] * m for _ in range(1 << m)]
    parent = [[-1] * m for _ in range(1 << m)]
    for j in range(m):
        best[1 << j][j] = from_depot[j]

    for mask in range(1, full + 1):  # Subsets come before their supersets.
        row = best[mask]
        for j in range(m):
            cost_j = row[j]
            if cost_j == math.inf:
                continue  # j is not in mask, or no path ends there.
            # Extend the path that ends at j to every site k not in mask.
            for k in range(m):
                if mask >> k & 1:
                    continue
                new_mask = mask | 1 << k
                cost = cost_j + d[j][k]
                if cost < best[new_mask][k]:
                    best[new_mask][k] = cost
                    parent[new_mask][k] = j

    closing = [best[full][j] + from_depot[j] for j in range(m)]  # Symmetric.
    last = min(range(m), key=closing.__getitem__)

    # Follow the parent links back to rebuild the tour.
    path, mask, j = [], full, last
    while j != -1:
        path.append(others[j])
        prev = parent[mask][j]
        mask &= ~(1 << j)
        j = prev
    return int(closing[last]), [data.depot, *reversed(path), data.depot]


def brute_force(data: TspData) -> int:
    """Return the optimal tour length by trying every order of the sites."""
    legs = leg_matrix(data).tolist()
    others = [k for k in range(data.size) if k != data.depot]
    best = math.inf
    for order in itertools.permutations(others):
        route = [data.depot, *order, data.depot]
        best = min(best, sum(legs[a][b] for a, b in itertools.pairwise(route)))
    return int(best)
