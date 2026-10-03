"""Independent checks for cutting plans. Uses no OR-Tools code.

* `plan_errors` checks that every pattern fits and every order is met.
* `best_worth` solves the pricing knapsack exactly by dynamic programming.
  It is a different algorithm from the CP-SAT pricing in model.py.
* `dual_bound` turns any non-negative dual values into a lower bound on
  the number of jumbo rolls. With the final duals of column generation it
  proves the LP optimal.
* `all_patterns` and `exact_min_rolls` solve tiny instances exhaustively.
"""

import math
from functools import cache

from .data import CuttingData

TOL = 1e-6
Pattern = tuple[int, ...]


def copy_limit(data: CuttingData, i: int) -> int:
    """Return the most rolls of order i one pattern may hold (as in model.py)."""
    o = data.orders[i]
    return min(data.roll_width // o.width, data.max_pieces, o.quantity)


def pattern_errors(data: CuttingData, pattern: Pattern) -> list[str]:
    """Return what is wrong with one pattern. An empty list means it fits."""
    errors = []
    used = sum(o.width * k for o, k in zip(data.orders, pattern, strict=True))
    if used > data.roll_width:
        errors.append(f"pattern {pattern} needs {used} mm > {data.roll_width} mm")
    if sum(pattern) > data.max_pieces:
        errors.append(
            f"pattern {pattern} has {sum(pattern)} pieces > {data.max_pieces}"
        )
    if any(k < 0 for k in pattern):
        errors.append(f"pattern {pattern} has a negative count")
    return errors


def plan_errors(data: CuttingData, cuts: dict[Pattern, int]) -> list[str]:
    """Return a list of broken rules for a whole plan. Empty means feasible."""
    errors = []
    for pattern, n in cuts.items():
        errors += pattern_errors(data, pattern)
        if n < 0:
            errors.append(f"pattern {pattern} used {n} times")
    for i, o in enumerate(data.orders):
        made = sum(p[i] * n for p, n in cuts.items())
        if made < o.quantity:
            errors.append(f"width {o.width}: {made} rolls cut, {o.quantity} ordered")
    return errors


def waste_mm(data: CuttingData, cuts: dict[Pattern, int]) -> int:
    """Return the total trim loss: jumbo width not cut into ordered rolls."""
    return sum(
        n
        * (
            data.roll_width
            - sum(o.width * k for o, k in zip(data.orders, p, strict=True))
        )
        for p, n in cuts.items()
    )


def best_worth(data: CuttingData, duals: list[float]) -> float:
    """Return max sum_i y_i a_i over all patterns, by dynamic programming.

    best[w][k] is the highest worth that fits in width w with at most k
    pieces, using the orders seen so far. Each order may be used up to its
    copy limit (a bounded knapsack), so we try 0, 1, 2, ... copies.
    """
    # Measure widths in units of their greatest common divisor (10 mm for
    # the paper mill). The table gets 10 times smaller; the answer is equal.
    unit = math.gcd(data.roll_width, *data.widths)
    W, K = data.roll_width // unit, data.max_pieces
    best = [[0.0] * (K + 1) for _ in range(W + 1)]
    for i, o in enumerate(data.orders):
        y = duals[i]
        if y <= 0:
            continue
        new = [row[:] for row in best]
        for c in range(1, copy_limit(data, i) + 1):
            width, value = c * o.width // unit, c * y
            for w in range(width, W + 1):
                old_row, new_row = best[w - width], new[w]
                for k in range(c, K + 1):
                    v = old_row[k - c] + value
                    if v > new_row[k]:
                        new_row[k] = v
        best = new
    return best[W][K]


def dual_bound(data: CuttingData, duals: list[float]) -> float:
    """Return a lower bound on the LP value, valid for ANY duals y >= 0.

    Let v = best_worth(y). Every pattern p has sum_i y_i a_ip <= v. Take
    any plan x that meets demand. Then

        sum_i d_i y_i <= sum_i y_i sum_p a_ip x_p
                      =  sum_p x_p (sum_i y_i a_ip)
                      <= v * sum_p x_p.

    So every plan uses at least sum_i d_i y_i / v jumbo rolls. (This is
    Farley's bound.) With optimal duals, v = 1 and the bound equals the LP
    value. Since rolls are whole, ceil(bound) also bounds every plan.
    """
    if any(y < -TOL for y in duals):
        raise ValueError("dual values must be non-negative")
    v = best_worth(data, duals)
    total = sum(o.quantity * y for o, y in zip(data.orders, duals, strict=True))
    return total / v if v > 0 else 0.0


def material_bound(data: CuttingData) -> float:
    """Return total ordered width divided by the jumbo width."""
    return sum(o.width * o.quantity for o in data.orders) / data.roll_width


def all_patterns(data: CuttingData) -> list[Pattern]:
    """List every non-empty pattern. Only for small instances.

    Go through the widths one by one and try 0, 1, 2, ... copies. Stop a
    branch as soon as the width or the number of knives runs out.
    """
    n = len(data.orders)
    found: list[Pattern] = []

    def extend(i: int, room: int, pieces: int, counts: list[int]) -> None:
        if i == n:
            if any(counts):
                found.append(tuple(counts))
            return
        width = data.orders[i].width
        for k in range(copy_limit(data, i) + 1):
            if k * width > room or pieces + k > data.max_pieces:
                break
            counts.append(k)
            extend(i + 1, room - k * width, pieces + k, counts)
            counts.pop()

    extend(0, data.roll_width, 0, [])
    return found


def maximal_patterns(data: CuttingData) -> list[Pattern]:
    """List patterns that cannot take one more roll of any width."""
    patterns = all_patterns(data)
    feasible = set(patterns)

    def grows(p: Pattern) -> bool:
        return any(p[:i] + (p[i] + 1,) + p[i + 1 :] in feasible for i in range(len(p)))

    return [p for p in patterns if not grows(p)]


def exact_min_rolls(data: CuttingData) -> int:
    """Return the fewest jumbo rolls by exhaustive search. Tiny instances only.

    f(d) = fewest rolls to meet remaining demand d. Try every maximal
    pattern as the next roll. Cutting more than needed does no harm, so we
    only need maximal patterns.
    """
    patterns = maximal_patterns(data)

    @cache
    def fewest(demand: tuple[int, ...]) -> int:
        if not any(demand):
            return 0
        best = math.inf
        for p in patterns:
            rest = tuple(max(0, d - k) for d, k in zip(demand, p, strict=True))
            if rest != demand:
                best = min(best, 1 + fewest(rest))
        return best

    return fewest(tuple(data.demand))
