"""Independent checks for feed blends. Uses no OR-Tools code.

One function, `lagrangian_bound`, does two jobs:

* With the real costs and the solver's dual values, it gives a lower bound
  on the cost of every feasible blend. If it equals the cost of our blend,
  our blend is optimal.
* With all costs set to zero, a bound above zero proves that NO blend
  exists. (Any blend would cost 0, and 0 cannot be above the bound.) The
  elastic model's dual values give such prices for an infeasible spec.
"""

import math

from .data import BlendData

TOL = 1e-6


def feasibility_errors(data: BlendData, share: dict[str, float]) -> list[str]:
    """Return a list of violated limits. An empty list means feasible."""
    errors = []
    if abs(sum(share.values()) - 1.0) > TOL:
        errors.append(f"shares add up to {sum(share.values())}, not 1")
    for i in data.ingredients:
        if share[i.name] < -TOL or share[i.name] > i.max_share + TOL:
            errors.append(f"{i.name}: share {share[i.name]} outside [0, {i.max_share}]")
    for r in data.requirements:
        level = sum(
            i.nutrients.get(r.nutrient, 0.0) * share[i.name] for i in data.ingredients
        )
        scale = max(1.0, abs(r.min), abs(r.max) if math.isfinite(r.max) else 0.0)
        if level < r.min - TOL * scale or level > r.max + TOL * scale:
            errors.append(f"{r.nutrient}: level {level} outside [{r.min}, {r.max}]")
    return errors


def blend_cost(data: BlendData, share: dict[str, float]) -> float:
    """Return the cost of one batch of this blend."""
    return data.batch_kg * sum(i.cost * share[i.name] for i in data.ingredients)


def lagrangian_bound(
    data: BlendData,
    total_price: float,
    nutrient_price: dict[str, float],
    with_costs: bool = True,
) -> float:
    """Return a lower bound on the cost of ANY feasible blend.

    The LP is: min c'x subject to sum(x) = 1, min <= Ax <= max, and
    0 <= x <= max_share. For any prices t (total row) and y (nutrient rows)
    let d = c - t - A'y. Then for every feasible x:

        c'x = t + y'Ax + d'x
            >= t + sum_i min(y_i * min_i, y_i * max_i)
                 + sum_j min(0, d_j * max_share_j)

    The prices can be anything; the bound is always valid. Good prices
    make it tight.
    """
    bound = total_price
    for r in data.requirements:
        y = nutrient_price[r.nutrient]
        if abs(y) <= 1e-12:
            continue
        side = r.min if y > 0 else r.max
        if not math.isfinite(side):
            return -math.inf  # Wrong sign for an open side: no bound.
        bound += y * side
    for i in data.ingredients:
        c = data.batch_kg * i.cost if with_costs else 0.0
        d = (
            c
            - total_price
            - sum(i.nutrients.get(n, 0.0) * y for n, y in nutrient_price.items())
        )
        bound += min(0.0, d * i.max_share)
    return bound


def optimality_gap(data: BlendData, share, total_price, nutrient_price) -> float:
    """Return cost minus the dual bound. Zero (within round-off) is optimal."""
    return blend_cost(data, share) - lagrangian_bound(data, total_price, nutrient_price)


def proves_infeasible(data: BlendData, total_price, nutrient_price) -> bool:
    """Return True if these prices prove that no blend meets the spec."""
    return lagrangian_bound(data, total_price, nutrient_price, with_costs=False) > TOL
