"""Independent checks for a production plan. Uses no OR-Tools code.

Two checks together prove that a plan is optimal:

1. Primal feasibility: the plan respects every limit.
2. An LP optimality certificate: the shadow prices that the solver reports
   give an upper bound on the profit of every possible plan. If that bound
   equals the profit of our plan, no better plan can exist.
"""

from .data import ProductionData
from .model import ProductionPlan

TOL = 1e-6


def feasibility_errors(data: ProductionData, plan: ProductionPlan) -> list[str]:
    """Return a list of violated limits. An empty list means feasible."""
    errors = []
    for p in data.products:
        q = plan.quantity[p.name]
        if q < p.min_order - TOL:
            errors.append(f"{p.name}: {q} below promised order {p.min_order}")
        if q > p.max_demand + TOL:
            errors.append(f"{p.name}: {q} above demand {p.max_demand}")
    for r in data.resources:
        used = sum(
            p.usage.get(r.name, 0.0) * plan.quantity[p.name] for p in data.products
        )
        if used > r.capacity + TOL:
            errors.append(f"{r.name}: uses {used} of {r.capacity}")
    profit = sum(p.profit * plan.quantity[p.name] for p in data.products)
    if abs(profit - plan.profit) > TOL * max(1.0, abs(profit)):
        errors.append(f"reported profit {plan.profit} != computed {profit}")
    return errors


def dual_bound(data: ProductionData, shadow_price: dict[str, float]) -> float:
    """Return an upper bound on the profit of ANY feasible plan.

    For max c'x subject to Ax <= b and l <= x <= u, pick any prices y >= 0
    and let d = c - A'y. Then for every feasible x:

        c'x = y'Ax + d'x <= y'b + sum_j max(d_j * l_j, d_j * u_j)

    The first step uses Ax <= b and y >= 0. The second uses l <= x <= u.
    So the right-hand side bounds every plan, whatever y we choose.
    """
    bound = sum(r.capacity * shadow_price[r.name] for r in data.resources)
    for p in data.products:
        d = p.profit - sum(p.usage.get(r, 0.0) * y for r, y in shadow_price.items())
        bound += max(d * p.min_order, d * p.max_demand)
    return bound


def optimality_errors(data: ProductionData, plan: ProductionPlan) -> list[str]:
    """Check the LP optimality certificate. An empty list means optimal.

    If the shadow prices are non-negative and their dual bound equals the
    profit of a feasible plan, no plan can do better. The reduced costs are
    also recomputed from the data and compared with the solver's values.
    """
    errors = []
    y = plan.shadow_price
    scale = max(1.0, abs(plan.profit))

    for name, price in y.items():
        if price < -TOL:
            errors.append(f"{name}: negative shadow price {price}")

    for p in data.products:
        d = p.profit - sum(p.usage.get(r, 0.0) * y[r] for r in y)
        if abs(d - plan.reduced_cost[p.name]) > TOL * scale:
            errors.append(f"{p.name}: reduced cost {plan.reduced_cost[p.name]} != {d}")

    bound = dual_bound(data, y)
    if abs(bound - plan.profit) > TOL * scale:
        errors.append(f"duality gap: profit {plan.profit}, dual bound {bound}")
    return errors
