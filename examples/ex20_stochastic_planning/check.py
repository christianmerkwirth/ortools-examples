"""Independent checks for season plans. Uses no OR-Tools code.

* `plan_errors` checks the summer decision: whole lines, no negative
  stock, and the sewing hours fit.
* `profits` plays out a plan in each scenario. The best winter reaction is
  simple once the stock is fixed: buy express only to cover a shortfall
  (and only while it is cheaper than the sale price), sell what customers
  want, and send the rest to the outlet. We use it to evaluate any plan on
  thousands of fresh scenarios, far more than the solver ever sees.
* `best_plan` finds the optimal plan for a set of scenarios without a
  solver. It works because this model has a special shape (see below). It
  is our referee for the deterministic, what-if, and stochastic models.
"""

import numpy as np

from .data import PlanningData, Scenarios

TOL = 1e-6


def plan_errors(data: PlanningData, lines: int, make: np.ndarray) -> list[str]:
    """Return a list of broken rules. An empty list means feasible."""
    errors = []
    if lines != int(lines) or not 0 <= lines <= data.plant.max_lines:
        errors.append(f"lines = {lines}, need a whole number in [0, max]")
    if (make < -TOL).any():
        errors.append(f"negative production {make}")
    hours = float(data.column("hours") @ make)
    if hours > data.plant.line_hours * lines + TOL:
        errors.append(f"{hours:.1f} sewing hours on {lines} lines")
    return errors


def product_profit(
    data: PlanningData, make: np.ndarray, scenarios: Scenarios
) -> np.ndarray:
    """Return profit[s, p] of each product, before the cost of the lines."""
    q, c, v, u = (data.column(a) for a in ("price", "cost", "salvage", "rush_cap"))
    d, r = scenarios.demand, scenarios.rush_price
    short = np.maximum(0.0, d - make)
    rush = np.where(r < q, np.minimum(u, short), 0.0)
    sold = np.minimum(d, make + rush)
    left = make + rush - sold
    return q * sold + v * left - r * rush - c * make


def profits(
    data: PlanningData, lines: int, make: np.ndarray, scenarios: Scenarios
) -> np.ndarray:
    """Return the season profit of a plan in each scenario."""
    product = product_profit(data, np.asarray(make, dtype=float), scenarios)
    return product.sum(axis=1) - data.plant.line_cost * lines


def best_plan(
    data: PlanningData, scenarios: Scenarios
) -> tuple[int, np.ndarray, float]:
    """Return (lines, make, average profit) of an optimal plan, without a solver.

    For a fixed stock x_p, the average profit of product p is a concave,
    piecewise linear function of x_p. Its kinks sit where x_p equals a
    demand d_sp (stock starts to go to the outlet) or d_sp - u_p (express
    alone can no longer close the gap). Between kinks, each extra unit
    earns a fixed amount, and each later unit earns no more than an earlier
    one.

    So for a fixed number of lines, the best stock fills the sewing hours
    greedily: take the pieces with the highest profit per sewing hour
    first, and stop when hours run out or no piece earns money. This is the
    fractional knapsack rule, and it is exact here. Then try every number
    of lines and keep the best.

    The trick needs this special shape. Add one constraint that links the
    products in winter (say, one shared express budget), and only a solver
    can do the job.
    """
    pieces = []  # (profit per hour, length in units, product, slope).
    base = 0.0
    for i, prod in enumerate(data.products):
        d = scenarios.demand[:, i]
        kinks = np.concatenate(([0.0], d, np.maximum(0.0, d - prod.rush_cap)))
        kinks = np.unique(kinks)
        # value[k] = average profit of product i if we make kinks[k] units.
        # Treat each kink as its own "product" column, so one call does all.
        alone = PlanningData((prod,), data.plant)
        one = Scenarios(d[:, None], scenarios.rush_price[:, [i]])
        value = product_profit(alone, kinks, one).mean(axis=0)
        base += value[0]
        slopes = np.diff(value) / np.diff(kinks)
        for length, slope in zip(np.diff(kinks), slopes, strict=True):
            if slope > 1e-12:
                pieces.append((slope / prod.hours, length, i, slope))
    pieces.sort(key=lambda t: -t[0])

    best = (0, np.zeros(len(data.products)), -np.inf)
    for lines in range(data.plant.max_lines + 1):
        hours = data.plant.line_hours * lines
        make = np.zeros(len(data.products))
        value = base - data.plant.line_cost * lines
        for _, length, i, slope in pieces:
            take = min(length, hours / data.products[i].hours)
            if take <= 0:
                break
            make[i] += take
            value += take * slope
            hours -= take * data.products[i].hours
        if value > best[2] + 1e-9:
            best = (lines, make, value)
    return best
