"""Independent checks for portfolios. Uses no OR-Tools code.

* `cvar` computes the tail loss directly: sort the scenario losses and
  average the worst ones. It does not use the LP formula at all, so it is a
  fair check of the solver's objective value.
* `feasibility_errors` checks every rule.
* `lp_lower_bound` turns the solver's dual values into a lower bound on the
  CVaR of ANY portfolio that meets the rules (LP case, no integer rules).
  If the bound equals our CVaR, our portfolio is optimal.
"""

import math

import numpy as np

from .data import SECTORS, PortfolioData, Rules

TOL = 1e-6


def portfolio_losses(data: PortfolioData, weights: np.ndarray) -> np.ndarray:
    """Return the loss in each scenario (a gain is a negative loss)."""
    return -(data.returns @ weights)


def cvar(data: PortfolioData, weights: np.ndarray, alpha: float) -> float:
    """Return the average loss over the worst (1 - alpha) share of scenarios.

    With S scenarios, the tail holds (1 - alpha) * S of them. If that is not
    a whole number, the boundary scenario counts with a fractional weight.
    """
    losses = np.sort(portfolio_losses(data, weights))[::-1]  # Worst first.
    tail = (1.0 - alpha) * len(losses)
    whole = math.floor(tail + 1e-9)
    total = losses[:whole].sum()
    if tail - whole > 1e-9:
        total += (tail - whole) * losses[whole]
    return float(total / tail)


def value_at_risk(data: PortfolioData, weights: np.ndarray, alpha: float) -> float:
    """Return the loss that the worst (1 - alpha) share of months exceed."""
    losses = np.sort(portfolio_losses(data, weights))[::-1]
    tail = (1.0 - alpha) * len(losses)
    return float(losses[max(0, math.ceil(tail - 1e-9) - 1)])


def zeta_range(
    data: PortfolioData, weights: np.ndarray, alpha: float
) -> tuple[float, float]:
    """Return the range of zeta values that are optimal in the CVaR LP.

    If the tail holds a whole number k of scenarios, any zeta between the
    (k+1)-th and the k-th worst loss gives the same objective. Otherwise
    zeta is unique: the loss of the boundary scenario.
    """
    losses = np.sort(portfolio_losses(data, weights))[::-1]
    tail = (1.0 - alpha) * len(losses)
    whole = math.floor(tail + 1e-9)
    if abs(tail - whole) < 1e-9:
        return float(losses[whole]), float(losses[whole - 1])
    return float(losses[whole]), float(losses[whole])


def feasibility_errors(
    data: PortfolioData, rules: Rules, weights: np.ndarray, target: float
) -> list[str]:
    """Return a list of broken rules. An empty list means feasible."""
    errors = []
    if abs(weights.sum() - 1.0) > TOL:
        errors.append(f"weights add up to {weights.sum():.8f}, not 1")
    held = weights > TOL
    for a, w, h in zip(data.assets, weights, held, strict=True):
        if w < -TOL or w > rules.max_weight + TOL:
            errors.append(f"{a.name}: weight {w:.6f} outside [0, {rules.max_weight}]")
        if h and w < rules.min_weight - TOL:
            errors.append(f"{a.name}: weight {w:.6f} below the minimum position")
    if rules.max_assets is not None and held.sum() > rules.max_assets:
        errors.append(f"holds {held.sum()} assets, limit is {rules.max_assets}")
    for sector in SECTORS:
        share = sum(
            w for a, w in zip(data.assets, weights, strict=True) if a.sector == sector
        )
        if share > rules.sector_cap + TOL:
            errors.append(f"{sector}: {share:.4f} above the sector cap")
    mean = float(data.mean_returns @ weights)
    if mean < target - TOL:
        errors.append(f"mean return {mean:.6f} below the target {target:.6f}")
    return errors


def lp_lower_bound(
    data: PortfolioData,
    rules: Rules,
    target: float,
    scenario_prices: np.ndarray,
    budget_price: float,
    return_price: float,
    sector_prices: dict[str, float],
) -> float:
    """Return a lower bound on the CVaR of every portfolio that meets the rules.

    This is the Lagrangian bound of the CVaR LP (without integer rules).
    Price each row: pi_s >= 0 for the scenario rows, lambda for the budget,
    gamma >= 0 for the return target, sigma_k <= 0 for the sector caps. The
    reduced cost of each column is its objective coefficient minus the
    priced rows. For every feasible point,

        objective >= sum of (price * row bound) + sum over columns of
                     min(reduced cost * lower bound, reduced cost * upper bound).

    Columns with an infinite bound need a reduced cost of zero (zeta, which
    is free) or at least zero (the u_s). Otherwise the bound is -infinity.
    """
    S, n = data.returns.shape
    pi = np.asarray(scenario_prices)
    if (pi < -TOL).any() or return_price < -TOL:
        return -math.inf
    if any(v > TOL for v in sector_prices.values()):
        return -math.inf

    tail = 1.0 / ((1.0 - rules.alpha) * S)
    bound = budget_price * 1.0 + return_price * target
    bound += sum(sector_prices[k] * rules.sector_cap for k in sector_prices)
    # The scenario rows have bound 0, so they add nothing here.

    # zeta: objective coefficient 1, appears in every scenario row.
    if abs(1.0 - pi.sum()) > 1e-7:
        return -math.inf
    # u_s: objective coefficient `tail`, appears in its own scenario row.
    if (tail - pi < -1e-7).any():
        return -math.inf
    # w_i: objective coefficient 0.
    mu = data.mean_returns
    sector_of = [a.sector for a in data.assets]
    for i in range(n):
        d = (
            -float(pi @ data.returns[:, i])
            - budget_price
            - return_price * mu[i]
            - sector_prices.get(sector_of[i], 0.0)
        )
        bound += min(0.0, d * rules.max_weight)
    return bound
