"""Portfolio selection: minimize CVaR, as an LP or a MIP, with MathOpt.

Decision: the share w_i of the money in each asset.
Goal: minimize CVaR, the average loss in the worst 5% of scenarios.
Limits: earn at least a target mean return, invest all the money, cap each
asset and each sector. The MIP version adds a cardinality limit (hold at
most K assets) and a minimum position size for every asset held.

CVaR is not linear in w, but Rockafellar and Uryasev showed that it is the
optimal value of a small LP. With one extra variable zeta (the value at
risk) and one variable u_s per scenario (the loss beyond zeta), we get

    CVaR(w) = min  zeta + 1 / ((1 - alpha) S) * sum_s u_s
              s.t. u_s >= loss_s(w) - zeta,  u_s >= 0.

So the whole problem stays linear. LP solvers handle thousands of
scenarios with ease, and MIP solvers can add yes/no choices on top.
"""

import datetime
from dataclasses import dataclass

import numpy as np
from ortools.math_opt.python import mathopt

from .data import SECTORS, PortfolioData, Rules

OPTIMAL = mathopt.TerminationReason.OPTIMAL
INFEASIBLE = mathopt.TerminationReason.INFEASIBLE


class NoPortfolioError(RuntimeError):
    """Raised when no portfolio meets the rules and the target."""


@dataclass
class PortfolioModel:
    """A MathOpt model plus handles to its variables and constraints."""

    model: mathopt.Model
    is_mip: bool
    weight: list[mathopt.Variable]
    held: list[mathopt.Variable] | None  # Binary z_i; only in the MIP.
    zeta: mathopt.Variable  # Lies at the value at risk at the optimum.
    excess: list[mathopt.Variable]  # u_s: loss beyond zeta in scenario s.
    scenario_rows: list[mathopt.LinearConstraint]
    budget_row: mathopt.LinearConstraint
    return_row: mathopt.LinearConstraint
    sector_rows: dict[str, mathopt.LinearConstraint]


@dataclass(frozen=True)
class Portfolio:
    """A solved portfolio and what the solver says about it."""

    weights: np.ndarray  # Share of the money per asset.
    cvar: float  # Objective value: average loss in the worst tail.
    # zeta at the optimum. It lies between the two losses at the edge of the
    # tail; any value in that range is optimal, so it is "a" VaR, not "the".
    zeta: float
    expected_return: float  # Mean return over the scenarios.
    proven_optimal: bool
    gap: float  # Relative gap between objective and best bound (MIP).
    # Dual values; only for the LP (no integer variables).
    return_price: float | None = None  # Extra CVaR per unit of extra target.
    budget_price: float | None = None
    scenario_prices: np.ndarray | None = None
    sector_prices: dict[str, float] | None = None


def needs_integers(rules: Rules) -> bool:
    """Return True if the rules need binary variables."""
    return rules.max_assets is not None or rules.min_weight > 0


def build_model(data: PortfolioData, rules: Rules, target: float) -> PortfolioModel:
    """Build the CVaR model for one target mean return."""
    model = mathopt.Model(name="cvar portfolio")
    n, S = data.n_assets, data.n_scenarios
    mu = data.mean_returns
    weight = [
        model.add_variable(lb=0.0, ub=rules.max_weight, name=a.name)
        for a in data.assets
    ]
    zeta = model.add_variable(lb=-np.inf, ub=np.inf, name="zeta")
    excess = [model.add_variable(lb=0.0, name=f"u{s}") for s in range(S)]

    # u_s >= loss_s - zeta, with loss_s = -(returns in s) . w. Moved to one
    # side: u_s + returns_s . w + zeta >= 0.
    scenario_rows = [
        model.add_linear_constraint(
            excess[s]
            + mathopt.fast_sum(float(data.returns[s, i]) * weight[i] for i in range(n))
            + zeta
            >= 0.0
        )
        for s in range(S)
    ]
    budget_row = model.add_linear_constraint(
        mathopt.fast_sum(weight) == 1.0, name="budget"
    )
    return_row = model.add_linear_constraint(
        mathopt.fast_sum(float(mu[i]) * weight[i] for i in range(n)) >= target,
        name="target return",
    )
    sector_rows = {
        sector: model.add_linear_constraint(
            mathopt.fast_sum(
                weight[i] for i, a in enumerate(data.assets) if a.sector == sector
            )
            <= rules.sector_cap,
            name=sector,
        )
        for sector in SECTORS
        if any(a.sector == sector for a in data.assets)
    }

    held = None
    if needs_integers(rules):
        # z_i = 1 if asset i is held. Two links tie w_i to z_i:
        #   w_i <= max_weight * z_i   (no money unless held), and
        #   w_i >= min_weight * z_i   (if held, at least the minimum).
        # Together they make w_i "semi-continuous": 0 or in [min, max].
        held = [model.add_binary_variable(name=f"hold {a.name}") for a in data.assets]
        for w, z in zip(weight, held, strict=True):
            model.add_linear_constraint(w <= rules.max_weight * z)
            model.add_linear_constraint(w >= rules.min_weight * z)
        if rules.max_assets is not None:
            model.add_linear_constraint(
                mathopt.fast_sum(held) <= rules.max_assets, name="max assets"
            )

    tail = 1.0 / ((1.0 - rules.alpha) * S)
    model.minimize(zeta + tail * mathopt.fast_sum(excess))
    return PortfolioModel(
        model,
        held is not None,
        weight,
        held,
        zeta,
        excess,
        scenario_rows,
        budget_row,
        return_row,
        sector_rows,
    )


def solve_model(
    pm: PortfolioModel,
    data: PortfolioData,
    solver_type: mathopt.SolverType | None = None,
    time_limit: float = 20.0,
) -> Portfolio:
    """Solve a built model. GLOP for the LP, HiGHS for the MIP by default."""
    if solver_type is None:
        solver_type = mathopt.SolverType.HIGHS if pm.is_mip else mathopt.SolverType.GLOP
    params = mathopt.SolveParameters(
        time_limit=datetime.timedelta(seconds=time_limit),
        relative_gap_tolerance=1e-6,
    )
    result = mathopt.solve(pm.model, solver_type, params=params)
    reason = result.termination.reason
    if reason == INFEASIBLE:
        raise NoPortfolioError("No portfolio meets the rules and the target.")
    if not result.has_primal_feasible_solution():
        raise RuntimeError(f"No solution found: {result.termination}")

    weights = np.array([result.variable_values(w) for w in pm.weight])
    weights[np.abs(weights) < 1e-9] = 0.0  # Clean solver round-off.
    cvar = result.objective_value()
    bound = result.termination.objective_bounds.dual_bound
    gap = abs(cvar - bound) / max(1e-9, abs(cvar))
    portfolio = dict(
        weights=weights,
        cvar=cvar,
        zeta=result.variable_values(pm.zeta),
        expected_return=float(data.mean_returns @ weights),
        proven_optimal=reason == OPTIMAL,
        gap=gap if reason != OPTIMAL else 0.0,
    )
    if not pm.is_mip and result.has_dual_feasible_solution():
        # Dual values exist only for LPs. The price of the target-return row
        # is the slope of the efficient frontier at this point.
        portfolio.update(
            return_price=result.dual_values(pm.return_row),
            budget_price=result.dual_values(pm.budget_row),
            scenario_prices=np.array([result.dual_values(r) for r in pm.scenario_rows]),
            sector_prices={k: result.dual_values(r) for k, r in pm.sector_rows.items()},
        )
    return Portfolio(**portfolio)


def solve_portfolio(
    data: PortfolioData,
    rules: Rules,
    target: float,
    solver_type: mathopt.SolverType | None = None,
    time_limit: float = 20.0,
) -> Portfolio:
    """Build and solve the model in one call."""
    return solve_model(build_model(data, rules, target), data, solver_type, time_limit)


def return_range(data: PortfolioData, rules: Rules) -> tuple[float, float]:
    """Return the lowest-risk portfolio's return and the highest reachable return.

    Targets below the first value do not change the answer (the return row
    is slack). Targets above the second are infeasible.
    """
    pm = build_model(data, rules, target=-1.0)
    low = solve_model(pm, data).expected_return
    mu = data.mean_returns
    pm.model.maximize(
        mathopt.fast_sum(float(mu[i]) * w for i, w in enumerate(pm.weight))
    )
    high = solve_model(pm, data).expected_return
    return low, high


def frontier(
    data: PortfolioData, rules: Rules, targets: list[float], time_limit: float = 20.0
) -> list[Portfolio | None]:
    """Solve for each target return. Build the model once; change one bound.

    Returns None for targets that no portfolio can reach.
    """
    pm = build_model(data, rules, target=targets[0])
    points = []
    for t in targets:
        pm.return_row.lower_bound = t
        try:
            points.append(solve_model(pm, data, time_limit=time_limit))
        except NoPortfolioError:
            points.append(None)
    return points
