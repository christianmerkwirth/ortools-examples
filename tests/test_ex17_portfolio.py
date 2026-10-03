"""Tests for example 17: CVaR portfolio selection with cardinality limits."""

import itertools

import numpy as np
import pytest
from ortools.math_opt.python import mathopt

from examples.ex17_portfolio.check import (
    cvar,
    feasibility_errors,
    lp_lower_bound,
    value_at_risk,
    zeta_range,
)
from examples.ex17_portfolio.data import PortfolioData, Rules, generate
from examples.ex17_portfolio.model import (
    NoPortfolioError,
    build_model,
    frontier,
    return_range,
    solve_model,
    solve_portfolio,
)

TARGET = 0.010
LP_RULES = Rules()
MIP_RULES = {
    6: Rules(min_weight=0.03, max_assets=6),
    4: Rules(min_weight=0.03, max_assets=4),
}


@pytest.fixture(scope="module")
def data():
    return generate()


@pytest.fixture(scope="module")
def lp(data):
    return solve_portfolio(data, LP_RULES, TARGET)


@pytest.fixture(scope="module")
def mips(data):
    return {k: solve_portfolio(data, r, TARGET) for k, r in MIP_RULES.items()}


def _bound(data, rules, target, p, **override):
    prices = dict(
        scenario_prices=p.scenario_prices,
        budget_price=p.budget_price,
        return_price=p.return_price,
        sector_prices=p.sector_prices,
    )
    prices.update(override)
    return lp_lower_bound(data, rules, target, **prices)


# ------------------------------------------------------------ the LP ---


def test_lp_portfolio_is_feasible(data, lp):
    assert feasibility_errors(data, LP_RULES, lp.weights, TARGET) == []
    assert lp.proven_optimal


def test_lp_cvar_matches_a_sorted_tail(data, lp):
    assert lp.cvar == pytest.approx(cvar(data, lp.weights, LP_RULES.alpha), abs=1e-9)
    lo, hi = zeta_range(data, lp.weights, LP_RULES.alpha)
    assert lo - 1e-9 <= lp.zeta <= hi + 1e-9


def test_lp_is_proven_optimal_by_its_dual_bound(data, lp):
    assert _bound(data, LP_RULES, TARGET, lp) == pytest.approx(lp.cvar, abs=1e-9)


def test_dual_bound_is_valid_for_other_prices(data, lp):
    """Any valid prices give a bound below the optimum, never above."""
    rng = np.random.default_rng(0)
    S = data.n_scenarios
    k = round((1 - LP_RULES.alpha) * S)
    for _ in range(50):
        pi = np.zeros(S)
        pi[rng.choice(S, size=k, replace=False)] = 1.0 / k  # Sums to 1, <= 1/k.
        sectors = {s: -rng.uniform(0, 0.05) for s in lp.sector_prices}
        bound = _bound(
            data,
            LP_RULES,
            TARGET,
            lp,
            scenario_prices=pi,
            budget_price=rng.uniform(-0.1, 0.1),
            return_price=rng.uniform(0, 10),
            sector_prices=sectors,
        )
        assert bound <= lp.cvar + 1e-9


@pytest.mark.parametrize(
    "solver",
    [mathopt.SolverType.HIGHS, mathopt.SolverType.PDLP],
    ids=lambda s: s.name,
)
def test_lp_solvers_agree(data, lp, solver):
    other = solve_portfolio(data, LP_RULES, TARGET, solver_type=solver)
    # PDLP is a first-order method; it is accurate to about 1e-4.
    assert other.cvar == pytest.approx(lp.cvar, rel=1e-3)


def test_shadow_price_is_the_frontier_slope(data, lp):
    step = 1e-5
    higher = solve_portfolio(data, LP_RULES, TARGET + step)
    assert (higher.cvar - lp.cvar) / step == pytest.approx(lp.return_price, rel=1e-3)


def test_lp_frontier_is_increasing_and_convex(data):
    low, high = return_range(data, LP_RULES)
    targets = list(np.linspace(low, high, 15))
    risks = [p.cvar for p in frontier(data, LP_RULES, targets)]
    steps = np.diff(risks)
    assert (steps >= -1e-9).all()
    # Equal target spacing, so convexity means growing steps.
    assert (np.diff(steps) >= -1e-9).all()


def test_target_above_the_best_return_is_infeasible(data):
    _, high = return_range(data, LP_RULES)
    solve_portfolio(data, LP_RULES, high - 1e-7)  # Must not raise.
    with pytest.raises(NoPortfolioError):
        solve_portfolio(data, LP_RULES, high + 1e-4)


# ----------------------------------------------------------- the MIPs ---


@pytest.mark.parametrize("k", [6, 4])
def test_mip_portfolios_meet_every_rule(data, mips, k):
    p = mips[k]
    assert feasibility_errors(data, MIP_RULES[k], p.weights, TARGET) == []
    assert p.proven_optimal
    assert p.cvar == pytest.approx(cvar(data, p.weights, MIP_RULES[k].alpha), abs=1e-9)
    lo, hi = zeta_range(data, p.weights, MIP_RULES[k].alpha)
    assert lo - 1e-9 <= p.zeta <= hi + 1e-9


def test_fewer_assets_never_lower_the_risk(lp, mips):
    assert lp.cvar <= mips[6].cvar + 1e-9 <= mips[4].cvar + 2e-9


def test_scip_agrees_with_highs(data, mips):
    other = solve_portfolio(
        data, MIP_RULES[4], TARGET, solver_type=mathopt.SolverType.GSCIP
    )
    assert other.cvar == pytest.approx(mips[4].cvar, rel=1e-5)


def _small_instance() -> tuple[PortfolioData, Rules, float]:
    data = generate().subset([0, 1, 5, 6, 10, 11, 15, 20, 25, 26])
    rules = Rules(max_weight=0.6, sector_cap=0.6, min_weight=0.1, max_assets=3)
    return data, rules, float(np.median(data.mean_returns))


def test_mip_matches_enumeration_of_all_subsets():
    """Solve the LP for every set of at most 3 held assets; keep the best."""
    data, rules, target = _small_instance()
    best = np.inf
    lp_rules = Rules(max_weight=rules.max_weight, sector_cap=rules.sector_cap)
    for size in range(1, rules.max_assets + 1):
        for held in itertools.combinations(range(data.n_assets), size):
            pm = build_model(data, lp_rules, target)
            for i, w in enumerate(pm.weight):
                # Held assets get the minimum position; the rest get nothing.
                w.lower_bound = rules.min_weight if i in held else 0.0
                w.upper_bound = rules.max_weight if i in held else 0.0
            try:
                best = min(best, solve_model(pm, data).cvar)
            except NoPortfolioError:
                continue
    mip = solve_portfolio(data, rules, target)
    assert np.isfinite(best)
    assert mip.cvar == pytest.approx(best, abs=1e-9)
    assert feasibility_errors(data, rules, mip.weights, target) == []


def test_mip_frontier_is_increasing(data):
    low, high = return_range(data, MIP_RULES[6])
    targets = list(np.linspace(low, high, 6))
    risks = [p.cvar for p in frontier(data, MIP_RULES[6], targets)]
    assert (np.diff(risks) >= -1e-9).all()


# ---------------------------------------------------------- the checks ---


def test_cvar_on_a_hand_example():
    # Two assets, four scenarios; alpha = 0.5 means the worst two months.
    returns = np.array([[0.02, -0.01], [-0.04, 0.01], [0.01, 0.00], [-0.02, -0.03]])
    data = PortfolioData(generate().assets[:2], returns)
    w = np.array([0.5, 0.5])
    # Portfolio returns: 0.5%, -1.5%, 0.5%, -2.5% -> losses 2.5% and 1.5%.
    assert cvar(data, w, 0.5) == pytest.approx(0.02)
    assert value_at_risk(data, w, 0.5) == pytest.approx(0.015)
    assert zeta_range(data, w, 0.5) == pytest.approx((-0.005, 0.015))
    # alpha = 0.6 -> tail of 1.6 months: 2.5% plus 0.6 of 1.5%.
    assert cvar(data, w, 0.6) == pytest.approx((0.025 + 0.6 * 0.015) / 1.6)


def test_checker_catches_broken_rules(data):
    w = np.zeros(data.n_assets)
    w[:5] = [0.3, 0.3, 0.2, 0.15, 0.01]  # Utility sector 96%; one tiny position.
    errors = feasibility_errors(data, Rules(min_weight=0.03, max_assets=4), w, 0.05)
    text = " ".join(errors)
    assert "outside" in text  # 30% > 25% cap.
    assert "minimum position" in text
    assert "limit is 4" in text
    assert "sector cap" in text
    assert "below the target" in text
    assert "add up" in text
