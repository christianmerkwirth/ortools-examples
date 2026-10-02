"""Tests for example 01: production planning LP."""

import itertools

import pytest
from ortools.math_opt.python import mathopt

from examples.ex01_production_planning.check import (
    dual_bound,
    feasibility_errors,
    optimality_errors,
)
from examples.ex01_production_planning.data import furniture_workshop
from examples.ex01_production_planning.model import (
    NoOptimalPlanError,
    profit_curve,
    solve_production_plan,
)

LP_SOLVERS = [
    mathopt.SolverType.GLOP,
    mathopt.SolverType.HIGHS,
    mathopt.SolverType.PDLP,
]


@pytest.fixture(scope="module")
def data():
    return furniture_workshop()


@pytest.fixture(scope="module")
def plan(data):
    return solve_production_plan(data)


def test_plan_is_feasible(data, plan):
    assert feasibility_errors(data, plan) == []


def test_plan_has_optimality_certificate(data, plan):
    assert optimality_errors(data, plan) == []


def test_known_optimum(plan):
    # Worked out by hand in the README: wood and carpentry are binding.
    assert plan.profit == pytest.approx(7920)
    assert plan.quantity == pytest.approx(
        {"chair": 120, "table": 24, "desk": 0, "bookshelf": 10}
    )
    assert plan.shadow_price == pytest.approx(
        {"wood": 1, "carpentry": 12, "finishing": 0}
    )


def test_no_vertex_beats_the_plan(data, plan):
    """Brute force: try every basis-like combination of active limits.

    An LP optimum lies at a vertex. With 4 variables, a vertex makes 4 of
    the limits tight. We solve every 4x4 system, keep feasible points, and
    check that none earns more than the solver's plan.
    """
    import numpy as np

    names = [p.name for p in data.products]
    rows, rhs = [], []
    for r in data.resources:
        rows.append([p.usage.get(r.name, 0.0) for p in data.products])
        rhs.append(r.capacity)
    for j, p in enumerate(data.products):
        unit = [1.0 if k == j else 0.0 for k in range(len(names))]
        rows += [unit, unit]
        rhs += [p.min_order, p.max_demand]
    A, b = np.array(rows), np.array(rhs)

    best = -np.inf
    for subset in itertools.combinations(range(len(rows)), len(names)):
        sub = list(subset)
        if abs(np.linalg.det(A[sub])) < 1e-9:
            continue
        x = np.linalg.solve(A[sub], b[sub])
        quantities = dict(zip(names, x, strict=True))
        lo_ok = all(quantities[p.name] >= p.min_order - 1e-6 for p in data.products)
        hi_ok = all(quantities[p.name] <= p.max_demand + 1e-6 for p in data.products)
        cap_ok = all(A[i] @ x <= b[i] + 1e-6 for i in range(len(data.resources)))
        if lo_ok and hi_ok and cap_ok:
            best = max(best, sum(p.profit * quantities[p.name] for p in data.products))
    assert best == pytest.approx(plan.profit)


@pytest.mark.parametrize("solver", LP_SOLVERS, ids=lambda s: s.name)
def test_lp_solvers_agree(data, plan, solver):
    other = solve_production_plan(data, solver)
    # PDLP is a first-order method. With default settings it is accurate to
    # about 1e-4 relative, so we only compare profits for it.
    assert other.profit == pytest.approx(plan.profit, rel=1e-4)
    if solver != mathopt.SolverType.PDLP:
        assert feasibility_errors(data, other) == []
        assert optimality_errors(data, other) == []


@pytest.mark.parametrize("resource", ["wood", "carpentry", "finishing"])
def test_shadow_price_predicts_extra_profit(data, plan, resource):
    """A small capacity increase earns shadow price x increase."""
    delta = 1.0
    cap = data.resource(resource).capacity
    bigger = solve_production_plan(data.with_capacity(resource, cap + delta))
    assert bigger.profit - plan.profit == pytest.approx(
        plan.shadow_price[resource] * delta
    )


def test_reduced_cost_is_the_profit_threshold(data, plan):
    """Desks enter the plan only once their profit rises past -reduced cost."""
    desk = next(p for p in data.products if p.name == "desk")
    threshold = desk.profit - plan.reduced_cost["desk"]
    below = solve_production_plan(data.with_profit("desk", threshold - 1))
    above = solve_production_plan(data.with_profit("desk", threshold + 1))
    assert below.quantity["desk"] == pytest.approx(0)
    assert above.quantity["desk"] > 1


def test_dual_bound_is_valid_for_any_prices(data, plan):
    """The bound holds for any non-negative prices, not just the optimal ones."""
    for prices in itertools.product([0.0, 0.5, 3.0, 20.0], repeat=3):
        y = dict(zip(["wood", "carpentry", "finishing"], prices, strict=True))
        assert dual_bound(data, y) >= plan.profit - 1e-6


def test_profit_curve_is_concave_and_matches_shadow_prices(data):
    caps = [float(c) for c in range(60, 700, 10)]
    points = profit_curve(data, "carpentry", caps)
    profits = [p for _, p, _ in points]
    assert all(p is not None for p in profits)
    slopes = [(b - a) / 10 for a, b in itertools.pairwise(profits)]
    # A maximization LP's value is concave in the right-hand side.
    assert all(s2 <= s1 + 1e-6 for s1, s2 in itertools.pairwise(slopes))
    # Each slope lies between the shadow prices at its two ends.
    for (_, _, y1), (_, _, y2), s in zip(points, points[1:], slopes, strict=False):
        assert y2 - 1e-6 <= s <= y1 + 1e-6


def test_too_little_capacity_is_infeasible(data):
    # 10 promised tables need 50 carpentry hours.
    with pytest.raises(NoOptimalPlanError):
        solve_production_plan(data.with_capacity("carpentry", 49))
    assert profit_curve(data, "carpentry", [49])[0][1] is None
