"""Tests for example 15: cutting stock with column generation."""

import math
import random

import pytest
from ortools.math_opt.python import mathopt

from examples.ex15_cutting_stock.check import (
    all_patterns,
    best_worth,
    dual_bound,
    exact_min_rolls,
    material_bound,
    pattern_errors,
    plan_errors,
)
from examples.ex15_cutting_stock.data import (
    paper_mill,
    random_instance,
    rush_order,
    tiny,
)
from examples.ex15_cutting_stock.model import (
    column_generation,
    first_fit_plan,
    kantorovich,
    price_pattern,
    round_down_and_repair,
    round_up,
    solve_pattern_ip,
)


@pytest.fixture(scope="module")
def mill():
    return paper_mill()


@pytest.fixture(scope="module")
def lp(mill):
    return column_generation(mill)


def lp_over(data, patterns) -> float:
    """Solve the pattern LP over a given list of patterns with GLOP."""
    model = mathopt.Model()
    x = [model.add_variable(lb=0) for _ in patterns]
    for i, o in enumerate(data.orders):
        model.add_linear_constraint(
            sum(p[i] * v for p, v in zip(patterns, x, strict=True) if p[i])
            >= o.quantity
        )
    model.minimize(sum(x))
    result = mathopt.solve(model, mathopt.SolverType.GLOP)
    assert result.termination.reason == mathopt.TerminationReason.OPTIMAL
    return result.objective_value()


# ------------------------------------------------------------- the LP ---


def test_lp_value_is_known(lp):
    assert lp.value == pytest.approx(61.27536, abs=1e-4)
    assert all(pattern_errors(paper_mill(), p) == [] for p in lp.patterns)


def test_lp_matches_full_enumeration(mill, lp):
    """Column generation must reach the LP over all 873 patterns."""
    patterns = all_patterns(mill)
    assert len(patterns) == 873
    assert lp.value == pytest.approx(lp_over(mill, patterns), rel=1e-9)


def test_dual_certificate(mill, lp):
    assert all(y >= -1e-9 for y in lp.duals)
    assert best_worth(mill, lp.duals) <= 1 + 1e-6
    assert dual_bound(mill, lp.duals) == pytest.approx(lp.value, rel=1e-9)


def test_dual_bound_is_valid_for_any_prices(mill, lp):
    rng = random.Random(0)
    for _ in range(20):
        y = [rng.uniform(0, 0.5) for _ in mill.orders]
        assert dual_bound(mill, y) <= lp.value + 1e-9


def test_lp_values_never_rise_between_rounds(lp):
    values = [h.lp_value for h in lp.history]
    assert all(b <= a + 1e-9 for a, b in zip(values, values[1:], strict=False))
    assert all(h.lower_bound <= lp.value + 1e-9 for h in lp.history)


@pytest.mark.parametrize("seed", range(5))
def test_pricing_dp_cp_sat_and_brute_force_agree(mill, seed):
    rng = random.Random(seed)
    duals = [rng.uniform(0, 0.4) for _ in mill.orders]
    brute = max(
        sum(y * k for y, k in zip(duals, p, strict=True)) for p in all_patterns(mill)
    )
    _, cp_sat = price_pattern(mill, duals)
    assert best_worth(mill, duals) == pytest.approx(brute, rel=1e-9)
    assert cp_sat == pytest.approx(brute, rel=1e-5)  # CP-SAT sees scaled duals.


# ----------------------------------------------------- integer plans ---


def test_all_plans_are_feasible(mill, lp):
    for plan in (
        first_fit_plan(mill),
        round_up(lp),
        round_down_and_repair(mill, lp),
        solve_pattern_ip(mill, lp.patterns),
    ):
        assert plan_errors(mill, plan.cuts) == [], plan.method
        assert plan.rolls >= math.ceil(lp.value - 1e-6)


def test_best_plan_reaches_the_lp_bound(mill, lp):
    plan = solve_pattern_ip(mill, lp.patterns)
    assert plan.rolls == math.ceil(lp.value - 1e-6) == 62


def test_full_pattern_ip_confirms_62(mill):
    assert solve_pattern_ip(mill, all_patterns(mill)).rolls == 62


def test_first_fit_respects_the_knife_limit(mill):
    plan = first_fit_plan(mill)
    assert all(sum(p) <= mill.max_pieces for p in plan.cuts)
    assert plan.rolls == 67


# --------------------------------------------------------- rush order ---


def test_rush_order_needs_all_patterns():
    data = rush_order()
    lp = column_generation(data)
    bound = math.ceil(lp.value - 1e-6)
    full = solve_pattern_ip(data, all_patterns(data))
    assert plan_errors(data, full.cuts) == []
    assert full.rolls == bound == exact_min_rolls(data) == 17
    # Price-and-branch can only be as good as its patterns allow. In our
    # runs it needs 18; it may never need fewer than the optimum.
    assert solve_pattern_ip(data, lp.patterns).rolls >= 17
    # Here the pattern LP bound is much stronger than total width.
    assert math.ceil(material_bound(data)) == 16


# ------------------------------------------------ tiny and random ---


def test_tiny_exhaustive_search():
    data = tiny()
    exact = exact_min_rolls(data)
    assert exact == solve_pattern_ip(data, all_patterns(data)).rolls == 6
    lp = column_generation(data)
    assert math.ceil(lp.value - 1e-6) <= exact


@pytest.mark.parametrize("seed", range(6))
def test_random_instances(seed):
    data = random_instance(5, seed=seed)
    lp = column_generation(data)
    patterns = all_patterns(data)
    assert lp.value == pytest.approx(lp_over(data, patterns), rel=1e-9)
    assert best_worth(data, lp.duals) <= 1 + 1e-6
    optimum = solve_pattern_ip(data, patterns).rolls
    plan = solve_pattern_ip(data, lp.patterns)
    assert plan_errors(data, plan.cuts) == []
    assert math.ceil(lp.value - 1e-6) <= optimum <= plan.rolls
    # Integer round-up property: the optimum is within 1 of the LP bound.
    assert optimum - math.ceil(lp.value - 1e-6) <= 1


# ------------------------------------------------------- Kantorovich ---


@pytest.mark.parametrize("make", [tiny, rush_order], ids=["tiny", "rush"])
def test_kantorovich_agrees_but_has_a_weaker_bound(make):
    data = make()
    upper = first_fit_plan(data).rolls
    value, optimal = kantorovich(data, upper, time_limit=20)
    assert optimal
    assert value == pytest.approx(exact_min_rolls(data))
    lp_bound, _ = kantorovich(data, upper, relax=True)
    assert lp_bound == pytest.approx(material_bound(data), rel=1e-6)
    assert lp_bound <= column_generation(data).value + 1e-9
