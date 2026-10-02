"""Tests for example 02: feed blending and infeasibility diagnosis."""

import itertools
import math

import pytest

from examples.ex02_feed_blending.check import (
    blend_cost,
    feasibility_errors,
    lagrangian_bound,
    optimality_gap,
    proves_infeasible,
)
from examples.ex02_feed_blending.data import broiler_grower, premium_finisher
from examples.ex02_feed_blending.model import (
    InfeasibleSpecError,
    elastic_blend,
    find_conflict,
    frontier,
    relaxable_limits,
    solve_blend,
)


@pytest.fixture(scope="module")
def grower():
    return broiler_grower()


@pytest.fixture(scope="module")
def premium():
    return premium_finisher()


@pytest.fixture(scope="module")
def blend(grower):
    return solve_blend(grower)


def test_blend_is_feasible(grower, blend):
    assert feasibility_errors(grower, blend.share) == []
    assert blend_cost(grower, blend.share) == pytest.approx(blend.cost)


def test_blend_is_provably_optimal(grower, blend):
    gap = optimality_gap(grower, blend.share, blend.total_price, blend.nutrient_price)
    assert gap == pytest.approx(0, abs=1e-6)


def test_known_optimum(blend):
    assert blend.cost == pytest.approx(317.8609, abs=1e-3)
    # Fish meal is too expensive; wheat bran adds no value at these prices.
    assert blend.share["fish meal"] == pytest.approx(0)
    assert blend.share["wheat bran"] == pytest.approx(0)


def test_no_random_blend_is_cheaper(grower, blend):
    """Sample many feasible blends; none may beat the optimum."""
    import random

    rng = random.Random(0)
    names = [i.name for i in grower.ingredients]
    found = 0
    for _ in range(20000):
        # Perturb the optimal blend; this finds feasible points far more
        # often than uniform sampling of the whole simplex.
        raw = {n: max(0.0, blend.share[n] + rng.gauss(0, 0.01)) for n in names}
        total = sum(raw.values())
        share = {n: v / total for n, v in raw.items()}
        if feasibility_errors(grower, share) == []:
            found += 1
            assert blend_cost(grower, share) >= blend.cost - 1e-6
    assert found > 50


@pytest.mark.parametrize("nutrient", ["protein", "calcium", "phosphorus", "energy"])
def test_shadow_price_predicts_cost_of_tighter_minimum(grower, blend, nutrient):
    req = grower.requirement(nutrient)
    step = 0.001 * req.min
    tighter = solve_blend(grower.with_requirement(nutrient, min=req.min + step))
    assert tighter.cost - blend.cost == pytest.approx(
        blend.nutrient_price[nutrient] * step, rel=1e-3
    )


def test_dual_bound_is_valid_for_any_prices(grower, blend):
    for t, yp, ye in itertools.product([-600, -300, 0], [0, 5, 20], [0, 0.1, 0.5]):
        prices = {r.nutrient: 0.0 for r in grower.requirements}
        prices.update(protein=yp, energy=ye)
        assert lagrangian_bound(grower, t, prices) <= blend.cost + 1e-6


def test_premium_spec_is_infeasible(premium):
    with pytest.raises(InfeasibleSpecError):
        solve_blend(premium)


def test_elastic_duals_prove_infeasibility(premium):
    elastic = elastic_blend(premium)
    assert elastic.total_violation > 0
    assert proves_infeasible(premium, elastic.total_price, elastic.nutrient_price)
    # Strong duality: the zero-cost bound equals the minimal violation.
    bound = lagrangian_bound(
        premium, elastic.total_price, elastic.nutrient_price, with_costs=False
    )
    assert bound == pytest.approx(elastic.total_violation, rel=1e-6)


def test_elastic_relaxation_is_feasible_and_minimal(premium):
    elastic = elastic_blend(premium)
    relaxed = premium
    for r in elastic.relaxations:
        relaxed = relaxed.with_requirement(r.limit.name, **{r.limit.side: r.new_value})
    assert feasibility_errors(relaxed, elastic.blend_share) == []
    solve_blend(relaxed)  # Must not raise.
    # Relaxing the energy minimum a bit less is not enough.
    (r,) = elastic.relaxations
    assert (r.limit.name, r.limit.side) == ("energy", "min")
    with pytest.raises(InfeasibleSpecError):
        solve_blend(premium.with_requirement("energy", min=r.new_value + 1.0))


def test_frontier_matches_elastic_result(grower, premium):
    # The elastic energy level is the highest energy at 24% protein.
    (point,) = frontier(grower, "protein", "energy", [24.0])
    assert point[1] == pytest.approx(elastic_blend(premium).relaxations[0].new_value)


def test_conflict_is_irreducible(premium):
    """Infeasible as a whole; feasible after dropping any single member."""
    conflict = find_conflict(premium)
    assert {str(c) for c in conflict} == {
        "protein >= 24",
        "energy >= 3200",
        "fish meal share <= 5%",
        "soybean oil share <= 5%",
    }
    for dropped in [None, *conflict]:
        data = _only(premium, [c for c in conflict if c is not dropped])
        if dropped is None:
            with pytest.raises(InfeasibleSpecError):
                solve_blend(data)
        else:
            solve_blend(data)  # Must not raise.


def _only(data, limits):
    """Return a copy of `data` that keeps only the given limits."""
    from dataclasses import replace

    keep = {(c.kind, c.name, c.side) for c in limits}
    reqs = tuple(
        replace(
            r,
            min=r.min if ("nutrient", r.nutrient, "min") in keep else 0.0,
            max=r.max if ("nutrient", r.nutrient, "max") in keep else math.inf,
        )
        for r in data.requirements
    )
    ings = tuple(
        replace(
            i, max_share=i.max_share if ("ingredient", i.name, "max") in keep else 1.0
        )
        for i in data.ingredients
    )
    return replace(data, requirements=reqs, ingredients=ings)


def test_feasible_spec_has_no_conflict(grower):
    assert find_conflict(grower) == []
    assert len(relaxable_limits(grower)) == 14
