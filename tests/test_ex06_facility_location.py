"""Tests for example 06: capacitated facility location MIP."""

import pytest
from ortools.math_opt.python import mathopt

from examples.ex06_facility_location.check import (
    brute_force_optimum,
    feasibility_errors,
    plan_cost,
)
from examples.ex06_facility_location.data import (
    random_region,
    small_region,
    wholesaler,
)
from examples.ex06_facility_location.model import (
    STRONG,
    WEAK,
    lp_bound,
    solve_location,
)

MIP_SOLVERS = [mathopt.SolverType.HIGHS, mathopt.SolverType.GSCIP]


@pytest.fixture(scope="module")
def data():
    return wholesaler()


@pytest.fixture(scope="module")
def plan(data):
    return solve_location(data)


def test_plan_is_feasible_and_cost_checks_out(data, plan):
    assert feasibility_errors(data, plan.open_sites, plan.assignment) == []
    fixed, transport = plan_cost(data, plan.open_sites, plan.assignment)
    assert fixed + transport == pytest.approx(plan.cost, rel=1e-9)


def test_known_optimum(plan):
    assert plan.proven_optimal
    assert plan.cost == pytest.approx(199_596.26, abs=0.01)
    assert sorted(plan.open_sites) == [
        "site B",
        "site F",
        "site G",
        "site H",
        "site K",
        "site L",
    ]
    assert plan.bound == pytest.approx(plan.cost, rel=1e-6)


@pytest.mark.parametrize("solver", MIP_SOLVERS, ids=lambda s: s.name)
@pytest.mark.parametrize("formulation", [WEAK, STRONG])
def test_solvers_and_formulations_agree(data, plan, solver, formulation):
    other = solve_location(data, formulation, solver)
    assert other.proven_optimal
    assert other.cost == pytest.approx(plan.cost, rel=1e-6)
    assert feasibility_errors(data, other.open_sites, other.assignment) == []


@pytest.mark.parametrize("solver", MIP_SOLVERS, ids=lambda s: s.name)
@pytest.mark.parametrize("formulation", [WEAK, STRONG])
def test_small_instance_matches_brute_force(solver, formulation):
    small = small_region()
    exact, exact_assignment = brute_force_optimum(small)
    found = solve_location(small, formulation, solver)
    assert found.cost == pytest.approx(exact, rel=1e-9)
    # The brute-force answer is itself a valid plan.
    used = sorted(set(exact_assignment.values()))
    assert feasibility_errors(small, used, exact_assignment) == []


def test_brute_force_on_other_small_instances():
    for seed in range(3):
        small = random_region(4, 7, seed=seed)
        exact, _ = brute_force_optimum(small)
        assert solve_location(small).cost == pytest.approx(exact, rel=1e-9)


def test_strong_lp_bound_is_tighter(data, plan):
    weak, strong = lp_bound(data, WEAK), lp_bound(data, STRONG)
    assert weak < strong <= plan.cost + 1e-6
    # The root gap shrinks from about 11% to below 0.5%.
    assert (plan.cost - weak) / plan.cost > 0.10
    assert (plan.cost - strong) / plan.cost < 0.005


def test_strong_lp_is_exact_with_loose_capacity(data):
    loose = data.with_capacity_factor(10)
    optimum = solve_location(loose).cost
    assert lp_bound(loose, STRONG) == pytest.approx(optimum, rel=1e-6)
    assert lp_bound(loose, WEAK) < 0.8 * optimum


def test_forced_site_counts_never_beat_the_optimum(data, plan):
    costs = {k: solve_location(data, force_open_count=k).cost for k in range(5, 10)}
    assert min(costs.values()) == pytest.approx(plan.cost, rel=1e-6)
    assert costs[len(plan.open_sites)] == pytest.approx(plan.cost, rel=1e-6)


def test_too_few_sites_is_infeasible(data):
    # Even the four largest sites hold less than total demand.
    largest = sorted((s.capacity for s in data.sites), reverse=True)[:4]
    assert sum(largest) < sum(c.demand for c in data.customers)
    with pytest.raises(RuntimeError):
        solve_location(data, force_open_count=4)


def test_gap_limit_gives_valid_bounds():
    large = random_region(25, 150, seed=6)
    p = solve_location(large, relative_gap=0.01, time_limit=30)
    assert feasibility_errors(large, p.open_sites, p.assignment) == []
    assert p.bound <= p.cost + 1e-6
    assert p.gap <= 0.01
