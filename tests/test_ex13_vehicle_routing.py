"""Tests for example 13: vehicle routing with capacities and time windows."""

from dataclasses import replace

import pytest

from examples.ex13_vehicle_routing.check import (
    exact_optimum,
    legs,
    plan_cost,
    plan_errors,
    replay,
    shortest_span,
    simple_lower_bound,
)
from examples.ex13_vehicle_routing.data import (
    flu_season_day,
    paid_time_day,
    pharmacy_day,
    small_instance,
)
from examples.ex13_vehicle_routing.model import Plan, solve_exact, solve_routing

# (customers, seed, drops allowed, minute cost)
SMALL_CASES = [
    (6, 1, False, 0),
    (8, 3, False, 500),
    (8, 4, True, 500),
    (7, 5, True, 1000),
    (9, 6, False, 1000),
    (9, 8, True, 300),
    (10, 9, False, 0),
]


@pytest.fixture(scope="module")
def normal():
    return pharmacy_day()


@pytest.fixture(scope="module")
def normal_plan(normal):
    return solve_routing(normal, time_limit=2)


@pytest.mark.parametrize(("n", "seed", "drops", "minute_cost"), SMALL_CASES)
def test_routing_finds_the_exact_optimum_on_small_instances(
    n, seed, drops, minute_cost
):
    """Brute force, CP-SAT, and the routing library must all agree."""
    data = small_instance(n, seed, drops=drops, minute_cost=minute_cost)
    best = exact_optimum(data)
    routing = solve_routing(data, time_limit=0.5)
    exact = solve_exact(data, time_limit=10)
    assert plan_errors(data, routing) == []
    assert plan_errors(data, exact) == []
    assert exact.proven_optimal
    assert routing.cost == exact.cost == best


def test_normal_day_plan_is_feasible(normal, normal_plan):
    assert plan_errors(normal, normal_plan) == []
    assert normal_plan.dropped == []


def test_normal_day_uses_the_fewest_possible_vans(normal, normal_plan):
    vans, _ = simple_lower_bound(normal)
    assert len(normal_plan.routes) == vans == 7


def test_normal_day_is_close_to_optimal(normal, normal_plan):
    """CP-SAT's proven lower bound limits how far from optimal the plan can be."""
    exact = solve_exact(normal, time_limit=10)
    assert plan_errors(normal, exact) == []
    assert exact.lower_bound <= normal_plan.cost
    assert normal_plan.cost <= 1.07 * exact.lower_bound


def test_simple_bound_is_valid_on_small_instances():
    for n, seed, _, _ in SMALL_CASES:
        data = small_instance(n, seed)
        assert simple_lower_bound(data)[1] <= exact_optimum(data)


def test_flu_day_drops_only_what_does_not_fit():
    data = flu_season_day()
    plan = solve_routing(data, time_limit=2)
    assert plan_errors(data, plan) == []
    demand = sum(s.demand for s in data.stops)
    dropped = sum(data.stops[c].demand for c in plan.dropped)
    capacity = data.num_vehicles * data.capacity
    assert dropped >= demand - capacity  # Some crates cannot fit.
    assert dropped <= demand - capacity + 8  # But not many more.
    assert len(plan.routes) == data.num_vehicles


def test_paid_time_cuts_working_hours(normal_plan):
    data = paid_time_day()
    plan = solve_routing(data, time_limit=2)
    assert plan_errors(data, plan) == []

    def working(p: Plan) -> int:
        return sum(t[-1] - t[0] for t in p.start_times)

    assert working(plan) < 0.75 * working(normal_plan)
    # Valued at the paid rate, the paid-time plan is cheaper.
    old = replace(normal_plan, cost=plan_cost(data, normal_plan))
    assert plan.cost < old.cost


def test_drops_not_allowed_means_infeasible():
    data = small_instance(8, 3)
    tight = replace(data, num_vehicles=1)
    with pytest.raises(ValueError):
        exact_optimum(tight)
    with pytest.raises(RuntimeError):
        solve_routing(tight, time_limit=0.5)


@pytest.mark.parametrize("seed", range(5))
def test_shortest_span_matches_trying_every_departure(seed):
    data = small_instance(9, seed, minute_cost=1)
    _, minutes = legs(data)
    plan = solve_routing(data, time_limit=0.5)
    for route in plan.routes:
        spans = []
        for leave in range(data.shift_start, data.shift_end + 1):
            result = replay(data, route, minutes, leave)
            if result is not None and result[1] <= data.shift_end:
                spans.append(result[1] - leave)
        assert shortest_span(data, route, minutes) == min(spans)


def test_checker_catches_broken_plans(normal, normal_plan):
    first = normal_plan.routes[0]
    # Move all customers onto one van: overloaded and too late.
    everyone = [0, *[c for r in normal_plan.routes for c in r[1:-1]], 0]
    one_van = replace(
        normal_plan,
        routes=[everyone],
        start_times=[[normal.shift_start] * len(everyone)],
    )
    errors = plan_errors(normal, one_van)
    assert any("crates" in e for e in errors)
    assert any("window" in e for e in errors)

    # Forget a customer.
    short = replace(
        normal_plan,
        routes=[[0, *first[2:]], *normal_plan.routes[1:]],
        start_times=[normal_plan.start_times[0][1:], *normal_plan.start_times[1:]],
    )
    assert any("unserved" in e for e in plan_errors(normal, short))

    # Misreport the cost.
    cheap = replace(normal_plan, cost=normal_plan.cost - 1)
    assert any("cost" in e for e in plan_errors(normal, cheap))
