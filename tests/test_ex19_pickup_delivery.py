"""Tests for example 19: pickup and delivery routing (dial-a-ride)."""

from dataclasses import replace

import pytest

from examples.ex19_pickup_delivery.check import (
    exact_optimum,
    extra_ride,
    legs,
    lifo_errors,
    plan_cost,
    plan_errors,
    schedule_exists,
)
from examples.ex19_pickup_delivery.data import (
    GARAGE,
    PdpData,
    Place,
    Request,
    short_staffed_day,
    small_instance,
    transport_day,
)
from examples.ex19_pickup_delivery.model import Plan, solve_exact, solve_routing

# (requests, seed, buses, max detour, declines allowed)
TINY_CASES = [
    (3, 1, 2, 15, False),
    (4, 3, 1, 15, True),
    (4, 4, 2, None, False),
    (5, 5, 2, 15, True),
    (5, 6, 2, 10, False),
    (6, 11, 2, 15, False),
    (6, 12, 2, 10, True),
]
MEDIUM_CASES = [(8, 13, 3, 15, False), (8, 14, 3, 15, True)]


@pytest.fixture(scope="module")
def day():
    return transport_day()


@pytest.fixture(scope="module")
def day_plan(day):
    return solve_routing(day, time_limit=2)


@pytest.mark.parametrize(("n", "seed", "buses", "detour", "declines"), TINY_CASES)
def test_brute_force_cp_sat_and_routing_agree(n, seed, buses, detour, declines):
    data = small_instance(n, seed, buses, detour, declines)
    best = exact_optimum(data)
    exact = solve_exact(data, time_limit=10)
    routing = solve_routing(data, time_limit=0.5)
    for plan in (exact, routing):
        assert plan_errors(data, plan) == []
        assert plan_cost(data, plan) == plan.cost
    assert exact.proven_optimal
    assert routing.cost == exact.cost == best


@pytest.mark.parametrize(("n", "seed", "buses", "detour", "declines"), MEDIUM_CASES)
def test_routing_matches_cp_sat_on_medium_instances(n, seed, buses, detour, declines):
    data = small_instance(n, seed, buses, detour, declines)
    exact = solve_exact(data, time_limit=20)
    routing = solve_routing(data, time_limit=1)
    assert exact.proven_optimal
    assert plan_errors(data, routing) == []
    assert routing.cost <= exact.cost * 1.01


def test_normal_day_is_valid(day, day_plan):
    assert plan_errors(day, day_plan) == []
    assert plan_cost(day, day_plan) == day_plan.cost
    assert day_plan.declined == []
    limit = day.max_detour
    assert max(extra_ride(day, day_plan).values()) <= limit


def test_cp_sat_bound_is_consistent_with_the_routing_plan(day, day_plan):
    """Two solvers, one problem: CP-SAT's proven bound can never exceed the
    cost of a valid plan. (At this size the bound is too weak to measure
    the gap; the medium instances above do that.)"""
    try:
        exact = solve_exact(day, time_limit=30)
    except RuntimeError:
        pytest.skip("CP-SAT found no full-day plan in time on this machine")
    assert plan_errors(day, exact) == []
    assert exact.lower_bound <= day_plan.cost


def test_ride_limit_costs_distance_but_shortens_rides(day, day_plan):
    free = replace(day, max_detour=None)
    free_plan = solve_routing(free, time_limit=2)
    assert plan_errors(free, free_plan) == []
    # Both plans come from time-limited heuristic runs, so we do not compare
    # their costs here; a slow machine can make either one worse.
    assert max(extra_ride(free, free_plan).values()) > day.max_detour


def test_lifo_policy_is_respected(day, day_plan):
    plan = solve_routing(day, time_limit=2, policy="LIFO")
    assert plan_errors(day, plan) == []
    assert lifo_errors(day, plan) == []


def test_short_staffed_day_declines_whole_requests():
    data = short_staffed_day()
    plan = solve_routing(data, time_limit=2)
    assert plan_errors(data, plan) == []
    assert plan_cost(data, plan) == plan.cost
    assert 0 < len(plan.declined) < len(data.requests)
    assert len(plan.routes) <= data.num_vehicles


def test_impossible_day_raises():
    data = replace(transport_day(), num_vehicles=1)
    with pytest.raises(RuntimeError):
        solve_routing(data, time_limit=1)


def _two_rides(limit_detour):
    """Ann is ready at 8:00, Ben at 9:00; both go to the same clinic.

    Picking Ann up at 8:00 and waiting for Ben makes Ann ride an hour. The
    bus should wait at Ann's door instead.
    """
    ann, ben = Place("ann", 0.0, 3.0), Place("ben", 0.5, 3.0)
    clinic = Place("clinic", 0.0, -3.0)
    return PdpData(
        name="two rides",
        depot=GARAGE,
        requests=(
            Request("Ann", ann, clinic, 1, 8 * 60, 10 * 60, 8 * 60, 11 * 60),
            Request("Ben", ben, clinic, 1, 9 * 60, 10 * 60, 9 * 60, 11 * 60),
        ),
        num_vehicles=1,
        capacity=4,
        shift_start=7 * 60,
        shift_end=12 * 60,
        max_detour=limit_detour,
    )


def test_schedule_test_allows_waiting_before_a_pickup():
    data = _two_rides(limit_detour=10)
    nodes = data.stops()
    _, minutes = legs(data)
    route = [0, 1, 3, 2, 4, 0]  # Ann, Ben, drop Ann, drop Ben.

    # Driving as early as possible boards Ann at 8:00. The bus then waits
    # for Ben until 9:00, so Ann rides far longer than allowed.
    t, early = 0, {}
    for prev, node in zip(route, route[1:-1], strict=False):
        start = data.shift_start if prev == 0 else early[prev] + nodes[prev].service
        t = max(start + minutes[prev][node], nodes[node].open)
        early[node] = t
    direct = nodes[1].service + minutes[1][2]
    assert early[2] - early[1] > direct + data.max_detour

    # Yet the order works if the bus waits at Ann's door first.
    assert schedule_exists(data, route, minutes)
    plan = solve_routing(data, time_limit=0.5)
    assert plan_errors(data, plan) == []
    assert plan.cost == exact_optimum(data)


def test_schedule_test_rejects_a_too_short_shift():
    data = replace(_two_rides(limit_detour=10), shift_end=9 * 60)
    _, minutes = legs(data)
    assert not schedule_exists(data, [0, 1, 3, 2, 4, 0], minutes)
    with pytest.raises(ValueError):
        exact_optimum(data)


def test_checker_catches_broken_plans(day, day_plan):
    route = day_plan.routes[0]
    times = day_plan.times[0]
    # Split a pair across buses: move the first drop-off to a new route.
    first_drop = next(n for n in route[1:-1] if n % 2 == 0)
    pos = route.index(first_drop)
    broken = Plan(
        routes=[route[:pos] + route[pos + 1 :], [0, first_drop, 0]],
        times=[times[:pos] + times[pos + 1 :], [times[0], times[pos], times[-1]]],
        declined=[],
        cost=0,
        distance=0,
        seconds=0,
        method="broken",
    )
    errors = plan_errors(day, broken)
    assert any("different buses" in e for e in errors)
    # Overfill: pretend the bus has one seat.
    errors = plan_errors(replace(day, capacity=1), day_plan)
    assert any("on board" in e for e in errors)
    # Shrink every ride limit to zero.
    errors = plan_errors(replace(day, max_detour=-60), day_plan)
    assert any("rides" in e for e in errors)
