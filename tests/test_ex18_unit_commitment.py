"""Tests for example 18: unit commitment."""

from dataclasses import replace

import pytest
from ortools.math_opt.python import mathopt

from examples.ex18_unit_commitment.check import (
    brute_force,
    cost_breakdown,
    feasibility_errors,
    merit_order_dispatch,
    min_up_down_errors,
)
from examples.ex18_unit_commitment.data import (
    cloudy_day,
    large_grid,
    sunny_day,
    tiny_grid,
)
from examples.ex18_unit_commitment.model import (
    STRONG,
    WEAK,
    hourly_prices,
    lp_bound,
    solve_commitment,
)


def errors_of(data, s):
    return feasibility_errors(data, s.on, s.power, s.solar_used, s.shed)


@pytest.fixture(scope="module")
def sunny():
    data = sunny_day()
    return data, solve_commitment(data)


@pytest.fixture(scope="module")
def cloudy():
    data = cloudy_day()
    return data, solve_commitment(data)


@pytest.mark.parametrize("day", ["sunny", "cloudy"])
def test_schedule_is_feasible_and_optimal(day, request):
    data, s = request.getfixturevalue(day)
    assert errors_of(data, s) == []
    assert s.proven_optimal
    assert sum(s.shed) == pytest.approx(0)
    total = cost_breakdown(data, s.on, s.power, s.shed)["total"]
    assert total == pytest.approx(s.cost, rel=1e-9)


def test_known_optimum(sunny, cloudy):
    assert sunny[1].cost == pytest.approx(868_200, rel=1e-6)
    assert cloudy[1].cost == pytest.approx(1_084_598, rel=1e-6)


@pytest.mark.parametrize("formulation", [STRONG, WEAK])
def test_tiny_grid_matches_brute_force(formulation):
    data = tiny_grid()
    best, plan = brute_force(data)
    s = solve_commitment(data, formulation)
    assert errors_of(data, s) == []
    assert s.cost == pytest.approx(best)
    assert best == pytest.approx(42_650)


@pytest.mark.parametrize("seed", range(4))
def test_random_tiny_grids_match_brute_force(seed):
    """Vary demand and solar on the tiny grid; MIP and brute force agree."""
    import random

    rng = random.Random(seed)
    base = tiny_grid()
    data = replace(
        base,
        demand=tuple(round(d * rng.uniform(0.8, 1.15)) for d in base.demand),
        solar=tuple(round(s * rng.uniform(0.0, 1.5)) for s in base.solar),
    )
    best, _ = brute_force(data)
    assert solve_commitment(data).cost == pytest.approx(best)


@pytest.mark.parametrize(
    "solver", [mathopt.SolverType.GSCIP, mathopt.SolverType.CP_SAT]
)
def test_other_solvers_agree(sunny, solver):
    data, s = sunny
    other = solve_commitment(data, solver=solver)
    assert other.proven_optimal
    assert other.cost == pytest.approx(s.cost, rel=1e-6)
    assert errors_of(data, other) == []


@pytest.mark.parametrize("make", [sunny_day, cloudy_day, lambda: large_grid(12, 1, 1)])
def test_strong_lp_bound_is_at_least_the_weak_one(make):
    data = make()
    strong, weak = lp_bound(data, STRONG), lp_bound(data, WEAK)
    assert strong >= weak - 1e-6
    assert strong <= solve_commitment(data).cost + 1e-6


def test_strong_bound_is_strictly_tighter_here(sunny):
    data, _ = sunny
    assert lp_bound(data, STRONG) > lp_bound(data, WEAK) + 1000


def test_weak_and_strong_have_the_same_optimum():
    data = large_grid(12, 1, 0)
    strong, weak = solve_commitment(data, STRONG), solve_commitment(data, WEAK)
    assert strong.cost == pytest.approx(weak.cost, rel=1e-6)
    assert errors_of(data, strong) == [] and errors_of(data, weak) == []


def test_fixed_plan_dispatch_cost_equals_mip_cost(sunny):
    data, s = sunny
    assert hourly_prices(data, s).dispatch_cost == pytest.approx(s.cost, rel=1e-9)


@pytest.mark.parametrize("hour", [3, 12, 19])
def test_price_predicts_cost_of_one_more_mw(sunny, hour):
    data, s = sunny
    prices = hourly_prices(data, s)
    demand = list(data.demand)
    demand[hour] += 1
    bigger = hourly_prices(replace(data, demand=tuple(demand)), s)
    assert bigger.dispatch_cost - prices.dispatch_cost == pytest.approx(
        prices.energy[hour], abs=1e-6
    )


def test_tiny_grid_prices_follow_the_merit_order():
    """Without ramps or water, the price is the cost of the partly loaded unit."""
    data = tiny_grid()
    s = solve_commitment(data)
    prices = hourly_prices(data, s)
    for t in range(data.hours):
        running = tuple(g for g in data.units if s.on[g.name][t])
        _, output = merit_order_dispatch(data, t, running)
        partial = [g for g in running if g.p_min < output[g.name] < g.p_max]
        curtailed = s.solar_used[t] < data.solar[t] - 1e-6
        if curtailed:
            assert prices.energy[t] == pytest.approx(0)
        elif len(partial) == 1 and prices.reserve[t] == pytest.approx(0):
            assert prices.energy[t] == pytest.approx(partial[0].marginal_cost)


def test_hydro_price_is_its_cost_plus_water_value(sunny):
    data, s = sunny
    prices = hourly_prices(data, s)
    dam = next(g for g in data.units if g.kind == "hydro")
    value = prices.water_value[dam.name]
    assert value > 0
    checked = 0
    for t in range(data.hours):
        p = s.power[dam.name][t]
        if dam.p_min + 1e-6 < p < dam.p_max - 1e-6 and prices.reserve[t] == 0:
            assert prices.energy[t] == pytest.approx(dam.marginal_cost + value)
            checked += 1
    assert checked >= 3


def test_midday_curtailment_makes_power_free(sunny):
    data, s = sunny
    prices = hourly_prices(data, s)
    for t in range(data.hours):
        assert prices.energy[t] >= -1e-6
        if s.solar_used[t] < data.solar[t] - 1e-6:
            assert prices.energy[t] == pytest.approx(0)
    assert sum(data.solar) - sum(s.solar_used) > 0


def test_cloudy_day_costs_more_and_wastes_no_sun(sunny, cloudy):
    assert cloudy[1].cost > sunny[1].cost
    data, s = cloudy
    assert sum(s.solar_used) == pytest.approx(sum(data.solar))


def test_short_grid_sheds_load_but_stays_feasible():
    data = sunny_day()
    data = replace(data, demand=tuple(1.2 * d for d in data.demand))
    s = solve_commitment(data)
    assert sum(s.shed) > 0
    assert errors_of(data, s) == []


def test_min_up_down_checker():
    unit = tiny_grid().units[0]  # min up 3, min down 2, on for 5 h before.
    assert min_up_down_errors(unit, (1, 1, 0, 0, 1, 1)) == []
    assert min_up_down_errors(unit, (1, 0, 1, 1, 1, 0)) != []  # Off only 1 h.
    assert min_up_down_errors(unit, (0, 0, 1, 0, 0, 0)) != []  # On only 1 h.
    # Runs cut off by the end of the day may be short; they go on tomorrow.
    assert min_up_down_errors(unit, (1, 1, 1, 1, 1, 0)) == []
    assert min_up_down_errors(unit, (1, 0, 0, 0, 0, 1)) == []
    fresh = replace(unit, initial_hours=1)  # Started 1 h ago: 2 more hours on.
    assert min_up_down_errors(fresh, (1, 0, 0, 0, 0, 0)) != []
    assert min_up_down_errors(fresh, (1, 1, 0, 0, 0, 0)) == []


def test_checker_catches_broken_schedules(sunny):
    data, s = sunny
    on = dict(s.on)
    power = dict(s.power)
    # Switch the second CCGT off for one hour in the middle of its run.
    name = "Lakeside CC2"
    t = s.on[name].index(1) + 2
    on[name] = tuple(0 if k == t else x for k, x in enumerate(on[name]))
    power[name] = tuple(0.0 if k == t else x for k, x in enumerate(power[name]))
    errors = feasibility_errors(data, on, power, s.solar_used, s.shed)
    assert any("supply" in e for e in errors)
    assert any("needs" in e for e in errors)
    # A nuclear jump of 200 MW breaks its ramp limit.
    power = dict(s.power)
    power["Riverbend"] = (500.0, *s.power["Riverbend"][1:])
    errors = feasibility_errors(data, s.on, power, s.solar_used, s.shed)
    assert any("ramps" in e for e in errors)
