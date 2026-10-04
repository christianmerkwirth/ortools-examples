"""Tests for example 20: season planning under uncertain demand."""

import numpy as np
import pytest

from examples.ex20_stochastic_planning.check import (
    best_plan,
    plan_errors,
    product_profit,
    profits,
)
from examples.ex20_stochastic_planning.data import (
    PlanningData,
    Product,
    Scenarios,
    point_forecast,
    sample,
)
from examples.ex20_stochastic_planning.model import (
    average_plan,
    solve_each,
    solve_plan,
)

DATA = PlanningData()


@pytest.fixture(scope="module")
def det():
    return solve_plan(DATA, point_forecast(DATA))


@pytest.fixture(scope="module")
def train():
    return sample(DATA, 0.3, 200, seed=1)


@pytest.fixture(scope="module")
def test_set():
    return sample(DATA, 0.3, 20_000, seed=2)


@pytest.fixture(scope="module")
def sto(train):
    return solve_plan(DATA, train)


# ------------------------------------------------------- deterministic ---


def test_point_forecast_plan(det):
    # 8 lines give 8,000 hours: all parkas, all gloves that pay off, and
    # fleece in the hours left. A 9th line would not pay for itself.
    assert det.proven_optimal
    assert det.plan.lines == 8
    assert det.plan.make == pytest.approx([2000, 2625, 5500])
    assert det.promised == pytest.approx(290_625)


def test_point_forecast_matches_referee(det):
    lines, make, value = best_plan(DATA, point_forecast(DATA))
    assert value == pytest.approx(det.promised, rel=1e-9)
    assert lines == det.plan.lines


# ---------------------------------------------------------- stochastic ---


@pytest.mark.parametrize("spread", [0.1, 0.3, 0.5])
@pytest.mark.parametrize("count", [1, 25, 200])
def test_stochastic_optimum_matches_referee(spread, count):
    scen = sample(DATA, spread, count, seed=7)
    sol = solve_plan(DATA, scen)
    _, _, value = best_plan(DATA, scen)
    assert sol.proven_optimal
    assert sol.promised == pytest.approx(value, rel=1e-7)
    assert plan_errors(DATA, sol.plan.lines, sol.plan.make) == []


def test_objective_equals_direct_playout(sto, train):
    direct = profits(DATA, sto.plan.lines, sto.plan.make, train).mean()
    assert sto.promised == pytest.approx(direct, rel=1e-9)


def test_zero_spread_gives_the_point_forecast_plan(det):
    sol = solve_plan(DATA, sample(DATA, 0.0, 20, seed=3))
    assert sol.promised == pytest.approx(det.promised)
    assert sol.plan.make == pytest.approx(det.plan.make)


def test_ws_rp_eev_order(det, sto, train):
    """Perfect information >= stochastic >= point-forecast plan (same scenarios)."""
    ws = np.mean([r.promised for r in solve_each(DATA, train)])
    eev = profits(DATA, det.plan.lines, det.plan.make, train).mean()
    assert ws >= sto.promised - 1e-6
    assert sto.promised >= eev - 1e-6
    # The point forecast over-promises: Jensen's inequality for a concave profit.
    assert det.promised > eev


def test_stochastic_plan_wins_on_fresh_scenarios(det, sto, test_set):
    a = profits(DATA, sto.plan.lines, sto.plan.make, test_set)
    b = profits(DATA, det.plan.lines, det.plan.make, test_set)
    diff = a - b
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    assert diff.mean() > 5 * se  # Clearly better on average,
    assert np.percentile(a, 5) > np.percentile(b, 5)  # and in a bad year.


def test_value_of_the_stochastic_solution_grows_with_spread(det):
    vss = []
    for spread in (0.1, 0.3, 0.5):
        plan = solve_plan(DATA, sample(DATA, spread, 200, seed=1)).plan
        fresh = sample(DATA, spread, 20_000, seed=2)
        gain = profits(DATA, plan.lines, plan.make, fresh) - profits(
            DATA, det.plan.lines, det.plan.make, fresh
        )
        vss.append(gain.mean())
    assert vss[0] < vss[1] < vss[2]
    assert vss[2] > 10_000


def test_referee_plan_beats_random_feasible_plans(train):
    lines, make, value = best_plan(DATA, train)
    rng = np.random.default_rng(0)
    hours = DATA.column("hours")
    for _ in range(200):
        n = int(rng.integers(0, DATA.plant.max_lines + 1))
        x = make * rng.uniform(0.7, 1.3, size=3)
        x *= min(1.0, DATA.plant.line_hours * n / max(1e-9, hours @ x))
        assert profits(DATA, n, x, train).mean() <= value + 1e-6


# ------------------------------------------------------------ what-if ---


def test_what_if_runs_disagree_and_average_fits(train):
    runs = solve_each(DATA, train.subset(range(40)))
    assert len({r.plan.lines for r in runs}) >= 3
    plan = average_plan(DATA, runs)
    assert plan_errors(DATA, plan.lines, plan.make) == []


# --------------------------------------------------------------- data ---


def test_sample_keeps_the_forecast_mean_and_clips_prices():
    scen = sample(DATA, 0.5, 200_000, seed=5)
    assert scen.demand.mean(axis=0) == pytest.approx(
        DATA.column("mean_demand"), rel=0.01
    )
    assert (scen.rush_price >= DATA.column("salvage")).all()
    assert (scen.rush_price <= DATA.column("price")).all()


# ------------------------------------------------------------- checks ---


def test_product_profit_by_hand():
    p = Product(
        "coat",
        price=100,
        cost=40,
        salvage=10,
        rush_price=70,
        rush_cap=5,
        hours=1,
        mean_demand=0,
    )
    data = PlanningData((p,))
    make = np.array([20.0])
    scen = Scenarios(
        np.array([[12.0], [23.0], [40.0], [30.0]]),
        np.array([[70.0], [70.0], [70.0], [100.0]]),
    )
    got = product_profit(data, make, scen)[:, 0]
    # 12: sell 12, 8 to outlet.          1200 + 80 - 800       = 480
    # 23: 3 express, sell 23.            2300 - 210 - 800      = 1290
    # 40: 5 express (cap), sell 25.      2500 - 350 - 800      = 1350
    # 30: express costs the sale price, so buy none; sell 20.  = 1200
    assert got == pytest.approx([480, 1290, 1350, 1200])


def test_checker_catches_broken_plans():
    assert plan_errors(DATA, 8, np.array([2000, 2625, 5500])) == []
    text = " ".join(plan_errors(DATA, 2, np.array([2000, -5, 0])))
    assert "negative" in text
    assert "sewing hours" in text
    assert plan_errors(DATA, 13, np.zeros(3))
