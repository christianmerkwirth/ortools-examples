"""Tests for example 10: job shop scheduling with CP-SAT."""

import itertools

import pytest

from examples.ex10_job_shop.check import (
    brute_force_makespan,
    critical_path,
    end_time,
    feasibility_errors,
    lower_bound,
    makespan,
    weighted_tardiness,
)
from examples.ex10_job_shop.data import (
    ft06,
    la01,
    machine_shop,
    random_instance,
    tiny,
)
from examples.ex10_job_shop.model import solve, solve_lexicographic


@pytest.fixture(scope="module")
def shop():
    return machine_shop()


@pytest.fixture(scope="module")
def shop_schedule(shop):
    return solve(shop)


@pytest.mark.parametrize("make", [ft06, la01], ids=["ft06", "la01"])
def test_benchmark_reaches_known_optimum(make):
    data = make()
    s = solve(data, time_limit=30)
    assert feasibility_errors(data, s.start) == []
    assert makespan(data, s.start) == s.makespan == data.known_optimum
    assert s.optimal


def test_ft06_cannot_beat_55():
    # With the makespan capped one below the optimum, no schedule exists.
    with pytest.raises(RuntimeError, match="INFEASIBLE"):
        solve(ft06(), max_makespan=54)


def test_la01_optimum_equals_machine_bound():
    # For la01 the busiest machine alone proves the optimum.
    assert lower_bound(la01())["busiest machine"] == 666


def test_tiny_matches_brute_force():
    data = tiny()
    assert solve(data).makespan == brute_force_makespan(data) == 11


@pytest.mark.parametrize(
    ("n_jobs", "n_machines", "seed"),
    [(3, 3, s) for s in range(5)] + [(4, 3, 0), (3, 4, 1)],
)
def test_random_small_instances_match_brute_force(n_jobs, n_machines, seed):
    data = random_instance(n_jobs, n_machines, seed)
    s = solve(data)
    assert feasibility_errors(data, s.start) == []
    assert s.optimal
    assert s.makespan == brute_force_makespan(data)


def test_story_instance_optimum(shop, shop_schedule):
    assert feasibility_errors(shop, shop_schedule.start) == []
    assert shop_schedule.makespan == 200
    assert shop_schedule.optimal
    assert max(lower_bound(shop).values()) == 195


def test_trade_off_between_goals(shop):
    fast = solve_lexicographic(shop, "makespan", "tardiness")
    on_time = solve_lexicographic(shop, "tardiness", "makespan")
    for s in (fast, on_time):
        assert feasibility_errors(shop, s.start) == []
        assert s.optimal
    assert (fast.makespan, weighted_tardiness(shop, fast.start)) == (200, 110)
    assert (on_time.makespan, weighted_tardiness(shop, on_time.start)) == (215, 40)


def test_tardiness_objective_matches_checker(shop):
    s = solve(shop, "tardiness")
    assert s.objective == weighted_tardiness(shop, s.start) == 40


@pytest.mark.parametrize("make", [tiny, ft06, la01, machine_shop])
def test_lower_bounds_are_valid(make):
    data = make()
    best = solve(data, time_limit=30).makespan
    assert all(b <= best for b in lower_bound(data).values())


@pytest.mark.parametrize("make", [ft06, machine_shop])
def test_critical_path_spans_the_schedule(make):
    data = make()
    s = solve(data)
    path = critical_path(data, s.start)
    # The chain starts at time 0, has no gaps, and ends at the makespan.
    assert s.start[path[0]] == 0
    for a, b in itertools.pairwise(path):
        assert end_time(data, s.start, a) == s.start[b]
    assert end_time(data, s.start, path[-1]) == s.makespan


def test_checker_catches_broken_schedules(shop, shop_schedule):
    start = dict(shop_schedule.start)

    # Route order: move a job's second operation to time 0.
    broken = start | {("pump housing", 1): 0}
    assert any("starts before" in e for e in feasibility_errors(shop, broken))

    # Machine overlap: start two saw operations at the same time.
    first_saw = [k for k in start if k[1] == 0 and k[0] in ("bracket", "coupling")]
    broken = start | {first_saw[0]: 0, first_saw[1]: 0}
    assert any("overlaps" in e for e in feasibility_errors(shop, broken))

    # A missing operation.
    broken = {k: v for k, v in start.items() if k != ("spindle", 1)}
    assert feasibility_errors(shop, broken) != []


def test_progress_is_monotone():
    s = solve(random_instance(10, 5, seed=3), record_progress=True)
    values = [v for _, v, _ in s.progress]
    bounds = [b for _, _, b in s.progress]
    assert all(b <= a for a, b in itertools.pairwise(values))
    assert all(b >= a for a, b in itertools.pairwise(bounds))
    assert s.optimal and values[-1] == bounds[-1] == s.makespan


def test_tardiness_needs_due_dates():
    with pytest.raises(ValueError, match="due date"):
        solve(ft06(), "tardiness")
