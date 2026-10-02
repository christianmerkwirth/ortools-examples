"""Tests for example 11: resource-constrained project scheduling."""

import pytest
from ortools.math_opt.python import mathopt

from examples.ex11_project_scheduling.check import (
    brute_force_makespan,
    critical_activities,
    critical_path_length,
    earliest_starts,
    energy_bound,
    feasibility_errors,
    latest_finish_rule,
    makespan,
    resource_profile,
    serial_schedule,
    topological_order,
)
from examples.ex11_project_scheduling.data import house, housing_row, random_project
from examples.ex11_project_scheduling.model import extra_unit_gains, solve_schedule


@pytest.fixture(scope="module")
def row():
    return housing_row(2)


@pytest.fixture(scope="module")
def schedule(row):
    return solve_schedule(row)


# ------------------------------------------------------------ the schedule --


def test_schedule_is_feasible(row, schedule):
    assert feasibility_errors(row, schedule.start) == []
    assert makespan(row, schedule.start) == schedule.makespan


def test_schedule_is_proven_optimal(schedule):
    assert schedule.proven_optimal
    assert schedule.lower_bound == schedule.makespan == 84


def test_lower_bounds(row, schedule):
    assert critical_path_length(row) == 57
    assert energy_bound(row) == 48
    assert max(critical_path_length(row), energy_bound(row)) <= schedule.makespan


def test_heuristic_is_feasible_but_worse(row, schedule):
    heuristic = latest_finish_rule(row)
    assert feasibility_errors(row, heuristic) == []
    assert makespan(row, heuristic) == 98 > schedule.makespan


def test_one_house_needs_only_three_days_more_than_the_critical_path():
    one = house()
    assert critical_path_length(one) == 57
    assert solve_schedule(one).makespan == 60


def test_extra_carpenter_is_the_only_crew_that_helps(row, schedule):
    gains = extra_unit_gains(row, schedule.makespan)
    assert gains == {
        "laborers": 0,
        "masons": 0,
        "carpenters": 6,
        "electricians": 0,
        "plumbers": 0,
        "crane": 0,
    }
    faster = solve_schedule(row.with_capacity("carpenters", 4))
    assert feasibility_errors(row.with_capacity("carpenters", 4), faster.start) == []


# ------------------------------------------------- independent referees --


def time_indexed_mip(data, upper_bound: int) -> int:
    """Solve the RCPSP as a time-indexed MIP with HiGHS; return the makespan.

    x[a, t] = 1 if activity a starts on day t. This is a completely
    different formulation from CP-SAT's intervals and cumulatives.
    """
    es = earliest_starts(data)
    # tail[a]: the shortest time from the start of a to the end of the
    # project (longest path through successors, including a itself).
    tail = {a.name: a.duration for a in data.activities}
    for name in reversed(topological_order(data)):
        for p in data.activity(name).predecessors:
            tail[p] = max(tail[p], data.activity(p).duration + tail[name])

    model = mathopt.Model(name="time-indexed rcpsp")
    x = {
        (a.name, t): model.add_binary_variable(name=f"x[{a.name},{t}]")
        for a in data.activities
        for t in range(es[a.name], upper_bound - tail[a.name] + 1)
    }
    starts = {a.name: [t for (n, t) in x if n == a.name] for a in data.activities}
    start_expr = {
        a.name: mathopt.fast_sum(t * x[a.name, t] for t in starts[a.name])
        for a in data.activities
    }
    for a in data.activities:
        model.add_linear_constraint(
            mathopt.fast_sum(x[a.name, t] for t in starts[a.name]) == 1
        )
        for p in a.predecessors:
            duration = data.activity(p).duration
            model.add_linear_constraint(start_expr[a.name] >= start_expr[p] + duration)
    for r in data.resources:
        for day in range(upper_bound):
            # An activity runs on `day` if it started in (day - duration, day].
            running = [
                a.needs(r.name) * x[a.name, t]
                for a in data.activities
                if a.needs(r.name)
                for t in starts[a.name]
                if day - a.duration < t <= day
            ]
            if running:
                model.add_linear_constraint(mathopt.fast_sum(running) <= r.capacity)
    end = model.add_variable(lb=0, ub=upper_bound, name="makespan")
    for a in data.activities:
        model.add_linear_constraint(end >= start_expr[a.name] + a.duration)
    model.minimize(end)
    result = mathopt.solve(model, mathopt.SolverType.HIGHS)
    assert result.termination.reason == mathopt.TerminationReason.OPTIMAL
    return round(result.objective_value())


def test_mip_agrees_on_one_house():
    one = house()
    ub = makespan(one, latest_finish_rule(one)) + 3  # Some room above the optimum.
    assert time_indexed_mip(one, ub) == solve_schedule(one).makespan


@pytest.mark.parametrize("seed", range(6))
def test_cp_sat_matches_brute_force_on_tiny_projects(seed):
    tiny = random_project(6, n_resources=2, seed=seed)
    result = solve_schedule(tiny, time_limit=5)
    assert result.proven_optimal
    assert feasibility_errors(tiny, result.start) == []
    assert result.makespan == brute_force_makespan(tiny)


# Seeds picked to keep the MIP fast; 100 and 105 start from a heuristic
# that is above the optimum, so the MIP has real work to do.
@pytest.mark.parametrize("seed", [100, 103, 105, 106])
def test_cp_sat_matches_mip_on_small_projects(seed):
    small = random_project(10, n_resources=3, seed=seed)
    result = solve_schedule(small, time_limit=5)
    ub = makespan(small, latest_finish_rule(small))
    assert result.makespan == time_indexed_mip(small, ub)


# ------------------------------------------------------ the checker itself --


def test_checker_catches_broken_schedules(row, schedule):
    early = dict(schedule.start)
    # Start drywall one day before insulation (its predecessor) ends.
    early["drywall [1]"] = early["insulation [1]"] + 2
    assert any(
        "before insulation [1] ends" in e for e in feasibility_errors(row, early)
    )

    crowded = {name: earliest_starts(row)[name] for name in schedule.start}
    errors = feasibility_errors(row, crowded)
    assert any("carpenters" in e for e in errors)


def test_serial_schedule_is_always_feasible():
    for seed in range(20):
        data = random_project(15, n_resources=3, seed=seed)
        order = topological_order(data)
        assert feasibility_errors(data, serial_schedule(data, order)) == []


def test_critical_activities_form_a_chain():
    one = house()
    chain = critical_activities(one)
    assert chain[0] == "survey and layout" and chain[-1] == "final inspection"
    assert sum(one.activity(n).duration for n in chain) >= critical_path_length(one)


def test_resource_profile_matches_demand(row, schedule):
    for r in row.resources:
        profile = resource_profile(row, schedule.start, r.name)
        assert sum(profile) == sum(a.duration * a.needs(r.name) for a in row.activities)
        assert max(profile) <= r.capacity
