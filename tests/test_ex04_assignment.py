"""Tests for example 04: assignment with a specialized solver and CP-SAT."""

from dataclasses import replace

import pytest
from ortools.math_opt.python import mathopt

from examples.ex04_assignment.check import (
    brute_force_day,
    brute_force_morning,
    certificate_errors,
    feasibility_errors,
    hungarian,
    plan_minutes,
)
from examples.ex04_assignment.data import (
    TECHNICIANS,
    AssignmentData,
    Job,
    Technician,
    full_day,
    morning,
    random_square,
)
from examples.ex04_assignment.model import (
    NoAssignmentError,
    solve_day,
    solve_morning,
    solve_morning_cp_sat,
)

# ---------------------------------------------------------------- Part A ---


@pytest.fixture(scope="module")
def morning_data():
    return morning()


@pytest.fixture(scope="module")
def morning_plan(morning_data):
    return solve_morning(morning_data)


def test_morning_plan_is_feasible(morning_data, morning_plan):
    assert feasibility_errors(morning_data, morning_plan.crew, one_job_each=True) == []
    assert plan_minutes(morning_data, morning_plan.crew) == morning_plan.minutes


def test_morning_known_optimum(morning_plan):
    assert morning_plan.minutes == 952
    assert morning_plan.crew["burst pipe"] == ("Hana",)


def test_morning_has_dual_certificate(morning_data, morning_plan):
    matrix = morning_data.cost_matrix()
    total, _, u, v = hungarian(matrix)
    assert total == morning_plan.minutes
    assert certificate_errors(matrix, morning_plan.minutes, u, v) == []


def test_certificate_rejects_a_worse_plan(morning_data, morning_plan):
    matrix = morning_data.cost_matrix()
    _, _, u, v = hungarian(matrix)
    assert certificate_errors(matrix, morning_plan.minutes + 1, u, v) != []


def test_morning_brute_force(morning_data, morning_plan):
    assert brute_force_morning(morning_data) == morning_plan.minutes  # 8! plans.


def test_cp_sat_agrees_on_morning(morning_data, morning_plan):
    cp_plan = solve_morning_cp_sat(morning_data)
    assert cp_plan.proven_optimal
    assert cp_plan.minutes == morning_plan.minutes
    assert feasibility_errors(morning_data, cp_plan.crew, one_job_each=True) == []


@pytest.mark.parametrize("seed", range(6))
def test_random_small_instances_match_brute_force(seed):
    data = random_square(7, seed=seed)
    expected = brute_force_morning(data)
    plan = solve_morning(data)
    assert plan.minutes == expected
    assert hungarian(data.cost_matrix())[0] == expected


@pytest.mark.parametrize("n", [30, 80])
def test_random_large_instances_have_certificates(n):
    data = random_square(n, seed=n)
    plan = solve_morning(data)
    matrix = data.cost_matrix()
    total, _, u, v = hungarian(matrix)
    assert total == plan.minutes
    assert certificate_errors(matrix, plan.minutes, u, v) == []
    assert feasibility_errors(data, plan.crew, one_job_each=True) == []


def test_no_perfect_assignment_is_detected():
    # Two plumbers, but only one plumbing job and one gas job.
    techs = (
        Technician("P1", "north", {"plumbing": 1.0}),
        Technician("P2", "east", {"plumbing": 1.0}),
    )
    jobs = (Job("tap", "plumbing", "north", 30), Job("boiler", "gas", "east", 60))
    data = AssignmentData(techs, jobs)
    with pytest.raises(NoAssignmentError):
        solve_morning(data)
    with pytest.raises(ValueError):
        hungarian(data.cost_matrix())


def test_non_square_is_rejected():
    with pytest.raises(ValueError):
        solve_morning(full_day())


# ---------------------------------------------------------------- Part B ---


@pytest.fixture(scope="module")
def day_data():
    return full_day()


@pytest.fixture(scope="module")
def day_plan(day_data):
    return solve_day(day_data)


def test_day_plan_is_feasible_and_optimal(day_data, day_plan):
    assert feasibility_errors(day_data, day_plan.crew) == []
    assert plan_minutes(day_data, day_plan.crew) == day_plan.minutes
    assert day_plan.proven_optimal
    assert day_plan.minutes == 2116


def test_day_plan_matches_mip(day_data, day_plan):
    """Solve the same model as a MIP with HiGHS: a second, independent solver."""
    techs, jobs = day_data.technicians, day_data.jobs
    cost = {
        (t, j): day_data.cost(techs[t], jobs[j])
        for t in range(len(techs))
        for j in range(len(jobs))
        if day_data.cost(techs[t], jobs[j]) is not None
    }
    model = mathopt.Model()
    x = {pair: model.add_binary_variable() for pair in cost}
    for j, job in enumerate(jobs):
        model.add_linear_constraint(
            mathopt.fast_sum(v for (_, jj), v in x.items() if jj == j) == job.team_size
        )
    for t, tech in enumerate(techs):
        model.add_linear_constraint(
            mathopt.fast_sum(cost[p] * v for p, v in x.items() if p[0] == t)
            <= tech.shift
        )
    model.minimize(mathopt.fast_sum(cost[p] * v for p, v in x.items()))
    result = mathopt.solve(model, mathopt.SolverType.HIGHS)
    assert result.termination.reason == mathopt.TerminationReason.OPTIMAL
    assert result.objective_value() == pytest.approx(day_plan.minutes)


def test_rule_costs_are_ordered(day_data, day_plan):
    free = solve_day(day_data, use_shifts=False, use_teams=False)
    teams = solve_day(day_data, use_shifts=False, use_teams=True)
    assert free.minutes <= teams.minutes <= day_plan.minutes
    assert (free.minutes, teams.minutes) == (1748, 2083)


def test_without_rules_each_job_goes_to_its_fastest_technician(day_data):
    free = solve_day(day_data, use_shifts=False, use_teams=False)
    fastest = sum(
        min(
            day_data.cost(t, j)
            for t in day_data.technicians
            if day_data.cost(t, j) is not None
        )
        for j in day_data.jobs
    )
    assert free.minutes == fastest
    assert (
        feasibility_errors(day_data, free.crew, use_shifts=False, use_teams=False) == []
    )


def test_tiny_day_matches_brute_force():
    techs = tuple(replace(t, shift=200) for t in TECHNICIANS[:4])
    jobs = (
        Job("tap", "plumbing", "north", 30),
        Job("sockets", "electrical", "east", 60),
        Job("radiator", "heating", "south", 45),
        Job("shelf", "carpentry", "west", 60, team_size=2),
        Job("leak", "plumbing", "south", 40),
        Job("boiler", "gas", "north", 50),
    )
    data = AssignmentData(techs, jobs)
    plan = solve_day(data)
    assert feasibility_errors(data, plan.crew) == []
    assert plan.minutes == brute_force_day(data)


def test_too_short_shifts_are_infeasible(day_data):
    short = AssignmentData(
        tuple(replace(t, shift=100) for t in day_data.technicians), day_data.jobs
    )
    with pytest.raises(NoAssignmentError):
        solve_day(short)


def test_checker_catches_broken_rules(day_data, day_plan):
    crew = dict(day_plan.crew)
    crew["burst pipe"] = ("Bo",)  # Bo is not a plumber.
    assert any("not qualified" in e for e in feasibility_errors(day_data, crew))
    crew = dict(day_plan.crew)
    crew["floor boards"] = crew["floor boards"][:1]  # A team job with one person.
    assert any("needs 2" in e for e in feasibility_errors(day_data, crew))
