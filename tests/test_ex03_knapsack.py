"""Tests for example 03: knapsack and multiple knapsack."""

import pytest
from ortools.math_opt.python import mathopt

from examples.ex03_knapsack.check import (
    best_single_vehicle_value,
    brute_force_value,
    feasibility_errors,
    loading_value,
    pooled_bound,
)
from examples.ex03_knapsack.data import (
    LoadingData,
    Vehicle,
    cargo_plane,
    random_instance,
    truck_convoy,
)
from examples.ex03_knapsack.model import (
    solve_convoy,
    solve_with_cp_sat,
    solve_with_knapsack_solver,
)

SINGLE_SOLVERS = [solve_with_knapsack_solver, solve_with_cp_sat]


@pytest.fixture(scope="module")
def plane():
    return cargo_plane()


@pytest.fixture(scope="module")
def convoy():
    return truck_convoy()


@pytest.fixture(scope="module")
def convoy_loading(convoy):
    return solve_convoy(convoy)


@pytest.mark.parametrize("solve", SINGLE_SOLVERS, ids=lambda f: f.__name__)
def test_plane_loading_is_feasible_and_optimal(plane, solve):
    loading = solve(plane)
    assert feasibility_errors(plane, loading.assignment) == []
    assert loading_value(plane, loading.assignment) == loading.value
    assert loading.proven_optimal
    assert loading.value == best_single_vehicle_value(plane) == 840


@pytest.mark.parametrize("seed", range(6))
def test_dynamic_program_matches_brute_force(seed):
    data = random_instance(12, 1, seed=seed)
    assert best_single_vehicle_value(data) == brute_force_value(data)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("solve", SINGLE_SOLVERS, ids=lambda f: f.__name__)
def test_single_solvers_match_dynamic_program(seed, solve):
    data = random_instance(30, 1, seed=seed)
    loading = solve(data)
    assert feasibility_errors(data, loading.assignment) == []
    assert loading.value == best_single_vehicle_value(data)


def test_cold_chain_items_stay_off_a_plain_vehicle():
    data = cargo_plane()
    warm = LoadingData(data.items, (Vehicle("van", 4000, 15.0, refrigerated=False),))
    for solve in SINGLE_SOLVERS:
        loading = solve(warm)
        cold = {i.name for i in data.items if i.cold_chain}
        assert not cold & loading.assignment.keys()
        assert loading.value == best_single_vehicle_value(warm)


def test_convoy_is_feasible_and_proven_optimal(convoy, convoy_loading):
    assert feasibility_errors(convoy, convoy_loading.assignment) == []
    assert loading_value(convoy, convoy_loading.assignment) == convoy_loading.value
    assert convoy_loading.proven_optimal
    assert convoy_loading.value == 1125


def test_convoy_matches_a_mip_formulation(convoy, convoy_loading):
    """Solve the convoy again as a MIP with HiGHS: a second solver."""
    model = mathopt.Model()
    x = {
        (i.name, v.name): model.add_binary_variable()
        for i in convoy.items
        for v in convoy.vehicles
        if v.refrigerated or not i.cold_chain
    }
    for i in convoy.items:
        model.add_linear_constraint(
            sum(x[i.name, v.name] for v in convoy.vehicles if (i.name, v.name) in x)
            <= 1
        )
    for v in convoy.vehicles:
        on = [(i, x[i.name, v.name]) for i in convoy.items if (i.name, v.name) in x]
        model.add_linear_constraint(sum(i.weight * y for i, y in on) <= v.max_weight)
        model.add_linear_constraint(
            sum(round(10 * i.volume) * y for i, y in on) <= round(10 * v.max_volume)
        )
        for a, b in convoy.incompatible:
            if (a, v.name) in x and (b, v.name) in x:
                model.add_linear_constraint(x[a, v.name] + x[b, v.name] <= 1)
    value = {i.name: i.value for i in convoy.items}
    model.maximize(sum(value[i] * y for (i, _), y in x.items()))
    result = mathopt.solve(model, mathopt.SolverType.HIGHS)
    assert result.termination.reason == mathopt.TerminationReason.OPTIMAL
    assert result.objective_value() == pytest.approx(convoy_loading.value)


def test_convoy_reaches_the_pooled_bound(convoy, convoy_loading):
    """One big pooled truck is a relaxation, so its optimum bounds the convoy.

    Here the convoy reaches the bound, which proves it optimal without
    trusting CP-SAT.
    """
    assert convoy_loading.value == pooled_bound(convoy)


@pytest.mark.parametrize("seed", range(4))
def test_pooled_bound_is_valid_on_random_instances(seed):
    data = random_instance(20, 3, seed=seed)
    assert solve_convoy(data).value <= pooled_bound(data)


@pytest.mark.parametrize("seed", range(4))
def test_convoy_matches_brute_force_on_small_instances(seed):
    data = random_instance(8, 2, seed=seed, tightness=0.5)
    names = [i.name for i in data.items[:2]]
    data = LoadingData(data.items, data.vehicles, incompatible=(tuple(names),))
    loading = solve_convoy(data)
    assert feasibility_errors(data, loading.assignment) == []
    assert loading.value == brute_force_value(data)


def test_hazardous_pair_never_shares_a_truck(convoy, convoy_loading):
    for a, b in convoy.incompatible:
        if a in convoy_loading.assignment and b in convoy_loading.assignment:
            assert convoy_loading.assignment[a] != convoy_loading.assignment[b]


def test_side_rules_never_help(convoy, convoy_loading):
    free = LoadingData(convoy.items, convoy.vehicles)  # No hazmat rule.
    assert solve_convoy(free).value >= convoy_loading.value


def test_checker_catches_broken_rules(convoy):
    over = {i.name: "medium truck" for i in convoy.items}
    errors = feasibility_errors(convoy, over)
    assert any("kg" in e for e in errors)
    assert any("fridge" in e for e in errors)
    assert any("carries both" in e for e in errors)
