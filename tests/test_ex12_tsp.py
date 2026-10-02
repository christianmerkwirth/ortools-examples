"""Tests for example 12: traveling salesperson."""

import pytest

from examples.ex12_tsp.check import brute_force, held_karp, tour_errors, tour_length
from examples.ex12_tsp.data import TspData, cell_towers, random_sites
from examples.ex12_tsp.main import FIRST_SOLUTION_STRATEGIES
from examples.ex12_tsp.model import solve_exact, solve_with_routing

GLS = "GUIDED_LOCAL_SEARCH"


# ------------------------------------------------------------- the checker ---


def test_checker_rejects_bad_tours():
    data = random_sites(5, seed=0)
    assert tour_errors(data, [0, 1, 2, 3, 4, 0]) == []
    assert tour_errors(data, [0, 1, 2, 3, 0])  # Misses site 4.
    assert tour_errors(data, [0, 1, 2, 2, 4, 0])  # Visits 2 twice.
    assert tour_errors(data, [1, 0, 2, 3, 4, 1])  # Does not start at depot.


@pytest.mark.parametrize("n", [5, 7, 9])
def test_held_karp_matches_brute_force(n):
    data = random_sites(n, seed=n)
    length, route = held_karp(data)
    assert tour_errors(data, route) == []
    assert tour_length(data, route) == length
    assert length == brute_force(data)


# ------------------------------------------- small instances: exact optimum ---


@pytest.mark.parametrize("n, seed", [(6, 1), (8, 2), (10, 3), (12, 4), (13, 5)])
def test_routing_finds_the_optimum_on_small_instances(n, seed):
    data = random_sites(n, seed=seed)
    tour = solve_with_routing(data, metaheuristic=GLS, time_limit=0.3)
    assert tour_errors(data, tour.route) == []
    assert tour.length == tour_length(data, tour.route)
    assert tour.length == held_karp(data)[0]


@pytest.mark.parametrize("n, seed", [(8, 6), (12, 7)])
def test_cp_sat_matches_held_karp(n, seed):
    data = random_sites(n, seed=seed)
    tour = solve_exact(data, time_limit=10)
    assert tour.proven_optimal
    assert tour_errors(data, tour.route) == []
    assert tour.length == tour_length(data, tour.route) == held_karp(data)[0]
    assert tour.lower_bound == tour.length


def test_depot_need_not_be_site_zero():
    base = random_sites(9, seed=8)
    data = TspData("depot at site 4", base.sites, depot=4)
    tour = solve_with_routing(data, metaheuristic=GLS, time_limit=0.3)
    assert tour.route[0] == tour.route[-1] == 4
    assert tour_errors(data, tour.route) == []
    assert tour.length == held_karp(data)[0] == solve_exact(data).length


# --------------------------------------- larger instances: gap to optimum ---


@pytest.fixture(scope="module")
def towers():
    return cell_towers()


@pytest.mark.parametrize("seed", [3, 4, 5])
def test_guided_local_search_is_within_two_percent(seed):
    """60 sites: CP-SAT proves the optimum; GLS must come within 2%.

    The routing library stops on a wall-clock limit, so the exact tour can
    vary between machines. We test the gap, not the length.
    """
    data = random_sites(60, seed=seed)
    exact = solve_exact(data, time_limit=30)
    assert exact.proven_optimal
    tour = solve_with_routing(data, metaheuristic=GLS, time_limit=3.0)
    assert tour_errors(data, tour.route) == []
    assert exact.length <= tour.length <= 1.02 * exact.length


@pytest.mark.parametrize("strategy", FIRST_SOLUTION_STRATEGIES)
def test_every_first_solution_strategy_gives_a_valid_tour(towers, strategy):
    tour = solve_with_routing(towers, strategy, first_solution_only=True)
    assert tour_errors(towers, tour.route) == []
    assert tour.length == tour_length(towers, tour.route)


def test_local_search_never_makes_a_tour_worse(towers):
    for strategy in ("PATH_CHEAPEST_ARC", "SAVINGS"):
        first = solve_with_routing(towers, strategy, first_solution_only=True)
        improved = solve_with_routing(towers, strategy)
        assert improved.length <= first.length


def test_matrix_and_python_callback_agree(towers):
    """Both ways to give distances must lead to the same search."""
    a = solve_with_routing(towers, first_solution_only=True)
    b = solve_with_routing(towers, first_solution_only=True, python_callback=True)
    assert a.route == b.route
    assert a.length == b.length
