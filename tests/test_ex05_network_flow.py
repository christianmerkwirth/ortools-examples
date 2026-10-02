"""Tests for example 05: max flow, min cut, and min-cost flow."""

import itertools

import pytest
from ortools.math_opt.python import mathopt

from examples.ex05_network_flow.check import (
    arcs,
    cut_capacity,
    flow_errors,
    has_negative_cycle,
    marginal_costs,
    nodes,
)
from examples.ex05_network_flow.data import random_network, water_network
from examples.ex05_network_flow.model import (
    SINK,
    SOURCE,
    NoFeasibleFlowError,
    solve_max_flow,
    solve_min_cost_flow,
    upgrade_options,
)

UPGRADE = ("south plant", "east DC")


@pytest.fixture(scope="module")
def data():
    return water_network()


@pytest.fixture(scope="module")
def upgraded(data):
    return data.with_lane_capacity(*UPGRADE, data.lane(*UPGRADE).capacity + 60)


def lp_flow(data, min_cost: bool, fixed_value: int | None = None) -> float:
    """Solve the same problem as an LP with GLOP: a second, general solver.

    Without `min_cost`, maximize the flow into SINK. With it, minimize cost
    subject to delivering `fixed_value` pallets.
    """
    model = mathopt.Model()
    f = {(t, h): model.add_variable(lb=0, ub=c) for t, h, c, _ in arcs(data)}
    for n in nodes(data):
        if n in (SOURCE, SINK):
            continue
        inflow = mathopt.fast_sum(v for (t, h), v in f.items() if h == n)
        outflow = mathopt.fast_sum(v for (t, h), v in f.items() if t == n)
        model.add_linear_constraint(inflow == outflow)
    value = mathopt.fast_sum(v for (t, h), v in f.items() if h == SINK)
    if min_cost:
        model.add_linear_constraint(value == fixed_value)
        model.minimize(mathopt.fast_sum(c * f[t, h] for t, h, _, c in arcs(data)))
    else:
        model.maximize(value)
    result = mathopt.solve(model, mathopt.SolverType.GLOP)
    assert result.termination.reason == mathopt.TerminationReason.OPTIMAL
    return result.objective_value()


def test_max_flow_is_valid_and_matches_cut(data):
    result = solve_max_flow(data)
    flow = result.flow
    assert flow_errors(data, flow.arc_flow, flow.value, flow.cost) == []
    assert cut_capacity(data, set(result.source_side)) == flow.value == 1360
    # Every arc in the cut is full.
    assert all(flow.arc_flow[a.tail, a.head] == a.capacity for a in result.cut)


def test_max_flow_matches_lp(data):
    assert solve_max_flow(data).flow.value == pytest.approx(lp_flow(data, False))


def test_no_cut_is_smaller_than_the_max_flow(data):
    """Brute force: try every cut. The smallest equals the max flow."""
    inner = [n for n in nodes(data) if n not in (SOURCE, SINK)]
    best = min(
        cut_capacity(
            data, {SOURCE, *(n for n, b in zip(inner, bits, strict=True) if b)}
        )
        for bits in itertools.product([0, 1], repeat=len(inner))
    )
    assert best == solve_max_flow(data).flow.value


def test_original_network_cannot_meet_demand(data):
    with pytest.raises(NoFeasibleFlowError):
        solve_min_cost_flow(data)


def test_cheapest_max_flow(data):
    plan = solve_min_cost_flow(data, deliver_all=False)
    assert plan.value == 1360
    assert flow_errors(data, plan.arc_flow, plan.value, plan.cost) == []
    assert not has_negative_cycle(data, plan.arc_flow)
    assert plan.cost == pytest.approx(lp_flow(data, True, plan.value))


def test_max_flow_solver_ignores_cost(data):
    # The pure max-flow answer delivers the same amount but is not the
    # cheapest; the checker finds a negative cycle in it.
    flow = solve_max_flow(data).flow
    assert has_negative_cycle(data, flow.arc_flow)
    assert flow.cost > solve_min_cost_flow(data, deliver_all=False).cost


def test_upgraded_plan_is_optimal(upgraded):
    plan = solve_min_cost_flow(upgraded)
    assert plan.value == upgraded.total_demand == 1420
    assert plan.cost == 35420
    assert flow_errors(upgraded, plan.arc_flow, plan.value, plan.cost) == []
    assert not has_negative_cycle(upgraded, plan.arc_flow)
    assert plan.cost == pytest.approx(lp_flow(upgraded, True, plan.value))


def test_marginal_costs_are_potentials(upgraded):
    """Reduced costs c + pi_u - pi_v are >= 0 on every residual arc."""
    plan = solve_min_cost_flow(upgraded)
    pi = marginal_costs(upgraded, plan.arc_flow)
    for tail, head, cap, cost in arcs(upgraded):
        f = plan.arc_flow[tail, head]
        if f < cap and pi[tail] < float("inf"):
            assert cost + pi[tail] - pi[head] >= 0
        if f > 0 and pi[head] < float("inf"):
            assert -cost + pi[head] - pi[tail] >= 0
    assert (pi["Alder"], pi["Birch"], pi["Cedar"]) == (27, 28, 26)


def test_marginal_cost_predicts_one_more_pallet(upgraded):
    """Raise Cedar's demand by one pallet: the cost rises by its potential."""
    from dataclasses import replace

    plan = solve_min_cost_flow(upgraded)
    pi = marginal_costs(upgraded, plan.arc_flow)
    stores = tuple(
        replace(s, demand=s.demand + 1) if s.name == "Cedar" else s
        for s in upgraded.stores
    )
    more = solve_min_cost_flow(replace(upgraded, stores=stores))
    assert more.cost - plan.cost == pi["Cedar"]


def test_upgrade_options(data):
    options = {u.lane: u for u in upgrade_options(data, 60)}
    # Dunmore's lane is in the cut, but upgrading it does not help: a
    # second cut of the same size (Dunmore's demand) still holds.
    assert options["central DC", "Dunmore"].delivered == 1360
    assert options[UPGRADE].delivered == 1420
    best = min(
        (u for u in options.values() if u.delivered == 1420), key=lambda u: u.cost
    )
    assert best.lane == UPGRADE


@pytest.mark.parametrize("seed", range(8))
def test_random_networks_match_lp(seed):
    data = random_network(seed)
    result = solve_max_flow(data)
    flow = result.flow
    assert flow_errors(data, flow.arc_flow, flow.value, flow.cost) == []
    assert cut_capacity(data, set(result.source_side)) == flow.value
    assert flow.value == pytest.approx(lp_flow(data, False))

    plan = solve_min_cost_flow(data, deliver_all=False)
    assert plan.value == flow.value
    assert flow_errors(data, plan.arc_flow, plan.value, plan.cost) == []
    assert not has_negative_cycle(data, plan.arc_flow)
    assert plan.cost == pytest.approx(lp_flow(data, True, plan.value))
