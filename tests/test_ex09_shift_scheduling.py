"""Tests for example 09: nurse shift scheduling with CP-SAT."""

from dataclasses import replace

import pytest

from examples.ex09_shift_scheduling.check import (
    brute_force_optimum,
    hard_rule_errors,
    penalty_breakdown,
    total_penalty,
)
from examples.ex09_shift_scheduling.data import (
    Request,
    tiny_ward,
    ward_two_weeks,
)
from examples.ex09_shift_scheduling.model import (
    NoRosterError,
    night_fairness_tradeoff,
    solve_lexicographic,
    solve_roster,
)


@pytest.fixture(scope="module")
def data():
    return ward_two_weeks()


@pytest.fixture(scope="module")
def roster(data):
    return solve_roster(data)


def test_roster_meets_every_hard_rule(data, roster):
    assert hard_rule_errors(data, roster.assignment) == []


def test_penalties_match_independent_count(data, roster):
    assert penalty_breakdown(data, roster.assignment) == roster.penalty
    assert total_penalty(data, roster.assignment) == roster.objective


def test_roster_is_proven_optimal(roster):
    assert roster.optimal
    assert roster.objective == 14
    assert roster.best_bound == pytest.approx(14)
    assert roster.gap == 0


def test_saturday_conflict_breaks_exactly_two_wishes(data, roster):
    """Saturday needs 3 seniors; 4 of 5 seniors asked for it off.

    So exactly two of them must work. The cheapest choice breaks the two
    2-point wishes (Chen and Eli) and grants the 3-point ones.
    """
    working = {
        n.name
        for n in data.nurses
        if n.senior and roster.assignment[n.name, 5] is not None
    }
    assert {"Chen", "Eli"} <= working
    assert roster.assignment["Alex", 5] is None
    assert roster.assignment["Bea", 5] is None


def test_brute_force_agrees_on_tiny_instance():
    tiny = tiny_ward()
    best, best_roster = brute_force_optimum(tiny)
    cp = solve_roster(tiny)
    assert cp.optimal
    assert cp.objective == best == 7
    assert hard_rule_errors(tiny, best_roster) == []
    assert hard_rule_errors(tiny, cp.assignment) == []


def test_brute_force_agrees_with_other_weights():
    tiny = tiny_ward()
    for night, weekend in [(0, 0), (1, 9), (10, 1)]:
        variant = replace(
            tiny,
            weights=replace(tiny.weights, night_spread=night, weekend_spread=weekend),
        )
        assert solve_roster(variant).objective == brute_force_optimum(variant)[0]


def test_leave_is_respected_and_releasing_it_helps(data, roster):
    for d in data.nurse("Dana").leave:
        assert roster.assignment["Dana", d] is None
    # If Dana could work during their leave, the roster can only get better.
    free = solve_roster(data.with_nurse("Dana", leave=frozenset()))
    assert free.objective <= roster.objective


def test_new_heavy_request_is_honored(data, roster):
    # Pick the first shift that Kai works. A heavy wish for that day off
    # must move Kai off it, whatever optimal roster the solver found.
    day = next(d for d in range(data.num_days) if roster.assignment["Kai", d])
    wish = Request("Kai", day, None, 50)
    new = solve_roster(replace(data, requests=(*data.requests, wish)))
    assert new.assignment["Kai", day] is None
    assert hard_rule_errors(data, new.assignment) == []


def test_too_few_nurses_is_infeasible(data):
    # Without Alex and Bea, three seniors cannot cover 42 shifts.
    short = data.without_nurse("Alex").without_nurse("Bea")
    with pytest.raises(NoRosterError):
        solve_roster(short, time_limit=10)


def test_lexicographic_orders(data, roster):
    fair1, fair2 = solve_lexicographic(data, ("night spread", "weekend spread"))
    wish1, wish2 = solve_lexicographic(data, ("requests",))
    for r in (fair2, wish2):
        assert hard_rule_errors(data, r.assignment) == []
        assert penalty_breakdown(data, r.assignment) == r.penalty
    # Stage 2 never undoes stage 1.
    assert fair2.penalty["night spread"] + fair2.penalty["weekend spread"] == 0
    assert wish2.penalty["requests"] == wish1.penalty["requests"] == 4
    # Neither order can beat the weighted optimum on the weighted score.
    assert fair2.objective >= roster.objective
    assert wish2.objective >= roster.objective


def test_fairness_tradeoff_is_monotone(data):
    points = night_fairness_tradeoff(data, [0, 1, 2, 3, 4])
    wishes = [r.penalty["requests"] + r.penalty["isolated days off"] for _, r in points]
    assert wishes == [10, 8, 6, 4, 4]
    for k, r in points:
        assert hard_rule_errors(data, r.assignment) == []
        nights = [
            sum(r.assignment[n.name, d] == "N" for d in range(data.num_days))
            for n in data.nurses
            if n.full_time
        ]
        assert max(nights) - min(nights) <= k
