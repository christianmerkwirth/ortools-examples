"""Tests for example 16: sports league scheduling."""

import pytest

from examples.ex16_sports_scheduling.check import (
    brute_force_min_breaks,
    patterns,
    schedule_errors,
    team_breaks,
    total_breaks,
)
from examples.ex16_sports_scheduling.circle import (
    circle_method,
    count_breaks,
    home_away,
    inverted,
    lower_bound,
    mirrored,
)
from examples.ex16_sports_scheduling.data import (
    plain_league,
    regional_league,
    with_rules,
)
from examples.ex16_sports_scheduling.model import (
    NoScheduleError,
    solve_league,
    solve_monolithic,
)

EVEN = [4, 6, 8, 10, 12, 16, 20]


@pytest.fixture(scope="module")
def league():
    return regional_league()


@pytest.fixture(scope="module")
def season(league):
    return solve_league(league)


def _named(n, schedule):
    teams = [t.name for t in plain_league(n).teams]
    return [[(teams[h], teams[a]) for h, a in games] for games in schedule]


# ------------------------------------------------------------- theory ---


@pytest.mark.parametrize("n", EVEN)
def test_circle_method_meets_every_bound(n):
    first = circle_method(n)
    for season, scheme in (
        (first, "single"),
        (mirrored(first), "mirrored"),
        (inverted(first), "phased"),
    ):
        assert sum(count_breaks(p) for p in home_away(season, n)) == lower_bound(
            n, scheme
        )


@pytest.mark.parametrize("n", EVEN)
def test_circle_method_schedules_are_valid(n):
    first = circle_method(n)
    for season in (mirrored(first), inverted(first)):
        assert schedule_errors(plain_league(n), _named(n, season)) == []


@pytest.mark.parametrize("scheme", ["single", "phased", "mirrored"])
def test_bounds_match_brute_force_for_four_teams(scheme):
    assert brute_force_min_breaks(4, scheme) == lower_bound(4, scheme)


@pytest.mark.parametrize("n", [4, 6])
@pytest.mark.parametrize("scheme", ["single", "phased", "mirrored"])
def test_direct_model_reaches_the_bounds(n, scheme):
    result = solve_monolithic(n, scheme, add_bound=True, time_limit=20)
    assert result.status == "OPTIMAL"
    assert result.breaks == lower_bound(n, scheme)


def test_direct_model_proves_the_bound_without_help():
    """Without the redundant bound, CP-SAT still proves 8 breaks for 6 teams."""
    result = solve_monolithic(6, "phased", add_bound=False, time_limit=30)
    assert result.status == "OPTIMAL"
    assert result.breaks == 8
    assert schedule_errors(plain_league(6), _named(6, result.schedule)) == []


@pytest.mark.parametrize("n", [6, 8, 10, 12, 16])
def test_template_model_reaches_the_bound(n):
    league = plain_league(n)
    season = solve_league(league)
    assert schedule_errors(league, season.rounds) == []
    assert (
        season.breaks == total_breaks(league, season.rounds) == lower_bound(n, "phased")
    )
    assert season.globally_optimal


# ------------------------------------------------------------- league ---


def test_league_season_is_valid(league, season):
    assert schedule_errors(league, season.rounds) == []


def test_league_season_is_globally_optimal(league, season):
    assert season.proven_optimal
    assert season.breaks == total_breaks(league, season.rounds) == 16
    assert season.breaks == lower_bound(len(league.teams), "phased")
    assert season.globally_optimal


def test_draw_is_a_permutation(league, season):
    assert sorted(season.draw.values()) == list(range(1, len(league.teams) + 1))


def test_local_rules_hold(league, season):
    p = patterns(league, season.rounds)
    assert p["Millbrook FC"][:2] == "AA"
    assert p["Harbor Town"][8] == "A"
    assert p["Northgate United"][0] == "H"
    derby = frozenset(league.derby)
    assert derby in {frozenset(g) for g in season.rounds[0]}
    assert derby in {frozenset(g) for g in season.rounds[-1]}
    for r in range(league.num_rounds):
        assert p["Riverside Rovers"][r] + p["Riverside Athletic"][r] != "HH"


def test_every_team_has_at_most_one_break_per_half(league, season):
    half = len(league.teams) - 1
    for rounds in team_breaks(league, season.rounds).values():
        assert sum(r <= half for r in rounds) <= 1
        assert sum(r > half for r in rounds) <= 1


def test_impossible_rules_are_detected(league):
    # Three away games in a row break the streak rule, so no schedule exists.
    banned = (*league.home_banned, ("Millbrook FC", 3))
    with pytest.raises(NoScheduleError):
        solve_league(with_rules(league, home_banned=banned))


def test_checker_catches_broken_rules(league, season):
    broken = [list(games) for games in season.rounds]
    h, a = broken[0][0]
    broken[0][0] = (a, h)  # Swap one venue.
    errors = schedule_errors(league, broken)
    assert any("hosts" in e for e in errors)
    errors = schedule_errors(league, season.rounds[1:] + season.rounds[:1])
    assert errors  # Rotating the season moves the derby and the opener.
