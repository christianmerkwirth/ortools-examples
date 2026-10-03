"""League scheduling with CP-SAT: two models for the same problem.

Decision: who plays whom, where, in which round.
Hard rules: every team plays once per round; every pair meets once at
each ground, once in each half; plus the league's local rules.
Goal: as few breaks (two home or two away games in a row) as possible.

`solve_monolithic` states the problem directly: one Boolean per (home,
away, round). It is the textbook model, and it struggles. Teams are
interchangeable, so the solver sees countless equivalent schedules, and
its lower bound stays weak.

`solve_league` uses the trick that many real leagues use. Start from a
fixed template built by the circle method, with *slots* 1 .. n in place of
teams. CP-SAT then only decides (a) which team gets which slot (the
"draw") and (b) who is at home in each template game. This model is tiny
and solves in a fraction of a second.

A template limits the search, so how can it be optimal? Theory gives a
lower bound on breaks for EVERY phased schedule (see circle.py). When the
template model reaches that bound, no schedule at all can do better.
"""

import itertools
from dataclasses import dataclass

from ortools.sat.python import cp_model

from .circle import circle_method, home_away, lower_bound
from .data import League


class NoScheduleError(RuntimeError):
    """Raised when CP-SAT proves that no schedule meets the rules."""


@dataclass(frozen=True)
class LeagueSchedule:
    """A solved season."""

    rounds: list[list[tuple[str, str]]]  # Per round: (home, away) team names.
    draw: dict[str, int]  # Team name -> template slot (1-based).
    breaks: int  # Recounted from the schedule.
    proven_optimal: bool  # Optimal among all template schedules.
    globally_optimal: bool  # Reaches the bound for ALL phased schedules.
    best_bound: float
    wall_time: float


@dataclass(frozen=True)
class MonolithicResult:
    """The outcome of the direct model."""

    status: str  # "OPTIMAL", "FEASIBLE", or "UNKNOWN".
    breaks: int | None
    best_bound: float
    wall_time: float
    schedule: list[list[tuple[int, int]]] | None


def _solver(time_limit: float, workers: int = 8) -> cp_model.CpSolver:
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = workers
    return solver


# ------------------------------------------------------- template model ---


def solve_league(league: League, time_limit: float = 10.0) -> LeagueSchedule:
    """Schedule the league on a circle-method template with CP-SAT."""
    teams = [t.name for t in league.teams]
    n = len(teams)
    half = n - 1
    slots = range(n)
    rounds = range(league.num_rounds)
    template = circle_method(n)  # First half; slots stand in for teams.
    model = cp_model.CpModel()

    # home[k, r] is true if slot k plays at home in round r. Each template
    # game gets ONE Boolean: true means its first slot is at home. The
    # other slot is then away, so we use the negated literal for it.
    home: dict[tuple[int, int], cp_model.IntVar] = {}
    for r, games in enumerate(template):
        for a, b in games:
            first_at_home = model.new_bool_var(f"slot{a}_home_r{r}")
            home[a, r] = first_at_home
            home[b, r] = first_at_home.Not()
    # Second half: the first-half rounds in reverse order, venues swapped
    # (the "inverted" scheme). It adds no break at the turn of the season.
    for s in range(half):
        for k in slots:
            home[k, half + s] = home[k, half - 1 - s].Not()

    # draw[t, k] is true if team t gets slot k: a one-to-one assignment.
    draw = {(t, k): model.new_bool_var(f"{t}_slot{k}") for t in teams for k in slots}
    for t in teams:
        model.add_exactly_one(draw[t, k] for k in slots)
    for k in slots:
        model.add_exactly_one(draw[t, k] for t in teams)

    # Local rules. Each rule is about a team, but home/away lives on slots,
    # so we write "IF team t has slot k THEN slot k ..." with enforcement.
    for team, rnd in league.home_banned:
        for k in slots:
            model.add_bool_and(home[k, rnd - 1].Not()).only_enforce_if(draw[team, k])
    if league.home_opener:
        for k in slots:
            model.add_bool_and(home[k, 0]).only_enforce_if(draw[league.home_opener, k])
    for a, b in league.shared_stadiums():
        for k, j in itertools.permutations(slots, 2):
            for r in rounds:
                # Not (a has k AND b has j AND both slots at home in r).
                model.add_bool_or(
                    [
                        draw[a, k].Not(),
                        draw[b, j].Not(),
                        home[k, r].Not(),
                        home[j, r].Not(),
                    ]
                )
    if league.derby:
        # The derby opens the season, so the two teams must get two slots
        # that meet in template round 1. The inverted second half then
        # repeats that game in the last round: the derby closes the season.
        opening = {frozenset(g) for g in template[0]}
        a, b = league.derby
        for k, j in itertools.permutations(slots, 2):
            if frozenset((k, j)) not in opening:
                model.add_bool_or([draw[a, k].Not(), draw[b, j].Not()])
    window = league.max_streak + 1
    for k in slots:
        for r in range(league.num_rounds - window + 1):
            run = [home[k, r + i] for i in range(window)]
            model.add_bool_or(run)  # Not all away.
            model.add_bool_or([x.Not() for x in run])  # Not all home.

    # breaks[k, r] must be true if slot k has the same venue in r-1 and r.
    breaks = []
    for k in slots:
        for r in range(1, league.num_rounds):
            b = model.new_bool_var(f"break_slot{k}_r{r}")
            model.add_bool_or([b, home[k, r].Not(), home[k, r - 1].Not()])  # HH
            model.add_bool_or([b, home[k, r], home[k, r - 1]])  # AA
            breaks.append(b)
    total = sum(breaks)
    # A redundant constraint: theory says no phased schedule has fewer
    # breaks. It changes no solution, but once CP-SAT finds a schedule
    # with this many breaks, it knows at once that it can stop.
    bound = lower_bound(n, "phased")
    model.add(total >= bound)
    model.minimize(total)

    solver = _solver(time_limit)
    status = solver.solve(model)
    if status == cp_model.INFEASIBLE:
        raise NoScheduleError(f"No schedule meets the rules of the {league.name}.")
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"CP-SAT found no schedule (status {status}).")

    slot_team = {k: t for (t, k), v in draw.items() if solver.boolean_value(v)}
    season = []
    for r in rounds:
        games = template[r] if r < half else template[2 * half - 1 - r]
        played = []
        for a, b in games:
            h, w = (a, b) if solver.boolean_value(home[a, r]) else (b, a)
            played.append((slot_team[h], slot_team[w]))
        season.append(played)

    index = {t: i for i, t in enumerate(teams)}
    indexed = [[(index[h], index[a]) for h, a in games] for games in season]
    counted = sum(
        sum(1 for x, y in itertools.pairwise(p) if x == y)
        for p in home_away(indexed, n)
    )
    optimal = status == cp_model.OPTIMAL
    return LeagueSchedule(
        rounds=season,
        draw={t: k + 1 for k, t in slot_team.items()},
        breaks=counted,
        proven_optimal=optimal,
        globally_optimal=optimal and counted == bound,
        best_bound=solver.best_objective_bound,
        wall_time=solver.wall_time,
    )


# ----------------------------------------------------- monolithic model ---


def solve_monolithic(
    n: int, scheme: str = "phased", add_bound: bool = False, time_limit: float = 10.0
) -> MonolithicResult:
    """Minimize breaks with the direct model (no league rules).

    scheme: "single" (one round robin), "phased" (two round robins in a
    row), or "mirrored" (phased, second half mirrors the first).
    """
    teams = range(n)
    half = n - 1
    num_rounds = half if scheme == "single" else 2 * half
    rounds = range(num_rounds)
    model = cp_model.CpModel()

    # play[h, a, r] is true if h hosts a in round r.
    play = {
        (h, a, r): model.new_bool_var(f"{h}v{a}_r{r}")
        for h in teams
        for a in teams
        if h != a
        for r in rounds
    }
    for t in teams:
        for r in rounds:
            model.add_exactly_one(
                [play[t, a, r] for a in teams if a != t]
                + [play[h, t, r] for h in teams if h != t]
            )
    for h, a in itertools.combinations(teams, 2):
        first = [play[h, a, r] for r in range(half)] + [
            play[a, h, r] for r in range(half)
        ]
        model.add_exactly_one(first)  # They meet once in the first half.
        if scheme != "single":
            # Each team hosts the other exactly once.
            model.add_exactly_one(play[h, a, r] for r in rounds)
            model.add_exactly_one(play[a, h, r] for r in rounds)
    if scheme == "mirrored":
        for (h, a, r), x in play.items():
            if r < half:
                model.add(x == play[a, h, r + half])

    home = {
        (t, r): sum(play[t, a, r] for a in teams if a != t)
        for t in teams
        for r in rounds
    }
    breaks = []
    for t in teams:
        for r in range(1, num_rounds):
            b = model.new_bool_var(f"break_{t}_r{r}")
            model.add(b >= home[t, r] + home[t, r - 1] - 1)  # HH
            model.add(b >= 1 - home[t, r] - home[t, r - 1])  # AA
            breaks.append(b)
    total = sum(breaks)
    if add_bound:
        model.add(total >= lower_bound(n, scheme))
    model.minimize(total)

    solver = _solver(time_limit)
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return MonolithicResult(
            "UNKNOWN", None, solver.best_objective_bound, solver.wall_time, None
        )
    schedule = [
        [
            (h, a)
            for (h, a, rr), x in play.items()
            if rr == r and solver.boolean_value(x)
        ]
        for r in rounds
    ]
    counted = sum(
        sum(1 for x, y in itertools.pairwise(p) if x == y)
        for p in home_away(schedule, n)
    )
    return MonolithicResult(
        solver.status_name(status),
        counted,
        solver.best_objective_bound,
        solver.wall_time,
        schedule,
    )
