"""Independent checks for league schedules. Uses no OR-Tools code.

* `schedule_errors` checks every rule of a season, round by round.
* `patterns` and `team_breaks` recount home/away patterns and breaks.
* `brute_force_min_breaks` tries every schedule of a tiny league, to test
  the break bounds of circle.py and the CP-SAT models.
"""

import itertools

from .data import League


def patterns(league: League, rounds: list[list[tuple[str, str]]]) -> dict[str, str]:
    """Return each team's home/away pattern, for example "HAHHA..."."""
    out = {t.name: ["?"] * len(rounds) for t in league.teams}
    for r, games in enumerate(rounds):
        for home, away in games:
            out[home][r] = "H"
            out[away][r] = "A"
    return {t: "".join(p) for t, p in out.items()}


def team_breaks(league: League, rounds) -> dict[str, list[int]]:
    """Return, per team, the rounds (1-based) in which a break happens.

    A break in round r means round r has the same venue as round r - 1.
    """
    return {
        team: [r + 1 for r in range(1, len(p)) if p[r] == p[r - 1]]
        for team, p in patterns(league, rounds).items()
    }


def schedule_errors(league: League, rounds: list[list[tuple[str, str]]]) -> list[str]:
    """Return a list of broken rules. An empty list means the season is valid."""
    errors = []
    names = [t.name for t in league.teams]
    n, half = len(names), len(names) - 1
    if len(rounds) != league.num_rounds:
        return [f"{len(rounds)} rounds instead of {league.num_rounds}"]

    for r, games in enumerate(rounds, start=1):
        playing = [t for g in games for t in g]
        for t in names:
            if playing.count(t) != 1:
                errors.append(f"round {r}: {t} plays {playing.count(t)} times")
        for h, a in games:
            if h == a:
                errors.append(f"round {r}: {h} plays itself")

    hosted = [g for games in rounds for g in games]
    for h, a in itertools.permutations(names, 2):
        if hosted.count((h, a)) != 1:
            errors.append(f"{h} hosts {a} {hosted.count((h, a))} times")
    for start in (0, half):
        met = [frozenset(g) for games in rounds[start : start + half] for g in games]
        for a, b in itertools.combinations(names, 2):
            if met.count(frozenset((a, b))) != 1:
                times = met.count(frozenset((a, b)))
                which = "first" if start == 0 else "second"
                errors.append(f"{a} and {b} meet {times}x in the {which} half")

    pattern = patterns(league, rounds)
    for team, rnd in league.home_banned:
        if pattern[team][rnd - 1] == "H":
            errors.append(
                f"{team} plays at home in round {rnd}, but its ground is booked"
            )
    for a, b in league.shared_stadiums():
        for r in range(len(rounds)):
            if pattern[a][r] == pattern[b][r] == "H":
                errors.append(f"round {r + 1}: {a} and {b} both at home")
    if league.home_opener and pattern[league.home_opener][0] != "H":
        errors.append(f"{league.home_opener} does not open the season at home")
    if league.derby:
        pair = frozenset(league.derby)
        for r in (0, len(rounds) - 1):
            if pair not in {frozenset(g) for g in rounds[r]}:
                errors.append(f"the derby is not in round {r + 1}")
    run = league.max_streak + 1
    for team, p in pattern.items():
        for r in range(len(p) - run + 1):
            if len(set(p[r : r + run])) == 1:
                errors.append(f"{team}: {run} games in a row with venue {p[r]}")
    if n % 2:
        errors.append("the number of teams must be even")
    return errors


def total_breaks(league: League, rounds) -> int:
    """Return the number of breaks of all teams together."""
    return sum(len(b) for b in team_breaks(league, rounds).values())


# --------------------------------------------------------- brute force ---


def _oriented_matchings(n: int) -> list[list[tuple[int, int]]]:
    """All ways to pair up n teams, with every choice of home team."""

    def pairings(rest):
        if not rest:
            yield []
            return
        first = rest[0]
        for k in range(1, len(rest)):
            for tail in pairings(rest[1:k] + rest[k + 1 :]):
                yield [(first, rest[k]), *tail]

    out = []
    for pairing in pairings(list(range(n))):
        for flips in itertools.product((False, True), repeat=len(pairing)):
            out.append(
                [
                    (b, a) if f else (a, b)
                    for (a, b), f in zip(pairing, flips, strict=True)
                ]
            )
    return out


def brute_force_min_breaks(n: int, scheme: str) -> int:
    """Return the fewest breaks over EVERY schedule of the scheme. Tiny n only.

    scheme: "single", "phased", or "mirrored" (as in circle.py). The search
    builds the season round by round and skips any game already played.
    """
    half = n - 1
    options = _oriented_matchings(n)
    best = [10**9]

    def breaks_of(season):
        venue = [[None] * len(season) for _ in range(n)]
        for r, games in enumerate(season):
            for h, a in games:
                venue[h][r], venue[a][r] = 1, 0
        return sum(1 for v in venue for x, y in itertools.pairwise(v) if x == y)

    def extend(season, hosted, met):
        r = len(season)
        if scheme == "single" and r == half:
            best[0] = min(best[0], breaks_of(season))
            return
        if scheme == "mirrored" and r == half:
            full = season + [[(a, h) for h, a in g] for g in season]
            best[0] = min(best[0], breaks_of(full))
            return
        if r == 2 * half:
            best[0] = min(best[0], breaks_of(season))
            return
        if r == half:
            met = set()  # A new half: every pair meets once more.
        for games in options:
            pairs = [frozenset(g) for g in games]
            if any(g in hosted for g in games) or any(p in met for p in pairs):
                continue
            extend(season + [games], hosted | set(games), met | set(pairs))

    extend([], set(), set())
    return best[0]
