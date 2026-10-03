"""The circle method and the theory of breaks. Pure Python, no OR-Tools.

A *schedule* here is a list of rounds. Each round is a list of games, and
each game is a (home, away) pair of team indices 0 .. n-1.

A *break* is two home games, or two away games, in a row for one team.
Fans and clubs dislike breaks, so leagues try to have as few as possible.
Three classic facts give hard lower bounds (proofs in the README):

* A single round robin (SRR) has at least n - 2 breaks.
* A *phased* double round robin, where each half is an SRR, has at least
  2n - 4 breaks: each half needs n - 2.
* A *mirrored* double round robin, where the second half repeats the first
  with home and away swapped, has at least 3n - 6 breaks.

The circle method builds schedules that meet these bounds.
"""

Game = tuple[int, int]
Schedule = list[list[Game]]


def circle_method(n: int) -> Schedule:
    """Return a single round robin for an even number n of teams.

    Fix team n - 1 in the middle. Put the other teams on a circle. In round
    r, team r plays the middle team, and the teams r + k and r - k (mod
    n - 1) play each other. Then turn the circle by one step.

    The home/away rule below (alternate for the middle team, and by the
    parity of k for the others) gives exactly n - 2 breaks, the minimum.
    """
    if n < 2 or n % 2:
        raise ValueError("the circle method needs an even number of teams")
    rounds = []
    for r in range(n - 1):
        games = [(r, n - 1) if r % 2 == 0 else (n - 1, r)]
        for k in range(1, n // 2):
            a, b = (r + k) % (n - 1), (r - k) % (n - 1)
            games.append((a, b) if k % 2 == 1 else (b, a))
        rounds.append(games)
    return rounds


def mirrored(first_half: Schedule) -> Schedule:
    """Repeat the rounds in the same order, with home and away swapped."""
    return first_half + [[(a, h) for h, a in games] for games in first_half]


def inverted(first_half: Schedule) -> Schedule:
    """Repeat the rounds in reverse order, with home and away swapped.

    The last round of the first half and the first round of the second half
    then hold the same games with venues swapped. So no team gets a break
    at the turn of the season.
    """
    return first_half + [[(a, h) for h, a in games] for games in reversed(first_half)]


def home_away(schedule: Schedule, n: int) -> list[str]:
    """Return each team's pattern as a string such as "HAHAAH..."."""
    pattern = [["?"] * len(schedule) for _ in range(n)]
    for r, games in enumerate(schedule):
        for h, a in games:
            pattern[h][r] = "H"
            pattern[a][r] = "A"
    return ["".join(p) for p in pattern]


def count_breaks(pattern: str) -> int:
    """Return the number of breaks in one home/away pattern."""
    return sum(1 for x, y in zip(pattern, pattern[1:], strict=False) if x == y)


def lower_bound(n: int, scheme: str) -> int:
    """Return the proven minimum number of breaks for a scheme.

    "single": one round robin. "phased": two round robins in a row.
    "mirrored": phased, and the second half mirrors the first.
    """
    return {"single": n - 2, "phased": 2 * n - 4, "mirrored": 3 * n - 6}[scheme]
