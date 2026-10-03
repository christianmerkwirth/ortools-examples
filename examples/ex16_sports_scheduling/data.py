"""Instance data for the sports scheduling example: a regional football league.

Ten clubs play a double round robin: every club meets every other club
twice, once at each home ground. The season has 18 rounds. Each half of
the season (9 rounds) is a full round robin of its own.

The league also has local rules:

* Riverside Rovers and Riverside Athletic share Riverside Park. They can
  never both play at home in the same round.
* Millbrook FC relays its pitch. It cannot play at home in rounds 1 and 2.
* Harbor Town's ground hosts a concert in round 9.
* The champions, Northgate United, open the season at home.
* The Riverside derby opens and closes the season.
* No club plays three home games or three away games in a row.

Round numbers in this file are 1-based, as a fan would count them.
Club names are fictional.
"""

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Team:
    """A club and its home ground."""

    name: str
    short: str  # Three-letter code for tables and plots.
    stadium: str


@dataclass(frozen=True)
class League:
    """A full problem instance."""

    teams: tuple[Team, ...]
    # (team name, round) pairs where the team's ground is not available.
    home_banned: tuple[tuple[str, int], ...] = ()
    derby: tuple[str, str] | None = None  # Opens and closes the season.
    home_opener: str | None = None  # Plays at home in round 1.
    max_streak: int = 2  # Most home (or away) games in a row.
    name: str = field(default="league")

    @property
    def num_rounds(self) -> int:
        """Two full round robins: 2 (n - 1) rounds."""
        return 2 * (len(self.teams) - 1)

    def team(self, name: str) -> Team:
        """Return the team with this name."""
        return next(t for t in self.teams if t.name == name)

    def shared_stadiums(self) -> list[tuple[str, str]]:
        """Return all pairs of teams that share a ground."""
        return [
            (a.name, b.name)
            for k, a in enumerate(self.teams)
            for b in self.teams[k + 1 :]
            if a.stadium == b.stadium
        ]


TEAMS = (
    Team("Riverside Rovers", "RRO", "Riverside Park"),
    Team("Riverside Athletic", "RAT", "Riverside Park"),
    Team("Northgate United", "NOR", "Northgate Arena"),
    Team("Eastbrook Town", "EAS", "Brook Lane"),
    Team("Westfield Wanderers", "WES", "Westfield Road"),
    Team("Millbrook FC", "MIL", "Mill Meadow"),
    Team("Oakdale City", "OAK", "Oakdale Stadium"),
    Team("Harbor Town", "HAR", "Quayside Ground"),
    Team("Lakeside FC", "LAK", "Lakeside Park"),
    Team("Hillcrest Albion", "HIL", "Hillcrest Field"),
)


def regional_league() -> League:
    """Return the default instance: ten clubs with local rules."""
    return League(
        teams=TEAMS,
        home_banned=(
            ("Millbrook FC", 1),
            ("Millbrook FC", 2),
            ("Harbor Town", 9),
        ),
        derby=("Riverside Rovers", "Riverside Athletic"),
        home_opener="Northgate United",
        name="regional league",
    )


def plain_league(n: int) -> League:
    """Return a league of n teams with no local rules (for tests and timing)."""
    teams = tuple(
        Team(f"Team {k + 1}", f"T{k + 1:02d}", f"Ground {k + 1}") for k in range(n)
    )
    return League(teams=teams, max_streak=n, name=f"{n} teams")


def with_rules(league: League, **changes) -> League:
    """Return a copy of a league with some fields changed."""
    return replace(league, **changes)
