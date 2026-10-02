"""Instance data for the assignment example: a home repair service.

A small repair company sends technicians to customers' homes. Each job
needs one trade (plumbing, electrical, gas, heating, or carpentry). Each
technician is fast in some trades, slower in others, and not qualified in
some. Gas work needs a gas certificate by law.

The time a technician needs for a job is:

    travel time from their home base to the job's district
    + the job's base duration x the technician's speed factor for the trade

* Part A: eight technicians, eight morning jobs, one job each.
* Part B: the full day. Sixteen jobs, shift limits (six hours, or four
  for part-timers), and two heavy jobs that need a team of two.
"""

import random
from dataclasses import dataclass

# Travel minutes between the four districts of the town.
DISTRICTS = ("north", "east", "south", "west")
TRAVEL = {
    ("north", "north"): 10,
    ("north", "east"): 25,
    ("north", "south"): 40,
    ("north", "west"): 25,
    ("east", "east"): 10,
    ("east", "south"): 25,
    ("east", "west"): 40,
    ("south", "south"): 10,
    ("south", "west"): 25,
    ("west", "west"): 10,
}


def travel_minutes(a: str, b: str) -> int:
    """Return the travel time between two districts."""
    return TRAVEL.get((a, b), TRAVEL.get((b, a), 0))


@dataclass(frozen=True)
class Technician:
    """A worker. A trade that is missing from `speed` is a trade they cannot do."""

    name: str
    base: str  # Home district.
    speed: dict[str, float]  # Trade -> factor on the base duration (1.0 = expert).
    shift: int = 360  # Minutes available per day (Part B only).


@dataclass(frozen=True)
class Job:
    """A customer visit."""

    name: str
    trade: str
    district: str
    minutes: int  # Base duration for an expert.
    team_size: int = 1  # Heavy jobs need two people (Part B only).


@dataclass(frozen=True)
class AssignmentData:
    """A full problem instance."""

    technicians: tuple[Technician, ...]
    jobs: tuple[Job, ...]

    def cost(self, tech: Technician, job: Job) -> int | None:
        """Return minutes for this pair, or None if the technician is not qualified."""
        factor = tech.speed.get(job.trade)
        if factor is None:
            return None
        return travel_minutes(tech.base, job.district) + round(job.minutes * factor)

    def cost_matrix(self) -> list[list[int | None]]:
        """Return costs as rows (technicians) by columns (jobs)."""
        return [[self.cost(t, j) for j in self.jobs] for t in self.technicians]


TECHNICIANS = (
    Technician("Alex", "north", {"plumbing": 1.0, "heating": 1.2, "gas": 1.1}),
    Technician("Bo", "north", {"electrical": 1.0, "carpentry": 1.4}),
    Technician("Chris", "east", {"plumbing": 1.3, "carpentry": 1.0, "electrical": 1.5}),
    Technician("Dana", "east", {"gas": 1.0, "heating": 1.0, "plumbing": 1.4}),
    Technician("Eli", "south", {"electrical": 1.1, "heating": 1.5}, shift=240),
    Technician("Fran", "south", {"carpentry": 1.1, "plumbing": 1.2}),
    Technician("Gus", "west", {"heating": 1.1, "gas": 1.3, "electrical": 1.6}),
    Technician("Hana", "west", {"plumbing": 1.1, "carpentry": 1.2}, shift=240),
)

JOBS = (
    # The eight morning jobs (Part A).
    Job("burst pipe", "plumbing", "north", 90),
    Job("gas boiler service", "gas", "east", 120),
    Job("fuse box upgrade", "electrical", "south", 150),
    Job("radiator fix", "heating", "west", 60),
    Job("kitchen cabinet", "carpentry", "east", 180),
    Job("blocked drain", "plumbing", "south", 60),
    Job("new sockets", "electrical", "north", 90),
    Job("gas leak check", "gas", "west", 45),
    # The afternoon jobs (Part B adds these).
    Job("door repair", "carpentry", "north", 60),
    Job("water heater swap", "heating", "south", 150, team_size=2),
    Job("shower install", "plumbing", "west", 120),
    Job("light fittings", "electrical", "east", 60),
    Job("floor boards", "carpentry", "south", 120, team_size=2),
    Job("thermostat", "heating", "north", 45),
    Job("gas hob fitting", "gas", "south", 60),
    Job("leaky tap", "plumbing", "east", 30),
)


def morning() -> AssignmentData:
    """Part A: eight technicians and eight jobs, one job each."""
    return AssignmentData(TECHNICIANS, JOBS[:8])


def full_day() -> AssignmentData:
    """Part B: all sixteen jobs, shift limits, and two team jobs."""
    return AssignmentData(TECHNICIANS, JOBS)


def random_square(n: int, seed: int = 0, gap: float = 0.2) -> AssignmentData:
    """Return a random n x n instance for tests and timing experiments.

    Each technician lacks a trade with probability `gap`, but always has
    at least one trade.
    """
    rng = random.Random(seed)
    trades = ("plumbing", "electrical", "gas", "heating", "carpentry")
    techs = []
    for k in range(n):
        speed = {
            t: round(rng.uniform(1.0, 1.6), 1) for t in trades if rng.random() > gap
        }
        speed = speed or {rng.choice(trades): 1.0}
        techs.append(Technician(f"tech {k}", rng.choice(DISTRICTS), speed))
    jobs = tuple(
        Job(
            f"job {k}",
            rng.choice(trades),
            rng.choice(DISTRICTS),
            rng.randrange(30, 181, 15),
        )
        for k in range(n)
    )
    return AssignmentData(tuple(techs), jobs)
