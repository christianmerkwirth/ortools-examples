"""Instance data for the job shop example.

A job shop makes parts to order. Each job (an order) visits some machines
in a fixed order, its route. A machine works on one job at a time, and an
operation cannot stop once it starts.

This file holds three kinds of instances:

* `machine_shop()`: a small story instance with named jobs and machines,
  plus due dates for the tardiness variant.
* `ft06()` and `la01()`: two classic benchmarks from the OR-Library with
  proven optimal makespans (55 and 666). We use them to test the model.
* `tiny()`: a 3 x 3 instance, small enough to check by brute force.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Operation:
    """One step of a job: a machine and a processing time."""

    machine: str
    duration: int  # Minutes (or abstract time units for benchmarks).


@dataclass(frozen=True)
class Job:
    """An order that visits machines in a fixed sequence."""

    name: str
    operations: tuple[Operation, ...]
    due: int | None = None  # Due time, for the tardiness objective.
    weight: int = 1  # Cost per minute of lateness.


@dataclass(frozen=True)
class JobShopData:
    """A full problem instance."""

    name: str
    machines: tuple[str, ...]
    jobs: tuple[Job, ...]
    known_optimum: int | None = None  # Best makespan, if proven in literature.

    def job(self, name: str) -> Job:
        """Return the job with this name."""
        return next(j for j in self.jobs if j.name == name)


def _route(*steps: tuple[str, int]) -> tuple[Operation, ...]:
    return tuple(Operation(m, d) for m, d in steps)


def machine_shop() -> JobShopData:
    """Return the story instance: eight orders in a precision parts shop.

    Times are in minutes. Rush orders have a higher weight: each minute
    late costs more.
    """
    jobs = (
        Job(
            "pump housing",
            _route(("saw", 12), ("mill", 45), ("drill", 20), ("grinder", 15)),
            due=150,
            weight=3,
        ),
        Job(
            "gear shaft",
            _route(("saw", 8), ("lathe", 40), ("mill", 25), ("grinder", 30)),
            due=180,
        ),
        Job(
            "bracket",
            _route(("saw", 10), ("drill", 15), ("mill", 20)),
            due=90,
            weight=2,
        ),
        Job(
            "flange",
            _route(("lathe", 30), ("drill", 25), ("grinder", 10)),
            due=120,
        ),
        Job(
            "valve body",
            _route(("saw", 15), ("mill", 50), ("lathe", 20), ("drill", 30)),
            due=220,
            weight=3,
        ),
        Job("spindle", _route(("lathe", 55), ("grinder", 35)), due=160),
        Job(
            "coupling",
            _route(("saw", 6), ("lathe", 25), ("drill", 10), ("mill", 15)),
            due=140,
            weight=2,
        ),
        Job(
            "bearing cap", _route(("mill", 30), ("drill", 12), ("grinder", 18)), due=200
        ),
    )
    machines = ("saw", "lathe", "mill", "drill", "grinder")
    return JobShopData("machine shop", machines, jobs)


def parse_orlib(name: str, text: str, known_optimum: int | None = None) -> JobShopData:
    """Parse an instance in OR-Library format.

    Each line is one job: pairs of (machine index, duration), in route
    order. Machines are numbered from 0.
    """
    rows = [list(map(int, line.split())) for line in text.strip().splitlines()]
    n_machines = len(rows[0]) // 2
    machines = tuple(f"M{m}" for m in range(n_machines))
    jobs = tuple(
        Job(
            f"J{j}",
            tuple(Operation(f"M{row[k]}", row[k + 1]) for k in range(0, len(row), 2)),
        )
        for j, row in enumerate(rows)
    )
    return JobShopData(name, machines, jobs, known_optimum)


# Fisher and Thompson (1963), 6 jobs x 6 machines. Optimal makespan: 55.
FT06 = """
2 1 0 3 1 6 3 7 5 3 4 6
1 8 2 5 4 10 5 10 0 10 3 4
2 5 3 4 5 8 0 9 1 1 4 7
1 5 0 5 2 5 3 3 4 8 5 9
2 9 1 3 4 5 5 4 0 3 3 1
1 3 3 3 5 9 0 10 4 4 2 1
"""

# Lawrence (1984), 10 jobs x 5 machines. Optimal makespan: 666.
LA01 = """
1 21 0 53 4 95 3 55 2 34
0 21 3 52 4 16 2 26 1 71
3 39 4 98 1 42 2 31 0 12
1 77 0 55 4 79 2 66 3 77
0 83 3 34 2 64 1 19 4 37
1 54 2 43 4 79 0 92 3 62
3 69 4 77 1 87 2 87 0 93
2 38 0 60 1 41 3 24 4 83
3 17 1 49 4 25 0 44 2 98
4 77 3 79 2 43 1 75 0 96
"""


def ft06() -> JobShopData:
    """Return the ft06 benchmark (optimal makespan 55)."""
    return parse_orlib("ft06", FT06, known_optimum=55)


def la01() -> JobShopData:
    """Return the la01 benchmark (optimal makespan 666)."""
    return parse_orlib("la01", LA01, known_optimum=666)


def tiny() -> JobShopData:
    """Return a 3 x 3 instance for brute-force tests."""
    return parse_orlib("tiny", "0 3 1 2 2 2\n0 2 2 1 1 4\n1 4 2 3 0 1")


def random_instance(n_jobs: int, n_machines: int, seed: int = 0) -> JobShopData:
    """Return a random instance in the style of Taillard (1993).

    Each job visits every machine once, in random order, with durations
    from 1 to 99. Larger ones are hard: the solver finds good schedules
    fast but needs much longer to prove them optimal.
    """
    import random

    rng = random.Random(seed)
    machines = tuple(f"M{m}" for m in range(n_machines))
    jobs = []
    for j in range(n_jobs):
        order = rng.sample(machines, n_machines)
        jobs.append(
            Job(f"J{j}", tuple(Operation(m, rng.randint(1, 99)) for m in order))
        )
    return JobShopData(f"random {n_jobs}x{n_machines}", machines, tuple(jobs))
