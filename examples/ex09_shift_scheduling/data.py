"""Instance data for the shift scheduling example.

A hospital ward needs a nurse roster for the next two weeks. Every day has
three shifts: day, evening, and night. The roster must cover each shift
with enough nurses and at least one senior nurse. It must also respect
labor rules, contracts, and approved leave. Within those hard rules, it
should honor the nurses' wishes and share nights and weekends fairly.

Day 0 is a Monday. Names and wishes are made up.
"""

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Shift:
    """One shift type."""

    code: str  # One letter, used in printed rosters.
    name: str
    start: int  # Start hour.
    hours: int


@dataclass(frozen=True)
class Nurse:
    """One nurse and their contract."""

    name: str
    senior: bool
    min_shifts: int  # Contract: shifts in the planning period.
    max_shifts: int
    full_time: bool = True  # Fairness compares full-time nurses only.
    leave: frozenset[int] = frozenset()  # Days of approved leave.


@dataclass(frozen=True)
class Request:
    """A wish not to work. A broken wish costs `weight` penalty points.

    `shift=None` means the whole day off; otherwise only that shift.
    """

    nurse: str
    day: int
    shift: str | None
    weight: int


@dataclass(frozen=True)
class Weights:
    """Penalty points for each kind of soft-rule violation."""

    isolated_day_off: int = 2  # Per single day off between two work days.
    night_spread: int = 4  # Per night of difference (most minus fewest).
    weekend_spread: int = 4  # Per weekend shift of difference.


@dataclass(frozen=True)
class RosterData:
    """A full problem instance."""

    name: str
    num_days: int
    shifts: tuple[Shift, ...]
    nurses: tuple[Nurse, ...]
    demand: tuple[dict[str, int], ...]  # Per day: shift code -> nurses needed.
    # (a, b): shift a on one day must not be followed by shift b next day.
    forbidden_sequences: tuple[tuple[str, str], ...]
    max_consecutive_days: int
    requests: tuple[Request, ...] = ()
    weekend_days: frozenset[int] = frozenset()
    night_shift: str = "N"
    weights: Weights = field(default_factory=Weights)

    def nurse(self, name: str) -> Nurse:
        """Return the nurse with this name."""
        return next(n for n in self.nurses if n.name == name)

    def with_nurse(self, name: str, **changes) -> "RosterData":
        """Return a copy with changed fields for one nurse."""
        nurses = tuple(
            replace(n, **changes) if n.name == name else n for n in self.nurses
        )
        return replace(self, nurses=nurses)

    def without_nurse(self, name: str) -> "RosterData":
        """Return a copy without one nurse (and without their requests)."""
        return replace(
            self,
            nurses=tuple(n for n in self.nurses if n.name != name),
            requests=tuple(r for r in self.requests if r.nurse != name),
        )


SHIFTS = (
    Shift("D", "day", 7, 8),
    Shift("E", "evening", 15, 8),
    Shift("N", "night", 23, 8),
)

# Labor law asks for 11 hours of rest between shifts. A night shift ends at
# 07:00, so the nurse cannot work the day (07:00) or evening (15:00) shift
# on the next day. An evening shift ends at 23:00, so no day shift follows.
REST_RULES = (("N", "D"), ("N", "E"), ("E", "D"))

DAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def day_label(day: int) -> str:
    """Return a short label such as 'Mon 1' for day 0."""
    return f"{DAY_NAMES[day % 7]} {day + 1}"


def ward_two_weeks() -> RosterData:
    """Return the default instance: 12 nurses, 14 days, 3 shifts."""
    num_days = 14
    weekend = frozenset(d for d in range(num_days) if d % 7 in (5, 6))
    weekday_demand = {"D": 3, "E": 2, "N": 2}
    weekend_demand = {"D": 2, "E": 2, "N": 2}
    demand = tuple(
        dict(weekend_demand if d in weekend else weekday_demand)
        for d in range(num_days)
    )
    nurses = (
        Nurse("Alex", senior=True, min_shifts=8, max_shifts=10),
        Nurse("Bea", senior=True, min_shifts=8, max_shifts=10),
        Nurse("Chen", senior=True, min_shifts=8, max_shifts=10),
        Nurse(
            "Dana", senior=True, min_shifts=6, max_shifts=7, leave=frozenset({7, 8, 9})
        ),
        Nurse("Eli", senior=True, min_shifts=8, max_shifts=10),
        Nurse("Femi", senior=False, min_shifts=8, max_shifts=10),
        Nurse("Gus", senior=False, min_shifts=8, max_shifts=10),
        Nurse("Hana", senior=False, min_shifts=8, max_shifts=10),
        Nurse(
            "Ivo", senior=False, min_shifts=6, max_shifts=7, leave=frozenset({0, 1, 2})
        ),
        Nurse("Jo", senior=False, min_shifts=4, max_shifts=6, full_time=False),
        Nurse("Kai", senior=False, min_shifts=4, max_shifts=6, full_time=False),
        Nurse("Lou", senior=False, min_shifts=4, max_shifts=6, full_time=False),
    )
    requests = (
        # Four of the five seniors want the first Saturday off. Each shift
        # needs a senior, so at most one of these wishes can come true.
        Request("Alex", 5, None, 3),  # Family wedding.
        Request("Alex", 6, None, 3),
        Request("Bea", 5, None, 3),
        Request("Chen", 5, None, 2),
        Request("Eli", 5, None, 2),
        Request("Bea", 2, "N", 1),
        Request("Bea", 3, "N", 1),
        Request("Chen", 11, None, 3),  # Exam.
        Request("Eli", 0, None, 2),
        Request("Femi", 12, None, 3),
        Request("Femi", 13, None, 3),
        Request("Gus", 4, "E", 1),  # Evening class on Fridays.
        Request("Gus", 11, "E", 1),
        Request("Hana", 9, None, 2),
        Request("Jo", 5, None, 2),
        Request("Jo", 6, None, 2),
        Request("Kai", 1, "N", 1),
        Request("Lou", 10, None, 2),
        # Gus and Hana would rather not work nights at all. Each night they
        # work costs one point. This wish pulls against night fairness.
        *(Request("Gus", d, "N", 1) for d in range(num_days)),
        *(Request("Hana", d, "N", 1) for d in range(num_days)),
    )
    return RosterData(
        name="ward, two weeks",
        num_days=num_days,
        shifts=SHIFTS,
        nurses=nurses,
        demand=demand,
        forbidden_sequences=REST_RULES,
        max_consecutive_days=5,
        requests=requests,
        weekend_days=weekend,
    )


def tiny_ward() -> RosterData:
    """Return a tiny instance that brute force can solve: 4 nurses, 3 days.

    It has only a day and a night shift, so each nurse has 3 choices per
    day. That gives 3^12 (about half a million) rosters in total.
    """
    shifts = (Shift("D", "day", 7, 12), Shift("N", "night", 19, 12))
    nurses = (
        Nurse("Alex", senior=True, min_shifts=1, max_shifts=2),
        Nurse("Bea", senior=True, min_shifts=1, max_shifts=2),
        Nurse("Chen", senior=True, min_shifts=1, max_shifts=2, leave=frozenset({2})),
        Nurse("Dana", senior=False, min_shifts=1, max_shifts=2, full_time=False),
    )
    requests = (
        Request("Alex", 1, None, 3),
        Request("Bea", 0, "N", 2),
        Request("Bea", 2, None, 1),
        Request("Dana", 0, None, 1),
    )
    return RosterData(
        name="tiny ward",
        num_days=3,
        shifts=shifts,
        nurses=nurses,
        demand=({"D": 1, "N": 1}, {"D": 2, "N": 1}, {"D": 1, "N": 1}),
        forbidden_sequences=(("N", "D"),),
        max_consecutive_days=2,
        requests=requests,
        weekend_days=frozenset({1, 2}),
    )
