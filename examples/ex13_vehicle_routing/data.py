"""Instance data for the vehicle routing example: pharmacy deliveries.

A pharmaceutical wholesaler delivers medicine crates from one depot to
pharmacies in and around a city. Each pharmacy orders a number of crates,
needs a few minutes to unload, and can only receive goods in a delivery
window (for example before it opens to customers). The wholesaler has a
small fleet of identical vans. Each van carries a limited number of crates
and works one shift.

Which van visits which pharmacies, in which order, so that every window
holds, no van is overloaded, and the day costs as little as possible?

Coordinates are in km. Road distance is the straight-line distance times a
road factor of 1.3, a common rule of thumb for city streets.
"""

import math
import random
from dataclasses import dataclass, replace

ROAD_FACTOR = 1.3  # Roads are longer than straight lines.
METERS_PER_KM = 1000


@dataclass(frozen=True)
class Stop:
    """The depot or a customer. Times are minutes after midnight."""

    name: str
    x: float  # km
    y: float  # km
    demand: int = 0  # Crates.
    service: int = 0  # Minutes to unload.
    open: int = 0  # Earliest start of service.
    close: int = 24 * 60  # Latest start of service.


@dataclass(frozen=True)
class VrpData:
    """A full problem instance. Stop 0 is the depot."""

    name: str
    stops: tuple[Stop, ...]
    num_vehicles: int
    capacity: int  # Crates per van.
    shift_start: int  # Vans may leave the depot from this time on.
    shift_end: int  # Vans must be back by this time.
    speed_kmh: float = 30.0  # Average speed in city traffic.
    # One van-day (driver, lease, insurance) in the same unit as distance:
    # meters. 40,000 means "a van costs as much as 40 km of driving".
    fixed_cost: int = 40_000
    # Cost of leaving one customer unserved, per crate, in meters. None
    # means every customer must be served.
    drop_penalty_per_crate: int | None = None
    # Cost of one minute of a van's working time (from leaving the depot to
    # coming back), in meters. 0 means driver time is free.
    minute_cost: int = 0

    @property
    def customers(self) -> range:
        """Return the stop indices of all customers (everything but 0)."""
        return range(1, len(self.stops))

    def drop_penalty(self, stop: int) -> int | None:
        """Return the cost of leaving a customer unserved (None: not allowed)."""
        if self.drop_penalty_per_crate is None:
            return None
        return self.drop_penalty_per_crate * self.stops[stop].demand


def distance_matrix(data: VrpData) -> list[list[int]]:
    """Return road distances in whole meters between all stops.

    The routing library adds costs as 64-bit integers, so we use meters.
    """
    return [
        [
            round(METERS_PER_KM * ROAD_FACTOR * math.hypot(a.x - b.x, a.y - b.y))
            for b in data.stops
        ]
        for a in data.stops
    ]


def travel_minutes(data: VrpData) -> list[list[int]]:
    """Return driving times in whole minutes between all stops."""
    meters_per_minute = data.speed_kmh * METERS_PER_KM / 60
    return [
        [round(d / meters_per_minute) for d in row] for row in distance_matrix(data)
    ]


def clock(minutes: int) -> str:
    """Format minutes after midnight as HH:MM."""
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


# ---------------------------------------------------------------- instances


WINDOWS = {
    # Name: (open, close) for the start of service, in minutes.
    "early": (7 * 60 + 30, 10 * 60),  # Before the shop opens.
    "midday": (10 * 60, 14 * 60),
    "late": (13 * 60, 16 * 60),
    "any": (7 * 60 + 30, 16 * 60),
}


def random_customers(n: int, seed: int) -> list[Stop]:
    """Return n customers in four towns plus some in the countryside."""
    rng = random.Random(seed)
    towns = [(-9.0, 6.0), (8.0, 9.0), (10.0, -7.0), (-6.0, -9.0)]
    kinds = ["early"] * 4 + ["midday"] * 2 + ["late"] * 2 + ["any"] * 2
    stops = []
    for k in range(1, n + 1):
        if rng.random() < 0.8:
            cx, cy = rng.choice(towns)
            x, y = rng.gauss(cx, 2.5), rng.gauss(cy, 2.5)
        else:
            x, y = rng.uniform(-15, 15), rng.uniform(-15, 15)
        open_, close = WINDOWS[rng.choice(kinds)]
        stops.append(
            Stop(
                f"pharmacy {k}",
                round(x, 2),
                round(y, 2),
                demand=rng.randint(1, 8),
                service=rng.choice([5, 10, 10, 15]),
                open=open_,
                close=close,
            )
        )
    return stops


def pharmacy_day(n: int = 50, seed: int = 7) -> VrpData:
    """Return the default instance: a normal day, every pharmacy is served."""
    depot = Stop("depot", 0.0, 0.0)
    return VrpData(
        name="normal day",
        stops=(depot, *random_customers(n, seed)),
        num_vehicles=10,
        capacity=40,
        shift_start=7 * 60,
        shift_end=17 * 60,
    )


def flu_season_day(n: int = 50, seed: int = 7) -> VrpData:
    """Return a peak day: demand is up 25%, and two drivers are off sick.

    Eight vans carry at most 320 crates, but the pharmacies order 346. Each
    pharmacy may be dropped (it gets its crates tomorrow) at a penalty
    worth 15 km of driving per crate.
    """
    base = pharmacy_day(n, seed)
    stops = (base.stops[0],) + tuple(
        replace(s, demand=math.ceil(1.25 * s.demand)) for s in base.stops[1:]
    )
    return replace(
        base,
        name="flu season day",
        stops=stops,
        num_vehicles=8,
        drop_penalty_per_crate=15_000,
    )


def paid_time_day(n: int = 50, seed: int = 7) -> VrpData:
    """Return the normal day, but driver time now costs money.

    A driver costs about $30 per hour and a van about $0.50 per km to run,
    so one working minute costs as much as one km of driving.
    """
    return replace(
        pharmacy_day(n, seed), name="normal day, paid time", minute_cost=1000
    )


def small_instance(
    n: int, seed: int, num_vehicles: int = 2, drops: bool = False, minute_cost: int = 0
) -> VrpData:
    """Return a tiny instance for exact cross-checks (brute force, CP-SAT)."""
    rng = random.Random(seed)
    stops = [Stop("depot", 0.0, 0.0)]
    for k in range(1, n + 1):
        open_ = rng.choice([7 * 60 + 30, 9 * 60, 11 * 60])
        stops.append(
            Stop(
                f"c{k}",
                round(rng.uniform(-10, 10), 2),
                round(rng.uniform(-10, 10), 2),
                demand=rng.randint(1, 6),
                service=rng.choice([5, 10, 15]),
                open=open_,
                close=open_ + rng.choice([60, 120, 240]),
            )
        )
    total = sum(s.demand for s in stops)
    return VrpData(
        name=f"small (n={n}, seed={seed})",
        stops=tuple(stops),
        num_vehicles=num_vehicles,
        capacity=max(8, math.ceil(0.6 * total)),
        shift_start=7 * 60,
        shift_end=13 * 60,
        fixed_cost=10_000,
        drop_penalty_per_crate=8_000 if drops else None,
        minute_cost=minute_cost,
    )
