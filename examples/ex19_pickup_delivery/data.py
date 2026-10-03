"""Instance data for the pickup and delivery example: patient transport.

A charity runs a few minibuses that take patients from home to hospital
and clinic appointments, and back home afterwards. Every ride request has
a pickup place, a drop-off place, a number of seats (the patient, maybe
with a companion), and time windows. Nobody should sit in the bus much
longer than a direct trip would take.

Which bus serves which request, and in which order should each bus stop,
so that everyone is on time, no bus is overfull, and the day costs as
little driving as possible?

Coordinates are in km. Road distance is the straight-line distance times
a road factor of 1.3, as in example 13.
"""

import math
import random
from dataclasses import dataclass, replace

ROAD_FACTOR = 1.3
METERS_PER_KM = 1000


@dataclass(frozen=True)
class Place:
    """A named point on the map (km)."""

    name: str
    x: float
    y: float


@dataclass(frozen=True)
class Request:
    """One ride: pick up at `pickup`, drop off at `dropoff`.

    Times are minutes after midnight and refer to the start of service
    (boarding at the pickup, getting off at the drop-off).
    """

    name: str
    pickup: Place
    dropoff: Place
    seats: int  # Patient plus companions.
    pickup_open: int
    pickup_close: int
    dropoff_open: int
    dropoff_close: int


@dataclass(frozen=True)
class Stop:
    """A node of the routing graph: the depot, a pickup, or a drop-off."""

    name: str
    x: float
    y: float
    load: int  # Seats taken (+) at a pickup, freed (-) at a drop-off.
    service: int  # Minutes to board or get off.
    open: int
    close: int


@dataclass(frozen=True)
class PdpData:
    """A full problem instance."""

    name: str
    depot: Place
    requests: tuple[Request, ...]
    num_vehicles: int
    capacity: int  # Passenger seats per bus.
    shift_start: int
    shift_end: int
    speed_kmh: float = 30.0
    # One bus-day (driver, fuel, insurance) as meters of driving.
    fixed_cost: int = 30_000
    # A passenger may ride at most this many minutes longer than the direct
    # trip. None means no limit.
    max_detour: int | None = 20
    # Cost of declining a request (the charity books a taxi instead), as
    # meters of driving. None means every request must be served.
    decline_penalty: int | None = None

    def stops(self) -> list[Stop]:
        """Return all nodes: 0 is the depot; request r has 2r+1 and 2r+2."""
        nodes = [Stop(self.depot.name, self.depot.x, self.depot.y, 0, 0, 0, 24 * 60)]
        for r in self.requests:
            board = boarding_minutes(r.seats)
            nodes.append(
                Stop(
                    f"{r.name} pickup",
                    r.pickup.x,
                    r.pickup.y,
                    r.seats,
                    board,
                    r.pickup_open,
                    r.pickup_close,
                )
            )
            nodes.append(
                Stop(
                    f"{r.name} drop-off",
                    r.dropoff.x,
                    r.dropoff.y,
                    -r.seats,
                    board,
                    r.dropoff_open,
                    r.dropoff_close,
                )
            )
        return nodes


def boarding_minutes(seats: int) -> int:
    """Return the minutes to board (or leave) the bus for a party."""
    return 2 + 2 * seats


def pickup_node(r: int) -> int:
    """Return the node index of request r's pickup."""
    return 2 * r + 1


def dropoff_node(r: int) -> int:
    """Return the node index of request r's drop-off."""
    return 2 * r + 2


def distance_matrix(data: PdpData) -> list[list[int]]:
    """Return road distances in whole meters between all nodes."""
    nodes = data.stops()
    return [
        [
            round(METERS_PER_KM * ROAD_FACTOR * math.hypot(a.x - b.x, a.y - b.y))
            for b in nodes
        ]
        for a in nodes
    ]


def travel_minutes(data: PdpData) -> list[list[int]]:
    """Return driving times in whole minutes between all nodes."""
    per_minute = data.speed_kmh * METERS_PER_KM / 60
    return [[round(d / per_minute) for d in row] for row in distance_matrix(data)]


def max_ride(data: PdpData, r: int) -> int | None:
    """Return the longest allowed time from boarding to drop-off (minutes).

    The clock starts when boarding starts at the pickup and stops when the
    bus arrives at the drop-off. The direct trip takes the boarding time
    plus the direct drive; the limit allows `max_detour` minutes more.
    """
    if data.max_detour is None:
        return None
    req = data.requests[r]
    direct = travel_minutes(data)[pickup_node(r)][dropoff_node(r)]
    return boarding_minutes(req.seats) + direct + data.max_detour


def clock(minutes: int) -> str:
    """Format minutes after midnight as HH:MM."""
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


# ---------------------------------------------------------------- instances

GARAGE = Place("garage", 0.0, 0.0)
HOSPITAL = Place("hospital", 1.0, 1.5)
DIALYSIS = Place("dialysis center", -5.0, 4.0)
PHYSIO = Place("physio clinic", 6.0, -3.0)
CLINICS = (HOSPITAL, HOSPITAL, HOSPITAL, DIALYSIS, PHYSIO)
VILLAGES = ((-9.0, 7.0), (8.0, 8.0), (9.0, -8.0), (-7.0, -8.0), (0.0, 10.0))


def random_requests(n: int, seed: int) -> list[Request]:
    """Return n ride requests: trips to appointments and trips back home.

    * To an appointment at time A: the patient is ready 70 to 25 minutes
      before A, and must arrive 30 to 5 minutes before A.
    * Home after an appointment that ends at E: pickup between E and E+30,
      drop-off any time in the next two hours (the ride limit decides).
    """
    rng = random.Random(seed)
    requests = []
    for k in range(1, n + 1):
        cx, cy = rng.choice(VILLAGES)
        home = Place(
            f"home {k}", round(rng.gauss(cx, 2.0), 2), round(rng.gauss(cy, 2.0), 2)
        )
        clinic = rng.choice(CLINICS)
        seats = rng.choice([1, 1, 1, 2, 2, 3])
        if rng.random() < 0.6:
            a = rng.randrange(8 * 60 + 30, 13 * 60 + 1, 15)  # Appointment.
            requests.append(
                Request(f"R{k:02d}", home, clinic, seats, a - 70, a - 25, a - 30, a - 5)
            )
        else:
            e = rng.randrange(9 * 60, 14 * 60 + 1, 15)  # Appointment ends.
            requests.append(
                Request(f"R{k:02d}", clinic, home, seats, e, e + 30, e, e + 120)
            )
    return requests


def transport_day(n: int = 24, seed: int = 3) -> PdpData:
    """Return the default instance: a normal Tuesday with eight minibuses."""
    return PdpData(
        name="normal day",
        depot=GARAGE,
        requests=tuple(random_requests(n, seed)),
        num_vehicles=8,
        capacity=6,
        shift_start=7 * 60,
        shift_end=16 * 60,
    )


def short_staffed_day(n: int = 24, seed: int = 3) -> PdpData:
    """Return the same requests with only four drivers at work.

    Requests may now be declined; the charity then books a taxi. A taxi
    ride costs about $30, as much as 60 km of bus driving.
    """
    return replace(
        transport_day(n, seed),
        name="short-staffed day",
        num_vehicles=4,
        decline_penalty=60_000,
    )


def small_instance(
    n: int,
    seed: int,
    num_vehicles: int = 2,
    max_detour: int | None = 15,
    declines: bool = False,
) -> PdpData:
    """Return a tiny instance for exact cross-checks (brute force, CP-SAT)."""
    rng = random.Random(seed)
    requests = []
    for k in range(1, n + 1):
        a, b = (
            Place(f"p{k}", round(rng.uniform(-6, 6), 2), round(rng.uniform(-6, 6), 2)),
            Place(f"d{k}", round(rng.uniform(-6, 6), 2), round(rng.uniform(-6, 6), 2)),
        )
        t = rng.choice([8 * 60, 8 * 60 + 30, 9 * 60])
        requests.append(
            Request(f"r{k}", a, b, rng.randint(1, 3), t, t + 60, t, t + 150)
        )
    return PdpData(
        name=f"small (n={n}, seed={seed})",
        depot=GARAGE,
        requests=tuple(requests),
        num_vehicles=num_vehicles,
        capacity=4,
        shift_start=7 * 60,
        shift_end=12 * 60,
        fixed_cost=10_000,
        max_detour=max_detour,
        decline_penalty=25_000 if declines else None,
    )
