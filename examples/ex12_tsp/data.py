"""Instance data for the traveling salesperson example.

A field technician must inspect cell towers spread over a region of
100 km x 100 km. The technician starts at the depot, visits every tower
once, and comes back. Which order gives the shortest drive?

We use straight-line distances between map coordinates. Real projects use
road distances from a map service; the model stays the same.
"""

import math
import random
from dataclasses import dataclass

# The routing library works with integers only (int64 costs). So we count
# distances in meters, not in kilometers with decimals. One meter of
# rounding per leg is far below anything that matters on a 100 km map.
METERS_PER_KM = 1000


@dataclass(frozen=True)
class Site:
    """A place to visit, with map coordinates in km."""

    name: str
    x: float
    y: float


@dataclass(frozen=True)
class TspData:
    """A full problem instance. Site 0 is the depot."""

    name: str
    sites: tuple[Site, ...]
    depot: int = 0

    @property
    def size(self) -> int:
        """Return the number of sites, depot included."""
        return len(self.sites)


def distance_matrix(data: TspData) -> list[list[int]]:
    """Return the integer distance in meters between every pair of sites."""
    return [
        [round(METERS_PER_KM * math.hypot(a.x - b.x, a.y - b.y)) for b in data.sites]
        for a in data.sites
    ]


def random_sites(n: int, seed: int = 0, size_km: float = 100.0) -> TspData:
    """Return n uniformly random sites in a square. Site 0 is the depot."""
    rng = random.Random(seed)
    sites = [Site("depot", rng.uniform(0, size_km), rng.uniform(0, size_km))]
    sites += [
        Site(f"tower {k}", rng.uniform(0, size_km), rng.uniform(0, size_km))
        for k in range(1, n)
    ]
    return TspData(f"{n} random sites (seed {seed})", tuple(sites))


def cell_towers() -> TspData:
    """Return the default instance: a depot and 99 cell towers."""
    data = random_sites(100, seed=42)
    return TspData("depot + 99 cell towers", data.sites, data.depot)
