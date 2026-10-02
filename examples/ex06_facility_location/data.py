"""Instance data for the facility location example.

A grocery wholesaler wants to serve the towns of a region from a few
distribution centers (DCs). There are candidate sites for the DCs. Each
site has a weekly fixed cost (rent, staff, energy) and a capacity. Each
town orders a number of pallets per week and must get all of them from
one DC (single sourcing keeps delivery routes and invoices simple).

Opening more DCs makes trucks drive less, but each DC costs money. The
model finds the best balance.

The towns and sites are random points on a 100 km x 100 km map, made
from a fixed seed, so every run gives the same instance.
"""

import math
import random
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Site:
    """A candidate location for a distribution center."""

    name: str
    x: float  # km.
    y: float  # km.
    fixed_cost: float  # Dollars per week if the DC is open.
    capacity: int  # Pallets per week.


@dataclass(frozen=True)
class Customer:
    """A town that orders pallets every week."""

    name: str
    x: float
    y: float
    demand: int  # Pallets per week.


@dataclass(frozen=True)
class LocationData:
    """A full problem instance."""

    name: str
    sites: tuple[Site, ...]
    customers: tuple[Customer, ...]
    # Dollars per pallet and km. It covers fuel, the driver's time, and
    # the truck, spread over the pallets on board.
    cost_per_pallet_km: float = 2.0

    def transport_cost(self, customer: Customer, site: Site) -> float:
        """Return the weekly cost to serve one customer from one site.

        Trucks drive out and back, so we count the distance twice.
        """
        distance = math.hypot(customer.x - site.x, customer.y - site.y)
        return 2 * distance * customer.demand * self.cost_per_pallet_km

    def with_capacity_factor(self, factor: float) -> "LocationData":
        """Return a copy with every site capacity scaled by `factor`."""
        sites = tuple(
            replace(s, capacity=round(s.capacity * factor)) for s in self.sites
        )
        return replace(self, sites=sites)


def random_region(
    n_sites: int, n_customers: int, seed: int = 0, name: str = "region"
) -> LocationData:
    """Return a random instance on a 100 km x 100 km map.

    Total capacity is about 2.5 times total demand, so the model has a real
    choice of which sites to open.
    """
    rng = random.Random(seed)
    customers = tuple(
        Customer(
            f"town {k + 1}",
            x=round(rng.uniform(0, 100), 1),
            y=round(rng.uniform(0, 100), 1),
            demand=rng.randint(10, 60),
        )
        for k in range(n_customers)
    )
    total_demand = sum(c.demand for c in customers)
    mean_capacity = 2.5 * total_demand / n_sites
    sites = []
    for k in range(n_sites):
        capacity = round(mean_capacity * rng.uniform(0.7, 1.3))
        sites.append(
            Site(
                _site_name(k),
                x=round(rng.uniform(5, 95), 1),
                y=round(rng.uniform(5, 95), 1),
                # Bigger DCs cost more, with economies of scale.
                fixed_cost=round(4000 + 9 * capacity**0.9, -1),
                capacity=capacity,
            )
        )
    return LocationData(name, tuple(sites), customers)


def _site_name(k: int) -> str:
    return "site " + chr(ord("A") + k)


def wholesaler() -> LocationData:
    """Return the default instance: 12 candidate sites, 60 towns."""
    return random_region(12, 60, seed=6, name="grocery wholesaler")


def small_region() -> LocationData:
    """Return a tiny instance that a brute-force search can solve."""
    return random_region(5, 8, seed=3, name="small region")
