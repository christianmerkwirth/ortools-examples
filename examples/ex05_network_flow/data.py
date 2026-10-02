"""Instance data for the supply chain network flow example.

A bottled-water company has two plants, three distribution centers (DCs),
and six stores. Trucks move pallets along fixed lanes. Each lane carries
at most a fixed number of pallets per week and has a cost per pallet.

All numbers are integers: pallets per week and dollars per pallet. The
OR-Tools graph solvers work with integers only (see the README).
"""

import random
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Plant:
    """A factory that makes pallets."""

    name: str
    capacity: int  # Pallets per week.
    unit_cost: int  # Production cost, dollars per pallet.


@dataclass(frozen=True)
class Store:
    """A store that sells pallets."""

    name: str
    demand: int  # Pallets per week.


@dataclass(frozen=True)
class Lane:
    """A truck lane from one site to another."""

    origin: str
    destination: str
    capacity: int  # Pallets per week.
    unit_cost: int  # Transport cost, dollars per pallet.

    @property
    def key(self) -> tuple[str, str]:
        """Return the (origin, destination) pair that names this lane."""
        return (self.origin, self.destination)


@dataclass(frozen=True)
class NetworkData:
    """A full problem instance."""

    plants: tuple[Plant, ...]
    warehouses: tuple[str, ...]
    stores: tuple[Store, ...]
    lanes: tuple[Lane, ...]

    @property
    def total_demand(self) -> int:
        """Return the pallets all stores want per week."""
        return sum(s.demand for s in self.stores)

    def lane(self, origin: str, destination: str) -> Lane:
        """Return the lane between two sites."""
        return next(x for x in self.lanes if x.key == (origin, destination))

    def with_lane_capacity(
        self, origin: str, destination: str, capacity: int
    ) -> "NetworkData":
        """Return a copy with a new capacity on one lane."""
        lanes = tuple(
            replace(x, capacity=capacity) if x.key == (origin, destination) else x
            for x in self.lanes
        )
        return replace(self, lanes=lanes)


def water_network() -> NetworkData:
    """Return the default instance: one week of bottled-water deliveries."""
    plants = (
        Plant("north plant", capacity=900, unit_cost=20),
        Plant("south plant", capacity=700, unit_cost=14),
    )
    warehouses = ("west DC", "central DC", "east DC")
    stores = (
        Store("Alder", 180),
        Store("Birch", 220),
        Store("Cedar", 260),
        Store("Dunmore", 200),
        Store("Elm", 240),
        Store("Fairview", 320),
    )
    lanes = (
        # Plants to DCs.
        Lane("north plant", "west DC", 400, 4),
        Lane("north plant", "central DC", 500, 3),
        Lane("north plant", "east DC", 120, 7),
        Lane("south plant", "central DC", 300, 5),
        Lane("south plant", "east DC", 200, 4),
        # Transfers between DCs.
        Lane("west DC", "central DC", 100, 2),
        Lane("central DC", "east DC", 80, 3),
        # DCs to stores.
        Lane("west DC", "Alder", 200, 3),
        Lane("west DC", "Birch", 250, 4),
        Lane("central DC", "Birch", 150, 5),
        Lane("central DC", "Cedar", 300, 3),
        Lane("central DC", "Dunmore", 200, 4),
        Lane("east DC", "Dunmore", 150, 3),
        Lane("east DC", "Elm", 250, 2),
        Lane("east DC", "Fairview", 250, 3),
        # A direct lane from a plant to a store, for urgent loads.
        Lane("south plant", "Fairview", 100, 9),
    )
    return NetworkData(plants, warehouses, stores, lanes)


def random_network(
    seed: int, n_plants: int = 3, n_warehouses: int = 4, n_stores: int = 8
) -> NetworkData:
    """Return a random layered network for tests."""
    rng = random.Random(seed)
    plants = tuple(
        Plant(f"P{k}", rng.randint(200, 900), rng.randint(5, 30))
        for k in range(n_plants)
    )
    warehouses = tuple(f"W{k}" for k in range(n_warehouses))
    stores = tuple(Store(f"S{k}", rng.randint(50, 300)) for k in range(n_stores))
    lanes = []
    for p in plants:
        for w in warehouses:
            if rng.random() < 0.7:
                lanes.append(Lane(p.name, w, rng.randint(50, 500), rng.randint(1, 10)))
    for a in warehouses:
        for b in warehouses:
            if a != b and rng.random() < 0.2:
                lanes.append(Lane(a, b, rng.randint(20, 200), rng.randint(1, 5)))
    for w in warehouses:
        for s in stores:
            if rng.random() < 0.5:
                lanes.append(Lane(w, s.name, rng.randint(30, 300), rng.randint(1, 10)))
    return NetworkData(plants, warehouses, stores, tuple(lanes))
