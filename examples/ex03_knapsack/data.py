"""Instance data for the knapsack example: loading relief supplies.

After an earthquake, a relief agency has more supplies in its warehouse
than it can move at once. Each item has a priority score (how much it
helps in the first days), a weight, and a volume.

* Part A: one cargo plane with a weight limit and a volume limit.
* Part B: a convoy of three trucks. One truck is refrigerated, and the
  cold-chain items may only go on that truck. Fuel and medical oxygen are
  hazardous goods and must not travel on the same truck.

Scores and sizes are illustrative.
"""

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Item:
    """One pallet or crate of supplies."""

    name: str
    value: int  # Priority score.
    weight: int  # kg.
    volume: float  # m³, one decimal.
    cold_chain: bool = False  # Must travel refrigerated.


@dataclass(frozen=True)
class Vehicle:
    """A plane or a truck."""

    name: str
    max_weight: int  # kg.
    max_volume: float  # m³.
    refrigerated: bool = False


@dataclass(frozen=True)
class LoadingData:
    """A full problem instance."""

    items: tuple[Item, ...]
    vehicles: tuple[Vehicle, ...]
    # Pairs of item names that must not share a vehicle.
    incompatible: tuple[tuple[str, str], ...] = ()


ITEMS = (
    Item("water purification unit", 95, 450, 2.4),
    Item("field hospital tent", 90, 700, 4.5),
    Item("surgical kits", 85, 180, 0.8),
    Item("vaccines", 80, 120, 0.6, cold_chain=True),
    Item("insulin", 75, 60, 0.3, cold_chain=True),
    Item("medical oxygen", 70, 400, 1.2),
    Item("diesel generator", 65, 600, 1.8),
    Item("fuel drums", 60, 800, 1.6),
    Item("family tents (20)", 60, 500, 3.5),
    Item("blankets (500)", 45, 350, 4.0),
    Item("food rations (1000)", 70, 900, 3.0),
    Item("baby formula", 55, 200, 0.9),
    Item("water bottles (2000 l)", 50, 2000, 2.6),
    Item("hygiene kits (300)", 40, 300, 1.5),
    Item("tarpaulins (200)", 35, 400, 1.4),
    Item("satellite phones", 50, 40, 0.2),
    Item("chainsaws and tools", 30, 250, 0.9),
    Item("solar lamps (500)", 30, 150, 0.8),
    Item("blood bags", 70, 90, 0.4, cold_chain=True),
    Item("cooking sets (200)", 25, 300, 1.6),
    Item("water containers (500)", 35, 250, 2.2),
    Item("mosquito nets (1000)", 30, 200, 1.0),
)


def cargo_plane() -> LoadingData:
    """Part A: one plane, a two-dimensional (weight, volume) knapsack."""
    plane = Vehicle("plane", max_weight=4000, max_volume=15.0, refrigerated=True)
    return LoadingData(ITEMS, (plane,))


def truck_convoy() -> LoadingData:
    """Part B: three different trucks with side rules (multiple knapsack)."""
    trucks = (
        Vehicle("reefer truck", 1800, 8.0, refrigerated=True),
        Vehicle("large truck", 3200, 12.0),
        Vehicle("medium truck", 2200, 9.0),
    )
    return LoadingData(ITEMS, trucks, incompatible=(("fuel drums", "medical oxygen"),))


def random_instance(
    n_items: int, n_vehicles: int = 1, seed: int = 0, tightness: float = 0.4
) -> LoadingData:
    """Return a random instance for tests and timing experiments.

    The vehicles together hold about `tightness` of the total weight and
    volume, so only some items fit.
    """
    rng = random.Random(seed)
    items = tuple(
        Item(
            f"item {k}",
            value=rng.randint(10, 100),
            weight=rng.randint(20, 800),
            volume=round(rng.uniform(0.2, 4.0), 1),
            cold_chain=rng.random() < 0.15,
        )
        for k in range(n_items)
    )
    total_w = sum(i.weight for i in items)
    total_v = sum(i.volume for i in items)
    vehicles = tuple(
        Vehicle(
            f"vehicle {k}",
            max_weight=int(tightness * total_w / n_vehicles),
            max_volume=round(tightness * total_v / n_vehicles, 1),
            refrigerated=(k == 0),
        )
        for k in range(n_vehicles)
    )
    return LoadingData(items, vehicles)
