"""Instance data for the planning-under-uncertainty example.

A clothing maker plans its winter season in summer. It must rent sewing
lines and make its stock months before it knows how cold the winter will
be. In the season it can buy a limited number of extra units from an
express subcontractor, and it sells leftovers at an outlet afterwards.

Demand is uncertain, and so is the express price: in a cold winter every
brand wants express capacity, so the price goes up exactly when we need it
most. `sample` draws scenarios for both, with a "spread" knob that sets
how uncertain the forecast is. A spread of 0 means the forecast is exact.

All numbers are synthetic and seeded.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Product:
    """One product. Money in euros per unit, time in sewing hours."""

    name: str
    price: float  # Sale price in the season.
    cost: float  # In-house cost: cloth, trims, and sewing labor.
    salvage: float  # Outlet price for a leftover unit after the season.
    rush_price: float  # Express price from the subcontractor, normal winter.
    rush_cap: float  # Most express units the subcontractor can deliver.
    hours: float  # Sewing hours per unit made in-house.
    mean_demand: float  # The point forecast.


@dataclass(frozen=True)
class Plant:
    """The sewing lines we can rent for the season."""

    line_cost: float = 30_000.0  # Rent and crew for one line, whole season.
    line_hours: float = 1_000.0  # Sewing hours one line gives per season.
    max_lines: int = 12


PRODUCTS = (
    # A parka earns a lot, and a leftover parka still sells well at the
    # outlet. Gloves earn little, and leftover gloves are worth nothing.
    Product("parka", 240.0, 70.0, 65.0, 180.0, 200.0, 2.00, 2_000.0),
    Product("fleece", 80.0, 35.0, 10.0, 65.0, 300.0, 1.00, 3_000.0),
    Product("gloves", 24.0, 12.0, 0.0, 20.0, 500.0, 0.25, 6_000.0),
)


@dataclass(frozen=True)
class PlanningData:
    """Products and plant."""

    products: tuple[Product, ...] = PRODUCTS
    plant: Plant = Plant()

    def column(self, attr: str) -> np.ndarray:
        """Return one attribute of every product as an array."""
        return np.array([getattr(p, attr) for p in self.products])


@dataclass(frozen=True)
class Scenarios:
    """Equally likely futures: demand and express price per product."""

    demand: np.ndarray = field(repr=False)  # demand[s, p]
    rush_price: np.ndarray = field(repr=False)  # rush_price[s, p]

    @property
    def count(self) -> int:
        """Return the number of scenarios."""
        return self.demand.shape[0]

    def subset(self, indices) -> "Scenarios":
        """Return only the given scenarios."""
        return Scenarios(self.demand[indices], self.rush_price[indices])


def point_forecast(data: PlanningData) -> Scenarios:
    """Return the single "average" future: mean demand, normal express price."""
    return Scenarios(
        data.column("mean_demand")[None, :], data.column("rush_price")[None, :]
    )


def sample(data: PlanningData, spread: float, count: int, seed: int) -> Scenarios:
    """Draw `count` equally likely scenarios.

    demand = mean demand * market * own

    `market` is one factor for the whole winter (how cold it is). It moves
    all products together. `own` is a separate factor per product (fashion,
    a competitor's sale). Both are lognormal with mean 1, so the average
    demand stays at the forecast for every spread. Each carries half of the
    variance; `spread` is the total standard deviation of log demand,
    about the coefficient of variation for small spreads.

    The express price follows the market: price * market, kept between the
    outlet price and the sale price. (Below the outlet price, buying express
    to dump it at the outlet would make money. Above the sale price, nobody
    buys express.)
    """
    rng = np.random.default_rng(seed)
    n = len(data.products)
    s = spread / np.sqrt(2.0)
    market = np.exp(s * rng.standard_normal((count, 1)) - s * s / 2)
    own = np.exp(s * rng.standard_normal((count, n)) - s * s / 2)
    demand = data.column("mean_demand") * market * own
    rush = np.clip(
        data.column("rush_price") * market,
        data.column("salvage"),
        data.column("price"),
    )
    return Scenarios(demand, rush)
