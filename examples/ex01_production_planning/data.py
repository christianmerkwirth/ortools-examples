"""Instance data for the production planning example.

A small furniture workshop makes four products. Each product needs wood,
carpentry time, and finishing time. The workshop has a fixed weekly supply of
each resource and wants to know how many units of each product to make.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Resource:
    """A limited input such as raw material or labor hours."""

    name: str
    unit: str  # Singular, for example "hour".
    capacity: float  # Amount available per week.


@dataclass(frozen=True)
class Product:
    """A product the workshop can make."""

    name: str
    profit: float  # Profit per unit in dollars.
    usage: dict[str, float]  # Resource name -> amount used per unit.
    max_demand: float  # The market buys at most this many units per week.
    min_order: float = 0.0  # Units already promised to customers.


@dataclass(frozen=True)
class ProductionData:
    """A full problem instance."""

    resources: tuple[Resource, ...]
    products: tuple[Product, ...]
    name: str = field(default="production")

    def resource(self, name: str) -> Resource:
        """Return the resource with this name."""
        return next(r for r in self.resources if r.name == name)

    def with_capacity(self, name: str, capacity: float) -> "ProductionData":
        """Return a copy with a changed capacity for one resource."""
        resources = tuple(
            Resource(r.name, r.unit, capacity) if r.name == name else r
            for r in self.resources
        )
        return ProductionData(resources, self.products, self.name)

    def with_profit(self, name: str, profit: float) -> "ProductionData":
        """Return a copy with a changed unit profit for one product."""
        products = tuple(
            Product(p.name, profit, p.usage, p.max_demand, p.min_order)
            if p.name == name
            else p
            for p in self.products
        )
        return ProductionData(self.resources, products, self.name)


def furniture_workshop() -> ProductionData:
    """Return the default instance: one week in a furniture workshop."""
    resources = (
        Resource("wood", "board-foot", 1200),
        Resource("carpentry", "hour", 400),
        Resource("finishing", "hour", 280),
    )
    products = (
        Product(
            "chair",
            profit=45,
            usage={"wood": 5, "carpentry": 2, "finishing": 1},
            max_demand=120,
        ),
        Product(
            "table",
            profit=80,
            usage={"wood": 20, "carpentry": 5, "finishing": 2},
            max_demand=40,
            min_order=10,  # A hotel has ordered 10 tables this week.
        ),
        Product(
            "desk",
            profit=110,
            usage={"wood": 30, "carpentry": 8, "finishing": 4},
            max_demand=25,
        ),
        Product(
            "bookshelf",
            profit=60,
            usage={"wood": 12, "carpentry": 4, "finishing": 3},
            max_demand=30,
        ),
    )
    return ProductionData(resources, products, name="furniture workshop")
