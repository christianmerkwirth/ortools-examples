"""Instance data for the cutting stock example.

A paper mill makes "jumbo" rolls of one standard width. Customers order
narrower rolls. A slitter cuts each jumbo roll across its width into a
few narrow rolls. The question: how should we cut, so that we use as few
jumbo rolls as possible?

The slitter has a fixed number of knives, so one jumbo roll gives at most
`max_pieces` narrow rolls. Widths and quantities are illustrative.
"""

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Order:
    """Narrow rolls that one customer wants, all of one width."""

    width: int  # mm.
    quantity: int  # Number of narrow rolls.


@dataclass(frozen=True)
class CuttingData:
    """A full problem instance."""

    name: str
    roll_width: int  # Width of one jumbo roll, mm.
    orders: tuple[Order, ...]
    max_pieces: int  # The slitter cuts at most this many rolls from one jumbo.

    @property
    def widths(self) -> list[int]:
        """Return the ordered widths, in order of `orders`."""
        return [o.width for o in self.orders]

    @property
    def demand(self) -> list[int]:
        """Return the ordered quantities, in order of `orders`."""
        return [o.quantity for o in self.orders]


def paper_mill() -> CuttingData:
    """Return the default instance: one week of orders at a paper mill."""
    orders = (
        Order(2150, 18),
        Order(1930, 20),
        Order(1820, 18),
        Order(1710, 14),
        Order(1520, 25),
        Order(1380, 22),
        Order(1150, 34),
        Order(860, 40),
        Order(620, 26),
        Order(450, 30),
    )
    return CuttingData("paper mill", 5000, orders, max_pieces=6)


def rush_order() -> CuttingData:
    """Return a small instance where column generation alone falls short.

    Wide rolls (more than half the jumbo width) fit poorly together. The
    LP bound is 17 rolls, and 17 is possible. But the patterns that column
    generation finds need not allow 17: in our runs they allow only 18.
    """
    orders = (Order(2600, 12), Order(1700, 15), Order(1300, 9), Order(1100, 10))
    return CuttingData("rush order", 5000, orders, max_pieces=6)


def tiny() -> CuttingData:
    """Return a tiny instance that check.py can solve by exhaustive search."""
    orders = (Order(450, 4), Order(360, 3), Order(310, 5), Order(140, 4))
    return CuttingData("tiny", 1000, orders, max_pieces=4)


def random_instance(
    n_widths: int, seed: int = 0, roll_width: int = 1000
) -> CuttingData:
    """Return a random instance for tests."""
    rng = random.Random(seed)
    widths = rng.sample(range(roll_width // 10, roll_width // 2), n_widths)
    orders = tuple(Order(w, rng.randint(1, 12)) for w in sorted(widths, reverse=True))
    return CuttingData(f"random {seed}", roll_width, orders, max_pieces=5)
