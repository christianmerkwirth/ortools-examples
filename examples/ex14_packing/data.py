"""Instance data for the packing example.

* Part 1 (1D bin packing): a wholesaler loads crates into delivery vans.
  Each van carries at most 1,000 kg. How few vans can carry one day's
  crates? A week of days shows different kinds of instances.
* Part 2 (2D strip packing): a sign maker cuts acrylic panels from a roll
  that is 120 cm wide. How little of the roll does one order need?

Crate weights come from a seeded random generator. The sign order and the
roll width are illustrative.
"""

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class BinPackingData:
    """Items with sizes, and bins with one shared capacity."""

    name: str
    sizes: dict[str, int]  # Item name -> size (here: kg).
    capacity: int  # Largest total size per bin.


@dataclass(frozen=True)
class Panel:
    """A rectangle to cut, in cm."""

    name: str
    width: int
    height: int
    can_rotate: bool = True  # False for materials with a grain direction.


@dataclass(frozen=True)
class StripData:
    """Rectangles to place on a strip of fixed width and open length."""

    name: str
    panels: tuple[Panel, ...]
    strip_width: int


VAN_CAPACITY = 1000  # kg.

# Seeds chosen so that each day shows a different case. See the README.
WEEK_SEEDS = {"Mon": 16, "Tue": 10, "Wed": 37, "Thu": 59, "Fri": 2}


def crates(day: str, n: int = 36) -> BinPackingData:
    """Return one day's crates: `n` weights between 100 and 700 kg."""
    rng = random.Random(WEEK_SEEDS[day])
    sizes = {f"crate {k + 1}": rng.randint(100, 700) for k in range(n)}
    return BinPackingData(f"{day} crates", sizes, VAN_CAPACITY)


def week() -> list[BinPackingData]:
    """Return the five working days of the week."""
    return [crates(day) for day in WEEK_SEEDS]


def random_bin_packing(n: int, capacity: int, seed: int) -> BinPackingData:
    """Return a random instance for tests."""
    rng = random.Random(seed)
    lo, hi = capacity // 10, 7 * capacity // 10
    sizes = {f"item {k}": rng.randint(lo, hi) for k in range(n)}
    return BinPackingData(f"random {seed}", sizes, capacity)


def sign_order() -> StripData:
    """Return the default 2D instance: one order for a sign maker."""
    panels = (
        Panel("shop front sign", 110, 30),
        Panel("menu board", 60, 80),
        Panel("door sign A", 30, 20),
        Panel("door sign B", 30, 20),
        Panel("door sign C", 30, 20),
        Panel("window display", 70, 50),
        Panel("price tag 1", 20, 20),
        Panel("price tag 2", 20, 20),
        Panel("price tag 3", 20, 20),
        Panel("price tag 4", 20, 20),
        Panel("parking sign", 40, 60),
        Panel("logo panel", 50, 50),
        Panel("arrow left", 80, 20),
        Panel("arrow right", 80, 20),
        Panel("opening hours", 40, 30),
        Panel("exhibition panel", 90, 60),
    )
    return StripData("sign order", panels, strip_width=120)


def guillotine_instance(
    width: int, height: int, n: int, seed: int, min_side: int = 2
) -> StripData:
    """Cut a width x height rectangle into n pieces, then shuffle them.

    The pieces fill the rectangle with no waste, so the best strip length
    is exactly `height`. This gives test instances with a known optimum.
    Each cut is straight across the current piece (a guillotine cut).
    Half of the pieces are turned by 90 degrees, so the solver must find
    the right orientation.
    """
    rng = random.Random(seed)
    pieces = [(width, height)]
    while len(pieces) < n:
        pieces.sort(key=lambda p: p[0] * p[1])
        w, h = pieces.pop()  # Cut the largest piece.
        if (w >= h and w >= 2 * min_side) or h < 2 * min_side:
            cut = rng.randint(min_side, w - min_side)
            pieces += [(cut, h), (w - cut, h)]
        else:
            cut = rng.randint(min_side, h - min_side)
            pieces += [(w, cut), (w, h - cut)]
    turned = [(h, w) if rng.random() < 0.5 else (w, h) for w, h in pieces]
    rng.shuffle(turned)
    panels = tuple(Panel(f"piece {k}", w, h) for k, (w, h) in enumerate(turned))
    return StripData(f"guillotine {seed}", panels, strip_width=width)
