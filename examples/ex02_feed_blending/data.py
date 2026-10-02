"""Instance data for the feed blending example.

A feed mill mixes ingredients into chicken feed. Every batch must meet a
nutrient specification: a minimum and maximum level for protein, fat,
fiber, minerals, and energy. Some ingredients are limited because too much
of them harms the birds or the taste.

Nutrient values are typical book values for poultry feed ingredients
(rounded). Prices are illustrative.
"""

import math
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Ingredient:
    """A raw material that can go into the blend."""

    name: str
    cost: float  # Dollars per kg.
    nutrients: dict[str, float]  # Nutrient name -> level in this ingredient.
    max_share: float = 1.0  # Largest allowed fraction of the blend.


@dataclass(frozen=True)
class Requirement:
    """The allowed range of one nutrient in the final blend."""

    nutrient: str
    unit: str
    min: float = 0.0
    max: float = math.inf


@dataclass(frozen=True)
class BlendData:
    """A full problem instance."""

    name: str
    ingredients: tuple[Ingredient, ...]
    requirements: tuple[Requirement, ...]
    batch_kg: float = 1000.0  # Costs are reported per batch (one tonne).

    def requirement(self, nutrient: str) -> Requirement:
        """Return the requirement for one nutrient."""
        return next(r for r in self.requirements if r.nutrient == nutrient)

    def with_requirement(self, nutrient: str, **changes: float) -> "BlendData":
        """Return a copy with a changed min and/or max for one nutrient."""
        reqs = tuple(
            replace(r, **changes) if r.nutrient == nutrient else r
            for r in self.requirements
        )
        return replace(self, requirements=reqs)


# Levels per kg of ingredient: protein, fat, fiber, calcium, and phosphorus
# in percent; metabolizable energy in kcal/kg.
INGREDIENTS = (
    Ingredient(
        "corn",
        0.22,
        {
            "protein": 8.5,
            "fat": 3.8,
            "fiber": 2.2,
            "calcium": 0.02,
            "phosphorus": 0.28,
            "energy": 3350,
        },
    ),
    Ingredient(
        "soybean meal",
        0.45,
        {
            "protein": 44.0,
            "fat": 1.5,
            "fiber": 7.0,
            "calcium": 0.30,
            "phosphorus": 0.65,
            "energy": 2230,
        },
    ),
    Ingredient(
        "wheat bran",
        0.18,
        {
            "protein": 15.5,
            "fat": 4.0,
            "fiber": 11.0,
            "calcium": 0.14,
            "phosphorus": 1.15,
            "energy": 1300,
        },
        max_share=0.10,  # High fiber lowers digestibility.
    ),
    Ingredient(
        "fish meal",
        1.40,
        {
            "protein": 62.0,
            "fat": 9.0,
            "fiber": 1.0,
            "calcium": 5.0,
            "phosphorus": 3.0,
            "energy": 2800,
        },
        max_share=0.05,  # More than 5% gives the meat a fishy taste.
    ),
    Ingredient(
        "soybean oil",
        1.10,
        {"fat": 99.0, "energy": 8800},
        max_share=0.05,  # More makes the pellets greasy.
    ),
    Ingredient("limestone", 0.05, {"calcium": 38.0}),
    Ingredient("dicalcium phosphate", 0.70, {"calcium": 22.0, "phosphorus": 18.5}),
)


def broiler_grower() -> BlendData:
    """Return the default instance: a standard feed for growing chickens."""
    requirements = (
        Requirement("protein", "%", 20.0, 23.0),
        Requirement("fat", "%", 3.0, 7.0),
        Requirement("fiber", "%", 0.0, 5.0),
        Requirement("calcium", "%", 0.90, 1.10),
        Requirement("phosphorus", "%", 0.60, 0.90),
        Requirement("energy", "kcal/kg", 3000, 3250),
    )
    return BlendData("broiler grower", INGREDIENTS, requirements)


def premium_finisher() -> BlendData:
    """Return an infeasible instance: a marketing team's "premium" spec.

    It asks for more protein and more energy than the ingredients can give
    together. The example shows how to find out why and what to relax.
    """
    data = broiler_grower()
    data = data.with_requirement("protein", min=24.0, max=26.0)
    data = data.with_requirement("energy", min=3200, max=3300)
    return replace(data, name="premium finisher")
