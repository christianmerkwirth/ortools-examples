"""Instance data for the unit commitment example.

A small grid operator plans tomorrow, hour by hour. Nine power plants can
run. Each plant has a minimum and maximum output when it runs, a cost per
MWh, a cost per hour just for running ("no-load"), a cost to start, and
minimum times it must stay on or off once switched. Solar panels feed the
grid for free during the day, so the load that the plants must cover dips
at noon and peaks in the evening: the "duck curve".

All numbers are illustrative, but their relative sizes follow real plant
types: nuclear is cheap to run and slow to change, gas peakers are
expensive to run and quick to start.
"""

import random
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Unit:
    """One power plant (a generating unit)."""

    name: str
    kind: str
    p_min: float  # MW when on.
    p_max: float  # MW when on.
    marginal_cost: float  # $ per MWh produced.
    no_load_cost: float  # $ per hour while on, whatever the output.
    startup_cost: float  # $ per start.
    min_up: int  # Hours the unit must stay on after a start.
    min_down: int  # Hours the unit must stay off after a stop.
    ramp: float | None  # MW per hour up or down; None means no limit.
    initial_on: bool  # State in the hour before the plan starts.
    initial_hours: int  # Hours it has been in that state.
    initial_output: float  # MW in the hour before the plan starts.
    energy_limit: float | None = None  # MWh per day (water in the dam).


@dataclass(frozen=True)
class GridData:
    """A full problem instance."""

    name: str
    units: tuple[Unit, ...]
    demand: tuple[float, ...]  # MW per hour.
    solar: tuple[float, ...]  # MW of solar power available per hour.
    reserve_share: float = 0.10  # Spare running capacity, share of demand.
    shed_cost: float = 5000.0  # $ per MWh of demand not served (last resort).

    @property
    def hours(self) -> int:
        """Return the number of hours in the plan."""
        return len(self.demand)

    def reserve(self, t: int) -> float:
        """Return the spinning reserve needed in hour t (MW)."""
        return self.reserve_share * self.demand[t]

    def with_solar(self, factor: float, name: str) -> "GridData":
        """Return a copy with the solar profile scaled, e.g. for a cloudy day."""
        return replace(self, name=name, solar=tuple(factor * s for s in self.solar))


# Demand in MW for hours 0..23 (hour 0 = midnight to 1 am). A typical
# weekday: low at night, a morning rise, and an evening peak at 19:00.
DEMAND = (
    1450, 1400, 1370, 1360, 1380, 1460,
    1620, 1820, 1950, 2000, 2020, 2030,
    2020, 2010, 2000, 2010, 2080, 2200,
    2350, 2400, 2300, 2100, 1850, 1600,
)  # fmt: skip

# Solar output in MW on a sunny day: zero at night, peak at noon. At
# midday it covers more than half of the demand.
SOLAR_SUNNY = (
    0, 0, 0, 0, 0, 0,
    30, 200, 490, 760, 970, 1110,
    1150, 1110, 970, 760, 490, 200,
    30, 0, 0, 0, 0, 0,
)  # fmt: skip

UNITS = (
    Unit("Riverbend", "nuclear", 500, 700, 9.0, 0.0, 80_000, 24, 24, 40,
         True, 200, 700),
    Unit("Coalport A", "coal", 150, 400, 28.0, 1_800.0, 30_000, 8, 8, 100,
         True, 30, 300),
    Unit("Coalport B", "coal", 150, 400, 30.0, 1_800.0, 30_000, 8, 8, 100,
         True, 30, 300),
    Unit("Lakeside CC1", "gas CCGT", 120, 350, 42.0, 2_500.0, 12_000, 6, 5, 180,
         True, 10, 150),
    Unit("Lakeside CC2", "gas CCGT", 120, 350, 44.0, 2_500.0, 12_000, 6, 5, 180,
         False, 6, 0),
    Unit("Highfield GT1", "gas peaker", 20, 120, 85.0, 200.0, 600, 1, 1, None,
         False, 12, 0),
    Unit("Highfield GT2", "gas peaker", 20, 120, 90.0, 200.0, 600, 1, 1, None,
         False, 12, 0),
    Unit("Glen Dam", "hydro", 20, 250, 2.0, 0.0, 100, 1, 1, None,
         True, 5, 60, energy_limit=1_800),
    Unit("Harbor diesel", "diesel", 10, 80, 160.0, 50.0, 100, 1, 1, None,
         False, 48, 0),
)  # fmt: skip


def sunny_day() -> GridData:
    """Return the default instance: a sunny weekday with strong solar."""
    return GridData("sunny weekday", UNITS, DEMAND, SOLAR_SUNNY)


def cloudy_day() -> GridData:
    """Return the same day with only 30% of the solar power."""
    return sunny_day().with_solar(0.3, "cloudy weekday")


def tiny_grid() -> GridData:
    """Return a 3-unit, 6-hour instance that brute force can solve.

    It has no ramp limits and no energy limit, so the best output for a
    fixed on/off schedule follows the simple merit order.
    """
    units = (
        Unit("base", "coal", 100, 300, 20.0, 500.0, 3_000, 3, 2, None,
             True, 5, 200),
        Unit("mid", "gas CCGT", 50, 200, 40.0, 300.0, 1_500, 2, 2, None,
             False, 1, 0),
        Unit("peak", "gas peaker", 10, 100, 80.0, 50.0, 300, 1, 1, None,
             False, 4, 0),
    )  # fmt: skip
    demand = (220, 260, 380, 450, 330, 240)
    solar = (0, 30, 80, 60, 20, 0)
    return GridData("tiny grid", units, demand, solar)


def large_grid(n_units: int = 40, days: int = 2, seed: int = 0) -> GridData:
    """Return a random fleet over several days, for the formulation benchmark.

    The fleet copies the plant types above with random sizes and costs, and
    the demand and solar profiles repeat each day with some noise. The
    total capacity is set so that the peak plus reserve needs about 85%
    of the fleet.
    """
    rng = random.Random(seed)
    templates = [g for g in UNITS if g.energy_limit is None]
    units = []
    for k in range(n_units):
        g = rng.choice(templates)
        size = rng.uniform(0.6, 1.4)
        cost = rng.uniform(0.9, 1.1)
        units.append(
            replace(
                g,
                name=f"{g.kind} {k}",
                p_min=round(g.p_min * size),
                p_max=round(g.p_max * size),
                marginal_cost=round(g.marginal_cost * cost, 2),
                no_load_cost=round(g.no_load_cost * cost),
                startup_cost=round(g.startup_cost * cost),
                ramp=None if g.ramp is None else round(g.ramp * size),
                initial_on=g.kind in ("nuclear", "coal"),
                initial_hours=48,
                initial_output=round(g.p_min * size)
                if g.kind in ("nuclear", "coal")
                else 0,
            )
        )
    capacity = sum(g.p_max for g in units)
    scale = 0.85 * capacity / (1.1 * max(DEMAND))
    demand, solar = [], []
    for _ in range(days):
        for t in range(24):
            demand.append(round(DEMAND[t] * scale * rng.uniform(0.97, 1.03)))
            solar.append(round(SOLAR_SUNNY[t] * scale * rng.uniform(0.6, 1.0)))
    return GridData(
        f"random fleet ({n_units} units, {days} days)",
        tuple(units),
        tuple(demand),
        tuple(solar),
    )
