"""Independent checks for unit commitment schedules. Uses no OR-Tools code.

* `feasibility_errors` checks every rule of a schedule, hour by hour.
* `cost_breakdown` recomputes the cost from the data.
* `brute_force` solves small instances exactly: it tries every on/off
  plan that obeys the minimum up/down times and dispatches each hour by
  the merit order.
"""

import itertools
import math

from .data import GridData, Unit

TOL = 1e-6


def starts_and_stops(unit: Unit, on: tuple[int, ...]) -> tuple[list[int], list[int]]:
    """Return the hours in which the unit starts and stops."""
    before = [1 if unit.initial_on else 0, *on[:-1]]
    starts = [t for t, (a, b) in enumerate(zip(before, on, strict=True)) if b > a]
    stops = [t for t, (a, b) in enumerate(zip(before, on, strict=True)) if b < a]
    return starts, stops


def min_up_down_errors(unit: Unit, on: tuple[int, ...]) -> list[str]:
    """Check the minimum up and down times, including the state before hour 0."""
    errors = []
    hours = len(on)
    # Treat the state before the plan as a run that started earlier.
    run_state = 1 if unit.initial_on else 0
    run_start = -unit.initial_hours
    for t in range(hours + 1):
        state = on[t] if t < hours else None  # None closes the last run.
        if state == run_state:
            continue
        length = t - run_start
        need = unit.min_up if run_state == 1 else unit.min_down
        # A run that ends at the horizon may be shorter: it continues tomorrow.
        if state is not None and length < need:
            kind = "on" if run_state == 1 else "off"
            errors.append(
                f"{unit.name}: {kind} for {length} h before hour {t}, needs {need} h"
            )
        run_state, run_start = state, t
    return errors


def feasibility_errors(data: GridData, on, power, solar_used, shed) -> list[str]:
    """Return a list of broken rules. An empty list means feasible.

    `on` and `power` map each unit name to a sequence over the hours.
    """
    errors = []
    hours = range(data.hours)
    for g in data.units:
        u, p = on[g.name], power[g.name]
        errors += min_up_down_errors(g, u)
        starts, stops = starts_and_stops(g, u)
        for t in hours:
            if u[t] not in (0, 1):
                errors.append(f"{g.name}, hour {t}: on = {u[t]} is not 0 or 1")
            low, high = (g.p_min, g.p_max) if u[t] else (0.0, 0.0)
            if p[t] < low - TOL or p[t] > high + TOL:
                errors.append(
                    f"{g.name}, hour {t}: {p[t]:.2f} MW not in [{low}, {high}]"
                )
            if g.ramp is not None:
                p_prev = p[t - 1] if t > 0 else g.initial_output
                jump = max(g.p_min, g.ramp)
                up_limit = jump if t in starts else g.ramp
                down_limit = jump if t in stops else g.ramp
                if p[t] - p_prev > up_limit + TOL:
                    errors.append(
                        f"{g.name}, hour {t}: ramps up {p[t] - p_prev:.1f} MW"
                    )
                if p_prev - p[t] > down_limit + TOL:
                    errors.append(
                        f"{g.name}, hour {t}: ramps down {p_prev - p[t]:.1f} MW"
                    )
        if g.energy_limit is not None and sum(p) > g.energy_limit + TOL:
            errors.append(f"{g.name}: uses {sum(p):.1f} MWh of {g.energy_limit} MWh")
    for t in hours:
        if solar_used[t] < -TOL or solar_used[t] > data.solar[t] + TOL:
            errors.append(f"hour {t}: solar {solar_used[t]:.1f} MW not available")
        if shed[t] < -TOL:
            errors.append(f"hour {t}: negative shed {shed[t]}")
        supply = sum(power[g.name][t] for g in data.units) + solar_used[t] + shed[t]
        if abs(supply - data.demand[t]) > 1e-4 * data.demand[t]:
            errors.append(f"hour {t}: supply {supply:.2f} != demand {data.demand[t]}")
        spare = sum(g.p_max * on[g.name][t] - power[g.name][t] for g in data.units)
        if spare < data.reserve(t) - 1e-4 * data.demand[t]:
            errors.append(
                f"hour {t}: reserve {spare:.1f} MW < {data.reserve(t):.1f} MW"
            )
    return errors


def cost_breakdown(data: GridData, on, power, shed) -> dict[str, float]:
    """Return the cost of a schedule, split into its parts."""
    energy = no_load = startup = 0.0
    for g in data.units:
        energy += g.marginal_cost * sum(power[g.name])
        no_load += g.no_load_cost * sum(on[g.name])
        startup += g.startup_cost * len(starts_and_stops(g, on[g.name])[0])
    shed_cost = data.shed_cost * sum(shed)
    return {
        "energy": energy,
        "no-load": no_load,
        "startups": startup,
        "unserved": shed_cost,
        "total": energy + no_load + startup + shed_cost,
    }


def merit_order_dispatch(
    data: GridData, t: int, running: tuple[Unit, ...]
) -> tuple[float, dict[str, float]] | None:
    """Return the cheapest output split for the running units in hour t.

    Solar is free, so use all of it unless the running units cannot go
    low enough. Every running unit gives at least its minimum; the rest of
    the load goes to the cheapest units first. With linear costs and no
    ramp or energy limits, this greedy rule is optimal. Return None if the
    running units cannot meet the load and the reserve.
    """
    floor = sum(g.p_min for g in running)
    ceiling = sum(g.p_max for g in running)
    load = max(data.demand[t] - data.solar[t], floor)
    if load > data.demand[t] + TOL or load > ceiling + TOL:
        return None
    if ceiling - load < data.reserve(t) - TOL:
        return None
    output = {g.name: g.p_min for g in running}
    rest = load - floor
    for g in sorted(running, key=lambda g: g.marginal_cost):
        extra = min(rest, g.p_max - g.p_min)
        output[g.name] += extra
        rest -= extra
    cost = sum(g.marginal_cost * output[g.name] + g.no_load_cost for g in running)
    return cost, output


def brute_force(data: GridData) -> tuple[float, dict[str, tuple[int, ...]]]:
    """Return the optimal cost and on/off plan of a small instance.

    Only for instances without ramp limits and energy limits: then each
    hour's dispatch depends only on which units run in that hour.
    """
    if any(g.ramp is not None or g.energy_limit is not None for g in data.units):
        raise ValueError(
            "brute force needs an instance without ramps and energy limits"
        )
    # Each unit's own valid on/off sequences (min up/down, initial state).
    options = []
    for g in data.units:
        valid = []
        for seq in itertools.product((0, 1), repeat=data.hours):
            if not min_up_down_errors(g, seq):
                valid.append((seq, g.startup_cost * len(starts_and_stops(g, seq)[0])))
        options.append(valid)

    hour_cost: dict[tuple[int, tuple[int, ...]], float | None] = {}
    best_cost, best_plan = math.inf, None
    for combo in itertools.product(*options):
        total = sum(cost for _, cost in combo)
        for t in range(data.hours):
            state = tuple(seq[t] for seq, _ in combo)
            if (t, state) not in hour_cost:
                running = tuple(g for g, s in zip(data.units, state, strict=True) if s)
                result = merit_order_dispatch(data, t, running)
                hour_cost[t, state] = None if result is None else result[0]
            cost = hour_cost[t, state]
            if cost is None:
                break
            total += cost
        else:
            if total < best_cost:
                best_cost = total
                best_plan = {
                    g.name: seq for g, (seq, _) in zip(data.units, combo, strict=True)
                }
    if best_plan is None:
        raise ValueError("no feasible on/off plan")
    return best_cost, best_plan
