"""Plan a day of patient transport: pickups, drop-offs, and ride times.

Run from the repository root:

    uv run python -m examples.ex19_pickup_delivery.main
    uv run python -m examples.ex19_pickup_delivery.main --plot
"""

import argparse
from dataclasses import replace
from pathlib import Path

from examples.common.report import format_table

from .check import extra_ride, lifo_errors, plan_cost, plan_errors
from .data import PdpData, clock, distance_matrix, short_staffed_day, transport_day
from .model import Plan, solve_routing

FIGURES = Path(__file__).parent / "figures"
TIME_LIMIT = 3.0  # Seconds of guided local search per plan.


def verify(data: PdpData, plan: Plan, label: str) -> list[str]:
    """Run the independent checks on one plan."""
    errors = [f"{label}: {e}" for e in plan_errors(data, plan)]
    if plan_cost(data, plan) != plan.cost:
        errors.append(
            f"{label}: cost {plan.cost} != recomputed {plan_cost(data, plan)}"
        )
    return errors


def print_buses(data: PdpData, plan: Plan) -> None:
    """Print one row per bus: requests, distance, times, peak load."""
    nodes = data.stops()
    rows = []
    for k, (route, times) in enumerate(zip(plan.routes, plan.times, strict=True)):
        load, peak = 0, 0
        for node in route[1:-1]:
            load += nodes[node].load
            peak = max(peak, load)
        rows.append(
            (
                f"bus {k + 1}",
                (len(route) - 2) // 2,
                plan_km(data, route),
                clock(times[0]),
                clock(times[-1]),
                f"{peak} / {data.capacity}",
            )
        )
    print(
        format_table(["bus", "requests", "km", "leaves", "back", "most on board"], rows)
    )


def plan_km(data: PdpData, route: list[int]) -> float:
    """Return the km driven on one route."""
    meters = distance_matrix(data)
    return sum(meters[a][b] for a, b in zip(route, route[1:], strict=False)) / 1000


def summary_row(label: str, data: PdpData, plan: Plan) -> tuple:
    """One row of the "price of the rules" table."""
    extra = list(extra_ride(data, plan).values())
    return (
        label,
        len(plan.routes),
        plan.distance / 1000,
        plan.cost / 1000,
        sum(extra) / len(extra),
        max(extra),
    )


def plot_map(data: PdpData, plan: Plan, path: Path) -> Path:
    """Map: one color per bus, pickups as circles, drop-offs as triangles."""
    from examples.common.plotting import COLORS, new_figure, save

    nodes = data.stops()
    fig, ax = new_figure(9, 7)
    for k, route in enumerate(plan.routes):
        color = COLORS[k % len(COLORS)]
        xs = [nodes[i].x for i in route]
        ys = [nodes[i].y for i in route]
        ax.plot(xs, ys, "-", color=color, lw=1.6, alpha=0.85, label=f"bus {k + 1}")
        for node in route[1:-1]:
            marker = "o" if node % 2 == 1 else "^"
            ax.plot(
                nodes[node].x, nodes[node].y, marker, color=color, ms=7, mec="white"
            )
    # Faint arrows from each pickup to its drop-off.
    for r in range(len(data.requests)):
        a, b = nodes[2 * r + 1], nodes[2 * r + 2]
        ax.annotate(
            "",
            xy=(b.x, b.y),
            xytext=(a.x, a.y),
            arrowprops={"arrowstyle": "->", "color": "0.6", "lw": 0.8, "ls": ":"},
        )
    places = {}  # Label the garage and the clinics, not every home.
    for req in data.requests:
        for place in (req.pickup, req.dropoff):
            if not place.name.startswith("home"):
                places[(place.x, place.y)] = place.name
    places[(data.depot.x, data.depot.y)] = data.depot.name
    for (x, y), name in places.items():
        ax.text(x + 0.3, y + 0.3, name, fontsize=8, weight="bold")
    ax.plot(data.depot.x, data.depot.y, "s", color="black", ms=10)
    ax.set_aspect("equal")
    ax.set_xlabel("km")
    ax.set_ylabel("km")
    ax.set_title(
        f"{data.name}: {len(plan.routes)} buses, {plan.distance / 1000:,.0f} km"
        "\n(o pickup, ^ drop-off, dotted arrow: request)"
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    return save(fig, path)


def plot_timeline(data: PdpData, plan: Plan, path: Path) -> Path:
    """People on board over the day, one lane per bus."""
    from examples.common.plotting import COLORS, new_figure, save

    nodes = data.stops()
    fig, ax = new_figure(9, 0.7 * len(plan.routes) + 1.5)
    for k, (route, times) in enumerate(zip(plan.routes, plan.times, strict=True)):
        color = COLORS[k % len(COLORS)]
        load, xs, ys = 0, [times[0] / 60], [0]
        for node, t in zip(route[1:-1], times[1:-1], strict=True):
            xs.append(t / 60)
            ys.append(load)
            load += nodes[node].load
            xs.append(t / 60)
            ys.append(load)
        xs.append(times[-1] / 60)
        ys.append(0)
        base = (len(plan.routes) - 1 - k) * (data.capacity + 2)  # Bus 1 on top.
        ax.fill_between(xs, base, [base + y for y in ys], color=color, alpha=0.7)
        ax.plot([xs[0], xs[-1]], [base, base], color="0.3", lw=1)  # On duty.
        ax.plot(
            [xs[0], xs[-1]], [base + data.capacity] * 2, color="0.6", lw=0.8, ls="--"
        )
    lanes = range(len(plan.routes))
    ax.set_yticks(
        [
            (len(plan.routes) - 1 - k) * (data.capacity + 2) + data.capacity / 2
            for k in lanes
        ],
        [f"bus {k + 1}" for k in lanes],
    )
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("time of day (h)")
    ax.set_title(
        "People on board over the day"
        f"\n(solid line: bus on duty; dashed line: all {data.capacity} seats taken)"
    )
    return save(fig, path)


def main() -> None:
    """Solve the three days, verify every plan, and report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plot", action="store_true", help="write PNG figures")
    args = parser.parse_args()

    day = transport_day()
    seats = sum(r.seats for r in day.requests)
    print(
        f"Normal day: {len(day.requests)} ride requests, {seats} passengers, "
        f"{day.num_vehicles} buses with {day.capacity} seats\n"
    )
    plan = solve_routing(day, time_limit=TIME_LIMIT)
    print(f"Cost: {plan.cost / 1000:,.1f} km-equivalent ({len(plan.routes)} buses)\n")
    print_buses(day, plan)
    errors = verify(day, plan, "normal day")

    print("\nThe price of the rules\n")
    free = replace(day, max_detour=None)
    free_plan = solve_routing(free, time_limit=TIME_LIMIT)
    lifo_plan = solve_routing(day, time_limit=TIME_LIMIT, policy="LIFO")
    rows = [
        summary_row("ride limit (+20 min)", day, plan),
        summary_row("no ride limit", free, free_plan),
        summary_row("ride limit + LIFO", day, lifo_plan),
    ]
    header = ["rules", "buses", "km", "cost (km-eq.)", "avg extra min", "max extra min"]
    print(format_table(header, rows))
    errors += verify(free, free_plan, "no ride limit")
    errors += verify(day, lifo_plan, "LIFO")
    errors += [f"LIFO: {e}" for e in lifo_errors(day, lifo_plan)]

    short = short_staffed_day()
    print(f"\nShort-staffed day: {short.num_vehicles} buses, a taxi costs 60 km-eq.\n")
    short_plan = solve_routing(short, time_limit=TIME_LIMIT)
    print(
        f"Cost: {short_plan.cost / 1000:,.1f} km-equivalent: "
        f"{short_plan.distance / 1000:,.1f} km driven, "
        f"{len(short_plan.declined)} taxis"
    )
    names = ", ".join(short.requests[r].name for r in short_plan.declined)
    print(f"Taxi rides: {names}")
    errors += verify(short, short_plan, "short-staffed day")

    if errors:
        print("\nVerification FAILED:", *errors, sep="\n  ")
    else:
        print(
            "\nVerification: every plan keeps each pair on one bus with the pickup"
            "\nfirst, never overfills a bus, meets all windows and ride limits, and"
            "\nits cost matches the map."
        )
    if args.plot:
        for path in (
            plot_map(day, plan, FIGURES / "routes.png"),
            plot_timeline(day, plan, FIGURES / "on_board.png"),
        ):
            print("Wrote", path)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
