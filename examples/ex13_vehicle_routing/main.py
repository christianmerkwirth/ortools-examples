"""Plan pharmacy deliveries with vans, capacities, and time windows.

Run from the repository root:

    uv run python -m examples.ex13_vehicle_routing.main
    uv run python -m examples.ex13_vehicle_routing.main --plot
    uv run python -m examples.ex13_vehicle_routing.main --exact   # CP-SAT bound
"""

import argparse
import itertools
from pathlib import Path

from examples.common.report import format_table

from .check import plan_errors, simple_lower_bound
from .data import (
    WINDOWS,
    VrpData,
    clock,
    distance_matrix,
    flu_season_day,
    paid_time_day,
    pharmacy_day,
    travel_minutes,
)
from .model import Plan, solve_exact, solve_routing

FIGURES = Path(__file__).parent / "figures"


def km(meters: float) -> float:
    """Convert meters to km."""
    return meters / 1000


def route_rows(data: VrpData, plan: Plan) -> list[tuple]:
    """Return one table row per van."""
    drive, dist = travel_minutes(data), distance_matrix(data)
    rows = []
    for k, (route, times) in enumerate(zip(plan.routes, plan.start_times, strict=True)):
        stops = route[1:-1]
        # Waiting = time between arriving and starting service.
        wait = sum(
            tb - (ta + data.stops[a].service + drive[a][b])
            for (a, ta), (b, tb) in itertools.pairwise(zip(route, times, strict=True))
        )
        meters = sum(dist[a][b] for a, b in itertools.pairwise(route))
        rows.append(
            (
                f"van {k + 1}",
                len(stops),
                f"{sum(data.stops[c].demand for c in stops)} / {data.capacity}",
                km(meters),
                clock(times[0]),
                clock(times[-1]),
                wait,
            )
        )
    return rows


def print_plan(data: VrpData, plan: Plan) -> None:
    """Print the routes, the cost split, and any unserved customers."""
    header = ["van", "stops", "crates", "km", "leaves", "back", "wait min"]
    print(format_table(header, route_rows(data, plan)))
    penalty = sum(data.drop_penalty(c) or 0 for c in plan.dropped)
    working = sum(times[-1] - times[0] for times in plan.start_times)
    paid = (
        f" + {working:,} working min x {km(data.minute_cost):g} km"
        if data.minute_cost
        else ""
    )
    print(
        f"\nCost: {len(plan.routes)} vans x {km(data.fixed_cost):g} km"
        f" + {km(plan.distance):,.1f} km driven"
        + (f" + {km(penalty):,.0f} km penalties" if penalty else "")
        + paid
        + f" = {km(plan.cost):,.1f} km-equivalent"
    )
    print(f"Total working time of all vans: {working:,} min ({working / 60:.1f} h)")
    if plan.dropped:
        crates = sum(data.stops[c].demand for c in plan.dropped)
        names = ", ".join(
            f"{data.stops[c].name} ({data.stops[c].demand})" for c in plan.dropped
        )
        print(
            f"Unserved today: {len(plan.dropped)} pharmacies, {crates} crates: {names}"
        )


def plot_maps(normal: VrpData, normal_plan: Plan, flu: VrpData, flu_plan: Plan) -> Path:
    """Route maps for both days, one color per van."""
    from examples.common.plotting import COLORS, new_figure, save

    fig, axes = new_figure(11, 5.4, ncols=2, sharex=True, sharey=True)
    palette = COLORS[:7] + ["#999999", "#882255", "#44AA99"]
    for ax, data, plan, title in (
        (axes[0], normal, normal_plan, "Normal day"),
        (axes[1], flu, flu_plan, "Flu season day"),
    ):
        for k, route in enumerate(plan.routes):
            xs = [data.stops[s].x for s in route]
            ys = [data.stops[s].y for s in route]
            ax.plot(xs, ys, "-o", color=palette[k % len(palette)], lw=1.6, ms=4)
        if plan.dropped:
            ax.plot(
                [data.stops[c].x for c in plan.dropped],
                [data.stops[c].y for c in plan.dropped],
                "x",
                color="black",
                ms=8,
                mew=2,
                label="unserved",
            )
            ax.legend(loc="lower left")
        ax.plot(0, 0, "s", color="black", ms=10)
        ax.annotate("depot", (0, 0), xytext=(6, -12), textcoords="offset points")
        ax.set_title(
            f"{title}: {len(plan.routes)} vans, {km(plan.distance):,.0f} km"
            + (f", {len(plan.dropped)} unserved" if plan.dropped else "")
        )
        ax.set_xlabel("km")
        ax.set_aspect("equal")
    axes[0].set_ylabel("km")
    return save(fig, FIGURES / "routes.png")


def plot_timeline(panels: list[tuple[VrpData, Plan, str]]) -> Path:
    """For each van: unloading (colored by window type) and waiting."""
    from matplotlib.patches import Patch

    from examples.common.plotting import COLORS, new_figure, save

    kind_of = {window: name for name, window in WINDOWS.items()}
    color = {"early": COLORS[0], "midday": COLORS[2], "late": COLORS[4], "any": "0.45"}
    heights = [len(plan.routes) for _, plan, _ in panels]
    fig, axes = new_figure(
        11,
        0.5 * sum(heights) + 2.6,
        nrows=len(panels),
        sharex=True,
        height_ratios=heights,
    )
    for ax, (data, plan, title) in zip(axes, panels, strict=True):
        drive = travel_minutes(data)
        for y, (route, times) in enumerate(
            zip(plan.routes, plan.start_times, strict=True)
        ):
            # Thin gray bar: the van is out on the road.
            ax.barh(y, times[-1] - times[0], left=times[0], height=0.12, color="0.8")
            prev, prev_done = route[0], times[0]
            for stop, start in zip(route[1:-1], times[1:-1], strict=True):
                s = data.stops[stop]
                arrive = prev_done + drive[prev][stop]
                if start > arrive:  # The van waits for the window to open.
                    ax.barh(
                        y,
                        start - arrive,
                        left=arrive,
                        height=0.5,
                        color=COLORS[1],
                        alpha=0.35,
                    )
                kind = kind_of[(s.open, s.close)]
                ax.barh(y, s.service, left=start, height=0.5, color=color[kind])
                prev, prev_done = stop, start + s.service
        ax.set_yticks(
            range(len(plan.routes)), [f"van {k + 1}" for k in range(len(plan.routes))]
        )
        ax.invert_yaxis()
        working = sum(t[-1] - t[0] for t in plan.start_times)
        ax.set_title(f"{title}: {working / 60:.1f} working hours in total", loc="left")
    data = panels[0][0]
    ticks = range(data.shift_start, data.shift_end + 1, 60)
    axes[-1].set_xticks(list(ticks), [clock(t) for t in ticks])
    axes[-1].set_xlim(data.shift_start - 10, data.shift_end + 10)
    handles = [
        Patch(color=color[k], label=f"unload, {k} window ({clock(o)}-{clock(c)})")
        for k, (o, c) in WINDOWS.items()
    ] + [Patch(color=COLORS[1], alpha=0.35, label="wait for a window")]
    axes[-1].legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        ncols=3,
        frameon=False,
    )
    return save(fig, FIGURES / "timeline.png")


def main() -> None:
    """Solve all three scenarios, verify the plans, and report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plot", action="store_true", help="write PNG figures")
    parser.add_argument("--exact", action="store_true", help="also run CP-SAT (60 s)")
    parser.add_argument("--time-limit", type=float, default=4.0, help="s per solve")
    args = parser.parse_args()

    errors = []
    normal, paid, flu = pharmacy_day(), paid_time_day(), flu_season_day()

    crates = sum(s.demand for s in normal.stops)
    print(
        f"Normal day: {len(normal.customers)} pharmacies, {crates} crates,"
        f" {normal.num_vehicles} vans of {normal.capacity} crates\n"
    )
    plan = solve_routing(normal, time_limit=args.time_limit)
    print_plan(normal, plan)
    errors += [f"normal day: {e}" for e in plan_errors(normal, plan)]
    vans, bound = simple_lower_bound(normal)
    print(
        f"\nSimple lower bound (check.py): at least {vans} vans and"
        f" {km(bound):,.1f} km-equivalent; the plan is at most"
        f" {100 * (plan.cost - bound) / bound:.0f}% above it."
    )
    if args.exact:
        exact = solve_exact(normal, time_limit=60)
        errors += [f"CP-SAT plan: {e}" for e in plan_errors(normal, exact)]
        print(
            f"CP-SAT, 60 s: best plan {km(exact.cost):,.1f}, proven lower bound"
            f" {km(exact.lower_bound):,.1f} km-equivalent. The routing plan is at"
            f" most {100 * (plan.cost - exact.lower_bound) / exact.lower_bound:.1f}%"
            " above the optimum."
        )

    print("\n" + "=" * 72 + "\n")
    print(
        f"Same day, but each working minute now costs {km(paid.minute_cost):g} km"
        " (driver wages).\n"
    )
    paid_plan = solve_routing(paid, time_limit=args.time_limit)
    print_plan(paid, paid_plan)
    errors += [f"paid time: {e}" for e in plan_errors(paid, paid_plan)]

    print("\n" + "=" * 72 + "\n")
    crates = sum(s.demand for s in flu.stops)
    print(
        f"Flu season day: {crates} crates, but {flu.num_vehicles} vans carry at most"
        f" {flu.num_vehicles * flu.capacity}. Dropping a pharmacy costs"
        f" {km(flu.drop_penalty_per_crate):g} km per crate.\n"
    )
    flu_plan = solve_routing(flu, time_limit=args.time_limit)
    print_plan(flu, flu_plan)
    errors += [f"flu day: {e}" for e in plan_errors(flu, flu_plan)]

    if errors:
        print("\nVerification FAILED:", *errors, sep="\n  ")
    else:
        print(
            "\nVerification: a replay of every route confirms capacities,"
            "\ntime windows, and shifts, and the costs match the map."
        )
    if args.plot:
        print("Wrote", plot_maps(normal, plan, flu, flu_plan))
        panels = [(normal, plan, "Free time"), (paid, paid_plan, "Paid time")]
        print("Wrote", plot_timeline(panels))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
