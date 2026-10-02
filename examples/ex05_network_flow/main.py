"""Find the bottleneck of a supply network, then plan the cheapest flow.

Run from the repository root:

    uv run python -m examples.ex05_network_flow.main
    uv run python -m examples.ex05_network_flow.main --plot
"""

import argparse
import math
from pathlib import Path

from examples.common.report import format_table

from .check import (
    cut_capacity,
    flow_errors,
    has_negative_cycle,
    marginal_costs,
)
from .data import NetworkData, water_network
from .model import (
    SINK,
    SOURCE,
    Flow,
    MaxFlowResult,
    solve_max_flow,
    solve_min_cost_flow,
    upgrade_options,
)

FIGURES = Path(__file__).parent / "figures"
EXTRA = 60  # Pallets per week a lane upgrade adds.


def print_max_flow(data: NetworkData, result: MaxFlowResult) -> list[str]:
    """Print Part 1: the max flow and the bottleneck. Return check errors."""
    flow = result.flow
    short = data.total_demand - flow.value
    print("Part 1 - How much can the network deliver?\n")
    print(f"Stores want {data.total_demand:,} pallets per week.")
    print(f"The network can deliver at most {flow.value:,}. Short: {short:,}.\n")

    print("Minimum cut (the arcs that limit the flow):")
    rows = [(a.kind, str(a), a.capacity) for a in result.cut]
    print(format_table(["kind", "arc", "capacity"], rows))
    print(f"Cut capacity: {sum(a.capacity for a in result.cut):,} pallets")

    errors = flow_errors(data, flow.arc_flow, flow.value, flow.cost)
    if cut_capacity(data, set(result.source_side)) != flow.value:
        errors.append("flow value differs from the cut capacity")
    return errors


def print_upgrades(data: NetworkData) -> None:
    """Print what each bottleneck lane upgrade would deliver and cost."""
    print(f"\nWhat if we add {EXTRA} pallets per week to one lane in the cut?")
    rows = [
        (f"{u.lane[0]} -> {u.lane[1]}", u.delivered, u.cost)
        for u in upgrade_options(data, EXTRA)
    ]
    print(format_table(["upgraded lane", "delivered", "weekly cost $"], rows))


def print_plan(data: NetworkData, plan: Flow) -> list[str]:
    """Print Part 2: the cheapest plan. Return check errors."""
    print(f"\nPart 2 - Cheapest plan after the upgrade: ${plan.cost:,} per week\n")
    rows = [
        (p.name, plan.arc_flow[SOURCE, p.name], p.capacity, p.unit_cost)
        for p in data.plants
    ]
    print(format_table(["plant", "made", "capacity", "$/pallet"], rows))
    print()
    rows = [
        (
            f"{x.origin} -> {x.destination}",
            plan.arc_flow[x.key],
            x.capacity,
            x.unit_cost,
        )
        for x in data.lanes
        if plan.arc_flow[x.key] > 0
    ]
    print(format_table(["lane", "pallets", "capacity", "$/pallet"], rows))

    errors = flow_errors(data, plan.arc_flow, plan.value, plan.cost)
    if plan.value != data.total_demand:
        errors.append("not all demand is met")
    if has_negative_cycle(data, plan.arc_flow):
        errors.append("a negative cycle exists: the plan is not the cheapest")

    print("\nCost of one more pallet at each store (node potentials):")
    price = marginal_costs(data, plan.arc_flow)
    rows = [
        (
            s.name,
            f"${price[s.name]:,.0f}" if math.isfinite(price[s.name]) else "no route",
        )
        for s in data.stores
    ]
    print(format_table(["store", "marginal cost"], rows))
    return errors


# ---------------------------------------------------------------- plots ---

# Fixed drawing positions: plants left, DCs middle, stores right.
POSITIONS = {
    "north plant": (0, 3.6),
    "south plant": (0, 1.2),
    "west DC": (1, 4.4),
    "central DC": (1, 2.6),
    "east DC": (1, 1.2),
    "Alder": (2, 5.0),
    "Birch": (2, 4.0),
    "Cedar": (2, 3.0),
    "Dunmore": (2, 2.0),
    "Elm": (2, 1.0),
    "Fairview": (2, 0.0),
}


def _draw_network(ax, data, flow: Flow, highlight=(), shaded=frozenset(), notes=None):
    """Draw lanes with width by flow; highlight some lanes in red."""
    from matplotlib.patches import FancyArrowPatch

    from examples.common.plotting import COLORS

    for x in data.lanes:
        f = flow.arc_flow[x.key]
        (x0, y0), (x1, y1) = POSITIONS[x.origin], POSITIONS[x.destination]
        # DC-to-DC transfers share a column, so bend them to the side.
        bend = "arc3,rad=-0.5" if x0 == x1 else "arc3,rad=0"
        if x1 - x0 == 2:
            bend = "arc3,rad=0.15"  # Plant-to-store lanes skip the DCs.
        red = x.key in highlight
        color = COLORS[3] if red else (COLORS[0] if f > 0 else "0.75")
        ax.add_patch(
            FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                connectionstyle=bend,
                arrowstyle="-|>",
                mutation_scale=10,
                lw=0.8 + 5 * f / 500,
                color=color,
                ls="--" if f == 0 else "-",
                shrinkA=24,
                shrinkB=24,
                zorder=1,
            )
        )
        # Label near the arc start; the shift avoids crowded midpoints.
        t = 0.32 if x0 != x1 else 0.5
        lx, ly = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
        if x0 == x1:
            lx += 0.22
        if x1 - x0 == 2:  # Follow the bend of the curved lane.
            lx, ly = 1.0, 0.25
        ax.text(
            lx,
            ly,
            f"{f}/{x.capacity}",
            fontsize=7,
            ha="center",
            va="center",
            color=COLORS[3] if red else "0.2",
            bbox={
                "boxstyle": "round,pad=0.15",
                "fc": "white",
                "ec": "none",
                "alpha": 0.9,
            },
            zorder=3,
        )

    for name, (px, py) in POSITIONS.items():
        fc = "#bcd7f0" if name in shaded else "white"
        ax.text(
            px,
            py,
            name,
            ha="center",
            va="center",
            fontsize=8,
            zorder=4,
            bbox={"boxstyle": "round,pad=0.35", "fc": fc, "ec": "0.3"},
        )
        if notes and name in notes:
            if px == 0:  # Plant notes go below the node.
                x, y, ha, va = px, py - 0.3, "center", "top"
            else:  # Store notes go to the right.
                x, y, ha, va = px + 0.13, py, "left", "center"
            ax.text(x, y, notes[name], fontsize=7, ha=ha, va=va, color="0.3")
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(-0.6, 5.4)
    ax.axis("off")


def plot_max_flow(data: NetworkData, result: MaxFlowResult) -> Path:
    """Draw the max flow with the minimum cut in red."""
    from examples.common.plotting import new_figure, save

    fig, ax = new_figure(8, 5.5)
    cut = {(a.tail, a.head) for a in result.cut if a.kind == "lane"}
    notes = {
        p.name: f"makes {result.flow.arc_flow[SOURCE, p.name]}/{p.capacity}"
        for p in data.plants
    }
    notes |= {
        s.name: f"gets {result.flow.arc_flow[s.name, SINK]}/{s.demand}"
        for s in data.stores
    }
    _draw_network(ax, data, result.flow, cut, result.source_side, notes)
    ax.set_title(
        f"Max flow: {result.flow.value:,} of {data.total_demand:,} pallets per week\n"
        "red = lanes in the minimum cut; shaded = source side of the cut",
        fontsize=10,
    )
    return save(fig, FIGURES / "max_flow_min_cut.png")


def plot_plan(data: NetworkData, plan: Flow, upgraded: tuple[str, str]) -> Path:
    """Draw the cheapest plan after the upgrade."""
    from examples.common.plotting import new_figure, save

    price = marginal_costs(data, plan.arc_flow)
    fig, ax = new_figure(8, 5.5)
    notes = {
        p.name: f"makes {plan.arc_flow[SOURCE, p.name]}/{p.capacity}"
        f"\n${p.unit_cost}/pallet"
        for p in data.plants
    }
    for s in data.stores:
        mc = (
            f"next pallet ${price[s.name]:.0f}"
            if math.isfinite(price[s.name])
            else "no spare route"
        )
        notes[s.name] = f"gets {plan.arc_flow[s.name, SINK]}/{s.demand}\n{mc}"
    _draw_network(ax, data, plan, {upgraded}, notes=notes)
    ax.set_title(
        f"Cheapest plan after the upgrade: ${plan.cost:,} per week\n"
        "red = upgraded lane; labels = pallets / capacity",
        fontsize=10,
    )
    return save(fig, FIGURES / "min_cost_plan.png")


def main() -> None:
    """Solve both parts, verify them, and report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plot", action="store_true", help="write PNG figures")
    args = parser.parse_args()

    data = water_network()
    result = solve_max_flow(data)
    errors = print_max_flow(data, result)
    print_upgrades(data)

    # Upgrade the option with the lowest cost that meets all demand.
    best = min(
        (u for u in upgrade_options(data, EXTRA) if u.delivered == data.total_demand),
        key=lambda u: u.cost,
    )
    upgraded = data.with_lane_capacity(
        *best.lane, data.lane(*best.lane).capacity + EXTRA
    )
    print(f"\nBest choice: upgrade {best.lane[0]} -> {best.lane[1]}.")
    plan = solve_min_cost_flow(upgraded)
    errors += print_plan(upgraded, plan)

    if errors:
        print("\nVerification FAILED:", *errors, sep="\n  ")
    else:
        print(
            "\nVerification: the max flow equals the cut capacity (so both are"
            "\noptimal), and the plan has no negative cycle (so it is the cheapest)."
        )
    if args.plot:
        for path in (plot_max_flow(data, result), plot_plan(upgraded, plan, best.lane)):
            print("Wrote", path)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
