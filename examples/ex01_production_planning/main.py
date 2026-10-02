"""Solve the furniture workshop LP and print a report.

Run from the repository root:

    uv run python -m examples.ex01_production_planning.main
    uv run python -m examples.ex01_production_planning.main --plot
"""

import argparse
from pathlib import Path

from examples.common.report import format_table

from .check import feasibility_errors, optimality_errors
from .data import ProductionData, furniture_workshop
from .model import ProductionPlan, profit_curve, solve_production_plan

FIGURES = Path(__file__).parent / "figures"


def print_report(data: ProductionData, plan: ProductionPlan) -> None:
    """Print the plan, resource use, and sensitivity information."""
    print(f"Optimal weekly profit: ${plan.profit:,.2f}\n")

    print("Production plan")
    rows = [
        (
            p.name,
            plan.quantity[p.name],
            f"{p.min_order:g}-{p.max_demand:g}",
            float(p.profit),
            plan.reduced_cost[p.name],
        )
        for p in data.products
    ]
    print(format_table(["product", "units", "allowed", "$/unit", "reduced cost"], rows))

    print("\nResources")
    rows = [
        (
            r.name,
            r.unit,
            plan.resource_used[r.name],
            float(r.capacity),
            r.capacity - plan.resource_used[r.name],
            plan.shadow_price[r.name],
        )
        for r in data.resources
    ]
    print(
        format_table(
            ["resource", "unit", "used", "capacity", "slack", "shadow price"], rows
        )
    )

    print("\nWhat the numbers say")
    for r in data.resources:
        y = plan.shadow_price[r.name]
        if y > 1e-9:
            print(f"* One more {r.unit} of {r.name} is worth ${y:,.2f}.")
        else:
            print(f"* {r.name.capitalize()} has slack; more of it is worth nothing.")
    for p in data.products:
        d = plan.reduced_cost[p.name]
        if d < -1e-9:
            print(
                f"* {p.name.capitalize()}s earn too little: their unit profit must "
                f"rise by more than ${-d:,.2f} before making more of them pays."
            )
        elif d > 1e-9:
            print(
                f"* {p.name.capitalize()}s are limited by demand: each extra "
                f"unit the market would buy adds ${d:,.2f}."
            )


def plot_resources(data: ProductionData, plan: ProductionPlan) -> Path:
    """Bar chart: share of each resource used, labeled with shadow prices."""
    from examples.common.plotting import COLORS, new_figure, save

    fig, ax = new_figure(6, 3)
    names = [r.name for r in data.resources]
    share = [100 * plan.resource_used[r.name] / r.capacity for r in data.resources]
    bars = ax.barh(names, share, color=COLORS[0])
    ax.axvline(100, color="black", lw=1)
    for bar, r in zip(bars, data.resources, strict=True):
        label = f"  shadow price ${plan.shadow_price[r.name]:,.2f}/{r.unit}"
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, label, va="center")
    ax.set_xlim(0, 160)
    ax.set_xlabel("capacity used (%)")
    ax.set_title("Resource use in the optimal plan")
    ax.invert_yaxis()
    return save(fig, FIGURES / "resource_use.png")


def plot_profit_curve(data: ProductionData, resource: str) -> Path:
    """Profit and shadow price as one resource's capacity changes."""
    from examples.common.plotting import COLORS, new_figure, save

    base = data.resource(resource).capacity
    caps = [base * k / 200 for k in range(0, 401)]
    points = [
        (c, p, y) for c, p, y in profit_curve(data, resource, caps) if p is not None
    ]
    xs = [c for c, _, _ in points]

    fig, (top, bottom) = new_figure(7, 5, nrows=2, sharex=True, height_ratios=[2, 1])
    top.plot(xs, [p for _, p, _ in points], color=COLORS[0], lw=2)
    top.axvline(base, color=COLORS[3], ls="--", lw=1, label="current capacity")
    top.set_ylabel("max profit ($)")
    top.set_title(f"What is more {resource} capacity worth?")
    top.legend(loc="lower right")
    bottom.step(xs, [y for _, _, y in points], where="post", color=COLORS[1], lw=2)
    bottom.axvline(base, color=COLORS[3], ls="--", lw=1)
    bottom.set_ylabel(f"shadow price\n($/{data.resource(resource).unit})")
    bottom.set_xlabel(f"{resource} capacity ({data.resource(resource).unit}s per week)")
    return save(fig, FIGURES / f"profit_vs_{resource}.png")


def main() -> None:
    """Parse arguments, solve, verify, and report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plot", action="store_true", help="write PNG figures")
    args = parser.parse_args()

    data = furniture_workshop()
    plan = solve_production_plan(data)
    print_report(data, plan)

    errors = feasibility_errors(data, plan) + optimality_errors(data, plan)
    if errors:
        print("\nVerification FAILED:", *errors, sep="\n  ")
    else:
        print("\nVerification: the plan is feasible and provably optimal.")

    if args.plot:
        for path in (plot_resources(data, plan), plot_profit_curve(data, "carpentry")):
            print("Wrote", path)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
