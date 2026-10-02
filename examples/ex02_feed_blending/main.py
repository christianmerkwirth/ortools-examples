"""Blend a least-cost chicken feed, then diagnose an impossible spec.

Run from the repository root:

    uv run python -m examples.ex02_feed_blending.main
    uv run python -m examples.ex02_feed_blending.main --plot
"""

import argparse
from pathlib import Path

from examples.common.report import format_table

from .check import feasibility_errors, optimality_gap, proves_infeasible
from .data import BlendData, broiler_grower, premium_finisher
from .model import (
    Blend,
    InfeasibleSpecError,
    elastic_blend,
    find_conflict,
    frontier,
    solve_blend,
)

FIGURES = Path(__file__).parent / "figures"


def print_blend(data: BlendData, blend: Blend) -> list[str]:
    """Print the recipe and the nutrient levels. Return check errors."""
    print(f"Cheapest '{data.name}' blend: ${blend.cost:,.2f} per tonne\n")
    rows = [
        (i.name, 100 * blend.share[i.name], f"{100 * i.max_share:g}", i.cost)
        for i in data.ingredients
        if blend.share[i.name] > 1e-9
    ]
    print(format_table(["ingredient", "% of blend", "max %", "$/kg"], rows))

    print()
    rows = []
    for r in data.requirements:
        y = blend.nutrient_price[r.nutrient]
        binding = "min" if y > 1e-9 else "max" if y < -1e-9 else ""
        rows.append(
            (
                r.nutrient,
                r.unit,
                blend.level[r.nutrient],
                f"{r.min:g}-{r.max:g}",
                binding,
                y,
            )
        )
    print(
        format_table(
            ["nutrient", "unit", "level", "allowed", "binding", "$/unit"], rows
        )
    )
    print(
        "\n$/unit is the shadow price: the change in batch cost if the binding"
        "\nbound moves by one unit (for example one percentage point of protein)."
    )

    errors = feasibility_errors(data, blend.share)
    gap = optimality_gap(data, blend.share, blend.total_price, blend.nutrient_price)
    if abs(gap) > 1e-6 * blend.cost:
        errors.append(f"cost is ${gap:.6f} above the dual bound")
    return errors


def diagnose(data: BlendData) -> list[str]:
    """Explain why a spec is infeasible and how to fix it. Return check errors."""
    try:
        solve_blend(data)
    except InfeasibleSpecError as e:
        print(f"{e}\n")
    else:
        return ["expected an infeasible spec"]

    conflict = find_conflict(data)
    print(
        "A minimal conflict (these limits alone cannot all hold; any smaller set can):"
    )
    for limit in conflict:
        print(f"  * {limit}")

    elastic = elastic_blend(data)
    print("\nSmallest change to the nutrient spec that makes it feasible:")
    for r in elastic.relaxations:
        unit = data.requirement(r.limit.name).unit
        op = ">=" if r.limit.side == "min" else "<="
        print(
            f"  * {r.limit} {unit}  ->  {r.limit.name} {op} {r.new_value:,.1f} {unit}"
            f"  ({r.change:+.1%})"
        )

    errors = []
    if not proves_infeasible(data, elastic.total_price, elastic.nutrient_price):
        errors.append("elastic duals do not prove infeasibility")
    # The elastic blend must meet the relaxed spec.
    relaxed = data
    for r in elastic.relaxations:
        relaxed = relaxed.with_requirement(r.limit.name, **{r.limit.side: r.new_value})
    errors += feasibility_errors(relaxed, elastic.blend_share)
    return errors


def plot_blend(data: BlendData, blend: Blend) -> Path:
    """Two panels: the recipe, and where each nutrient sits in its range."""
    from examples.common.plotting import COLORS, new_figure, save

    fig, (left, right) = new_figure(10, 3.6, ncols=2, width_ratios=[1, 1.2])

    used = [i for i in data.ingredients if blend.share[i.name] > 1e-9]
    left.barh(
        [i.name for i in used],
        [100 * blend.share[i.name] for i in used],
        color=COLORS[0],
    )
    left.invert_yaxis()
    left.set_xlabel("% of blend")
    left.set_title(f"Recipe: ${blend.cost:,.0f} per tonne")

    # Show each nutrient's level as a position inside its allowed range.
    for k, r in enumerate(data.requirements):
        pos = (blend.level[r.nutrient] - r.min) / (r.max - r.min)
        right.plot([0, 1], [k, k], color="0.8", lw=8, solid_capstyle="butt")
        at_bound = pos < 1e-6 or pos > 1 - 1e-6
        right.plot(pos, k, "o", color=COLORS[3] if at_bound else COLORS[2], ms=9)
        right.text(1.04, k, f"{blend.level[r.nutrient]:,.2f} {r.unit}", va="center")
    right.set_yticks(
        range(len(data.requirements)), [r.nutrient for r in data.requirements]
    )
    right.set_xticks([0, 1], ["min", "max"])
    right.set_xlim(-0.1, 1.45)
    right.invert_yaxis()
    right.grid(False)
    right.set_title("Nutrient levels (red = at a bound)")
    return save(fig, FIGURES / "blend.png")


def plot_frontier(feasible: BlendData, premium: BlendData) -> Path:
    """Plot the protein-energy trade-off, with both specs drawn in."""
    from matplotlib.patches import Rectangle

    from examples.common.plotting import COLORS, new_figure, save

    xs = [16 + 0.25 * k for k in range(57)]
    points = [(x, y) for x, y in frontier(feasible, "protein", "energy", xs) if y]
    fig, ax = new_figure(7, 4.2)
    ax.plot([x for x, _ in points], [y for _, y in points], color=COLORS[0], lw=2)
    ax.fill_between(
        [x for x, _ in points],
        [y for _, y in points],
        2800,
        color=COLORS[0],
        alpha=0.08,
    )
    ax.text(17, 3020, "reachable", color=COLORS[0])
    for data, color, label in (
        (feasible, COLORS[2], "grower spec"),
        (premium, COLORS[3], "premium spec"),
    ):
        p, e = data.requirement("protein"), data.requirement("energy")
        ax.add_patch(
            Rectangle(
                (p.min, e.min),
                p.max - p.min,
                e.max - e.min,
                fill=False,
                ec=color,
                lw=2,
                label=label,
            )
        )
    ax.set_xlim(16, 30)
    ax.set_ylim(2850, 3350)
    ax.set_xlabel("protein (% of blend)")
    ax.set_ylabel("energy (kcal/kg)")
    ax.set_title("More protein costs energy: the premium spec is out of reach")
    ax.legend(loc="upper right")
    return save(fig, FIGURES / "protein_energy_frontier.png")


def main() -> None:
    """Solve the feasible spec, diagnose the infeasible one, and verify both."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plot", action="store_true", help="write PNG figures")
    args = parser.parse_args()

    grower, premium = broiler_grower(), premium_finisher()
    blend = solve_blend(grower)
    errors = print_blend(grower, blend)
    print("\n" + "=" * 72 + "\n")
    errors += diagnose(premium)

    if errors:
        print("\nVerification FAILED:", *errors, sep="\n  ")
    else:
        print(
            "\nVerification: the grower blend is feasible and provably optimal;"
            "\nthe premium spec is provably infeasible."
        )
    if args.plot:
        for path in (plot_blend(grower, blend), plot_frontier(grower, premium)):
            print("Wrote", path)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
