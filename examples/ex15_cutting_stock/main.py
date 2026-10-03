"""Cut jumbo paper rolls into customer orders with as few rolls as possible.

Run from the repository root:

    uv run python -m examples.ex15_cutting_stock.main
    uv run python -m examples.ex15_cutting_stock.main --plot
    uv run python -m examples.ex15_cutting_stock.main --kantorovich
"""

import argparse
import math
import time
from pathlib import Path

from examples.common.report import format_table

from .check import (
    all_patterns,
    best_worth,
    dual_bound,
    material_bound,
    plan_errors,
    waste_mm,
)
from .data import CuttingData, paper_mill, rush_order
from .model import (
    CuttingPlan,
    LpResult,
    column_generation,
    first_fit_plan,
    kantorovich,
    round_down_and_repair,
    round_up,
    solve_pattern_ip,
)

FIGURES = Path(__file__).parent / "figures"


def describe(data: CuttingData, pattern) -> str:
    """Return a pattern as text, for example '2 x 1930 + 1 x 1150'."""
    parts = [f"{k} x {o.width}" for o, k in zip(data.orders, pattern, strict=True) if k]
    return " + ".join(parts)


def print_column_generation(data: CuttingData, lp: LpResult) -> None:
    """Print the convergence of column generation, a few rounds at a time."""
    n = len(lp.history)
    shown = sorted({0, 1, 2, 4, 8, 12, n - 2, n - 1} & set(range(n)))
    rows = [
        (k + 1, h.n_patterns, h.lp_value, h.lower_bound, h.best_worth)
        for k, h in enumerate(lp.history)
        if k in shown
    ]
    print(f"Column generation: {n} rounds, {len(lp.patterns)} patterns\n")
    print(
        format_table(
            ["round", "patterns", "LP value", "lower bound", "best worth"], rows
        )
    )


def print_plans(data: CuttingData, plans: list[CuttingPlan], bound: int) -> list[str]:
    """Print one row per integer plan. Return check errors."""
    rows, errors = [], []
    for plan in plans:
        waste = waste_mm(data, plan.cuts)
        share = waste / (plan.rolls * data.roll_width)
        extra = [
            sum(p[i] * n for p, n in plan.cuts.items()) - o.quantity
            for i, o in enumerate(data.orders)
        ]
        rows.append(
            (plan.method, plan.rolls, plan.rolls - bound, f"{share:.1%}", sum(extra))
        )
        errors += [f"{plan.method}: {e}" for e in plan_errors(data, plan.cuts)]
    print(
        format_table(
            ["method", "rolls", "above bound", "trim loss", "extra rolls"], rows
        )
    )
    return errors


def part_1(data: CuttingData) -> tuple[LpResult, CuttingPlan, list[str]]:
    """Solve the paper mill: column generation, bounds, and four integer plans."""
    print(f"Part 1: {data.name}, jumbo width {data.roll_width} mm, ", end="")
    print(f"{sum(data.demand)} rolls in {len(data.orders)} widths, ", end="")
    print(f"at most {data.max_pieces} pieces per jumbo\n")

    start = time.perf_counter()
    lp = column_generation(data)
    seconds = time.perf_counter() - start
    print_column_generation(data, lp)
    print(f"\nColumn generation took {seconds:.2f} s.")

    bound = math.ceil(lp.value - 1e-6)
    print("\nLower bounds on the number of jumbo rolls")
    rows = [
        (
            "total width / jumbo width",
            material_bound(data),
            math.ceil(material_bound(data)),
        ),
        ("pattern LP (column generation)", lp.value, bound),
    ]
    print(format_table(["bound", "value", "rounded up"], rows))

    plans = [
        first_fit_plan(data),
        round_up(lp),
        round_down_and_repair(data, lp),
        solve_pattern_ip(data, lp.patterns),
    ]
    print("\nWhole-number plans")
    errors = print_plans(data, plans, bound)
    # Fewest rolls first; among equals, the least trim loss.
    best = min(plans, key=lambda p: (p.rolls, waste_mm(data, p.cuts)))

    print(f"\nBest plan: {best.rolls} jumbo rolls ({best.method})")
    rows = [
        (
            n,
            describe(data, p),
            data.roll_width - sum(w * k for w, k in zip(data.widths, p, strict=True)),
        )
        for p, n in sorted(best.cuts.items(), key=lambda t: -t[1])
    ]
    print(format_table(["jumbos", "cut into (mm)", "trim (mm)"], rows))

    # The certificate: exact pricing finds no pattern worth more than one
    # jumbo, and the dual bound equals the LP value.
    worth = best_worth(data, lp.duals)
    if worth > 1 + 1e-6:
        errors.append(f"a pattern of worth {worth} > 1 exists: LP not optimal")
    if abs(dual_bound(data, lp.duals) - lp.value) > 1e-6 * lp.value:
        errors.append("dual bound does not match the LP value")
    if best.rolls == bound:
        print(f"\n{best.rolls} rolls = rounded-up LP bound, so the plan is optimal.")
    return lp, best, errors


def part_2(data: CuttingData) -> list[str]:
    """Solve a rush order where price-and-branch is not enough."""
    print(f"Part 2: {data.name}: ", end="")
    print(", ".join(f"{o.quantity} x {o.width} mm" for o in data.orders), "\n")
    lp = column_generation(data)
    bound = math.ceil(lp.value - 1e-6)
    pnb = solve_pattern_ip(data, lp.patterns)
    every = all_patterns(data)
    full = solve_pattern_ip(data, every)
    rows = [
        ("pattern LP bound", f"{lp.value:.2f}", f"{bound}"),
        (f"integer master, {len(lp.patterns)} generated patterns", "", f"{pnb.rolls}"),
        (f"integer master, all {len(every)} patterns", "", f"{full.rolls}"),
    ]
    print(format_table(["method", "LP", "rolls"], rows))
    if pnb.rolls > full.rolls:
        print(
            f"\nThe generated patterns allow only {pnb.rolls} rolls; "
            f"the best plan needs {full.rolls}."
        )
    errors = [
        f"{p.method}: {e}" for p in (pnb, full) for e in plan_errors(data, p.cuts)
    ]
    if full.rolls != bound:
        errors.append("the full pattern IP does not reach the LP bound")
    return errors


def compare_kantorovich(data: CuttingData, time_limit: float = 10.0) -> None:
    """Solve the direct assignment model, for comparison."""
    upper = first_fit_plan(data).rolls
    lp_value, _ = kantorovich(data, upper, relax=True)
    start = time.perf_counter()
    value, optimal = kantorovich(data, upper, time_limit=time_limit)
    seconds = time.perf_counter() - start
    print(
        f"\nKantorovich model ({upper} candidate jumbos, HiGHS, {time_limit:g} s limit)"
    )
    print(f"  LP bound {lp_value:.2f}; best plan {value:.0f} rolls; ", end="")
    print(f"proven optimal: {optimal}; {seconds:.1f} s")


def plot_patterns(data: CuttingData, plan: CuttingPlan) -> Path:
    """Draw each pattern of the plan as a bar split into narrow rolls."""
    from examples.common.plotting import new_figure, save

    cuts = sorted(plan.cuts.items(), key=lambda t: -t[1])
    fig, ax = new_figure(8, 0.45 * len(cuts) + 1.4)
    # Shade by width: wide rolls dark, narrow rolls light. Ten widths need
    # more colors than the house palette has, and a ramp is easy to read.
    import matplotlib

    ramp = matplotlib.colormaps["viridis"]
    lo, hi = min(data.widths), max(data.widths)
    color = {w: ramp(0.85 - 0.8 * (w - lo) / (hi - lo)) for w in data.widths}
    for row, (pattern, n) in enumerate(cuts):
        left = 0
        for o, k in zip(data.orders, pattern, strict=True):
            for _ in range(k):
                ax.barh(
                    row, o.width, left=left, color=color[o.width], ec="white", lw=1.5
                )
                r, g, b, _ = color[o.width]
                dark = 0.299 * r + 0.587 * g + 0.114 * b < 0.5
                ax.text(
                    left + o.width / 2,
                    row,
                    str(o.width),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if dark else "black",
                )
                left += o.width
        if left < data.roll_width:
            ax.barh(
                row,
                data.roll_width - left,
                left=left,
                color="0.9",
                hatch="///",
                ec="0.7",
            )
        ax.text(data.roll_width + 60, row, f"x {n}", va="center")
    ax.set_yticks([])
    ax.set_xlim(0, data.roll_width + 400)
    ax.invert_yaxis()
    ax.grid(False)
    ax.set_xlabel("position across the jumbo roll (mm); hatched = trim loss")
    ax.set_title(
        f"{plan.rolls} jumbo rolls: each bar is one pattern, x n = jumbos cut this way"
    )
    return save(fig, FIGURES / "patterns.png")


def plot_convergence(data: CuttingData, lp: LpResult, best: CuttingPlan) -> Path:
    """LP value and lower bound per round of column generation."""
    from examples.common.plotting import COLORS, new_figure, save

    rounds = range(1, len(lp.history) + 1)
    fig, ax = new_figure(7, 4)
    ax.plot(
        rounds,
        [h.lp_value for h in lp.history],
        "o-",
        color=COLORS[0],
        ms=4,
        label="restricted master LP (upper)",
    )
    ax.plot(
        rounds,
        [h.lower_bound for h in lp.history],
        "s-",
        color=COLORS[1],
        ms=4,
        label="Farley lower bound",
    )
    ax.axhline(
        best.rolls,
        color=COLORS[2],
        ls="--",
        lw=1.5,
        label=f"best whole-number plan: {best.rolls}",
    )
    ax.axhline(
        material_bound(data),
        color="0.5",
        ls=":",
        lw=1.5,
        label=f"total width bound: {material_bound(data):.1f}",
    )
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.set_xlabel("round of column generation")
    ax.set_ylabel("jumbo rolls")
    ax.set_title(f"The LP value and its lower bound meet at {lp.value:.2f}")
    ax.legend(loc="lower right")
    return save(fig, FIGURES / "convergence.png")


def main() -> None:
    """Parse arguments, solve both parts, verify, and report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plot", action="store_true", help="write PNG figures")
    parser.add_argument(
        "--kantorovich", action="store_true", help="also solve the direct model (slow)"
    )
    args = parser.parse_args()

    mill = paper_mill()
    lp, best, errors = part_1(mill)
    if args.kantorovich:
        compare_kantorovich(mill)
    print("\n" + "=" * 72 + "\n")
    errors += part_2(rush_order())

    if errors:
        print("\nVerification FAILED:", *errors, sep="\n  ")
    else:
        print(
            "\nVerification: every plan fits and meets all orders. Exact pricing"
            "\nfinds no better pattern, so the LP is optimal, and both best plans"
            "\nreach the rounded-up LP bound, so they are optimal too."
        )
    if args.plot:
        for path in (plot_patterns(mill, best), plot_convergence(mill, lp, best)):
            print("Wrote", path)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
