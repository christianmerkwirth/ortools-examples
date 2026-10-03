"""Pack crates into vans (1D), then cut sign panels from a roll (2D).

Run from the repository root:

    uv run python -m examples.ex14_packing.main
    uv run python -m examples.ex14_packing.main --plot
    uv run python -m examples.ex14_packing.main --compare-grid
"""

import argparse
from pathlib import Path

from examples.common.report import format_table

from .bounds import first_fit_decreasing, l1_bound, l2_bound, strip_lower_bound
from .check import bin_packing_errors, strip_packing_errors, waste
from .data import BinPackingData, StripData, crates, sign_order, week
from .model import StripPacking, solve_bin_packing, solve_strip_packing

FIGURES = Path(__file__).parent / "figures"
# Both bin packing models run on one worker with a fixed work limit, so the
# comparison gives the same result on every machine.
WORK_LIMIT = 0.5


def part_1() -> list[str]:
    """Solve a week of van loading. Return check errors."""
    print(f"Part 1: crates into vans (capacity {crates('Mon').capacity:,} kg)\n")
    rows, errors = [], []
    for data in week():
        day = data.name.split()[0]
        lb1, lb2 = (
            l1_bound(data.sizes, data.capacity),
            l2_bound(data.sizes, data.capacity),
        )
        ffd = len(first_fit_decreasing(data.sizes, data.capacity))
        best, plain = (
            solve_bin_packing(data, sb, workers=1, work_limit=WORK_LIMIT)
            for sb in (True, False)
        )
        rows.append(
            (
                day,
                sum(data.sizes.values()),
                lb1,
                lb2,
                ffd,
                best.n_bins,
                f"{best.work:.3f}",
                _plain_summary(plain),
            )
        )
        errors += [f"{day}: {e}" for e in bin_packing_errors(data, best.bins)]
        errors += [f"{day} plain: {e}" for e in bin_packing_errors(data, plain.bins)]
        if not best.proven_optimal:
            errors.append(f"{day}: no optimality proof")
        if best.n_bins < lb2:
            errors.append(f"{day}: {best.n_bins} vans beat the lower bound {lb2}")
        if plain.n_bins < best.n_bins:
            errors.append(f"{day}: plain model beat the proven optimum")
    header = ["day", "total kg", "L1", "L2", "FFD", "optimum", "proof work", "plain"]
    print(format_table(header, rows))
    print(
        "\nL1, L2: lower bounds. FFD: first-fit decreasing. 'optimum' and 'proof"
        "\nwork': CP-SAT with symmetry breaking, and the work it needed for the"
        "\nproof. 'plain': the same model without symmetry breaking, stopped"
        f"\nafter {WORK_LIMIT:g} work units (vans found / best proven lower bound)."
    )
    return errors


def _plain_summary(plain) -> str:
    if plain.proven_optimal:
        return f"{plain.n_bins}, proven after {plain.work:.3f}"
    return f"{plain.n_bins} / {plain.bound}, no proof"


def part_2(
    data: StripData, compare_grid: bool = False
) -> tuple[dict[str, StripPacking], list[str]]:
    """Cut the sign order with and without rotation. Return results and errors."""
    print(f"Part 2: {len(data.panels)} panels from a roll {data.strip_width} cm wide\n")
    runs = {
        "no rotation": dict(allow_rotation=False),
        "rotation": dict(allow_rotation=True),
    }
    if compare_grid:
        runs["rotation, 1 cm grid"] = dict(allow_rotation=True, scale_to_grid=False)
    results, rows, errors = {}, [], []
    for label, kwargs in runs.items():
        result = solve_strip_packing(data, **kwargs)
        results[label] = result
        rotate = kwargs["allow_rotation"]
        bound = strip_lower_bound(data.panels, data.strip_width, rotate)
        rows.append(
            (
                label,
                result.length,
                bound,
                result.bound,
                f"{waste(data, result.length):.1%}",
                f"{result.seconds:.2f}",
            )
        )
        errors += [
            f"{label}: {e}"
            for e in strip_packing_errors(
                data, result.placements, result.length, rotate
            )
        ]
        if result.length < bound:
            errors.append(f"{label}: length {result.length} beats the bound {bound}")
        # The 1 cm grid run is a speed demo: it may stop before its proof.
        if not result.proven_optimal and kwargs.get("scale_to_grid", True):
            errors.append(f"{label}: no optimality proof")
    header = ["setting", "length cm", "our bound", "CP-SAT bound", "waste", "seconds"]
    print(format_table(header, rows))
    print(
        "\n'our bound': area and grid bound from bounds.py. 'CP-SAT bound': the"
        "\nlower bound the solver proved. Equal length and bound means optimal."
    )
    return results, errors


def plot_vans(data: BinPackingData) -> Path:
    """Stacked bars: first-fit decreasing versus the optimal packing."""
    from examples.common.plotting import COLORS, new_figure, save

    packings = {
        "first-fit decreasing": first_fit_decreasing(data.sizes, data.capacity),
        "CP-SAT optimum": solve_bin_packing(data).bins,
    }
    fig, axes = new_figure(10, 3.8, ncols=2, sharey=True)
    for ax, (title, bins) in zip(axes, packings.items(), strict=True):
        for k, b in enumerate(bins):
            bottom = 0
            for j, name in enumerate(sorted(b, key=lambda n: -data.sizes[n])):
                size = data.sizes[name]
                ax.bar(
                    k + 1, size, bottom=bottom, color=COLORS[j % 6], ec="white", lw=0.8
                )
                bottom += size
        ax.axhline(data.capacity, color="black", lw=1)
        ax.set_title(f"{title}: {len(bins)} vans")
        ax.set_xlabel("van")
        ax.set_xticks(range(1, len(bins) + 1))
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("load (kg)")
    return save(fig, FIGURES / "vans.png")


def plot_layouts(data: StripData, results: dict[str, StripPacking]) -> Path:
    """Draw the roll for the runs without and with rotation."""
    from matplotlib.patches import Rectangle

    from examples.common.plotting import COLORS, new_figure, save

    shown = {k: results[k] for k in ("no rotation", "rotation")}
    longest = max(r.length for r in shown.values())
    fig, axes = new_figure(10, 7, ncols=2, sharey=True)
    for ax, (label, result) in zip(axes, shown.items(), strict=True):
        ax.add_patch(
            Rectangle((0, 0), data.strip_width, result.length, fc="0.93", ec="0.5")
        )
        for k, (name, q) in enumerate(result.placements.items()):
            ax.add_patch(
                Rectangle(
                    (q.x, q.y),
                    q.width,
                    q.height,
                    fc=COLORS[k % 6],
                    ec="white",
                    lw=1.5,
                    hatch="//" if q.rotated else None,
                    alpha=0.85,
                )
            )
            tall = q.height > 1.5 * q.width
            text = _short_label(name)
            if q.width < 50 and not tall:
                text = text.replace(" ", "\n")
            ax.text(
                q.x + q.width / 2,
                q.y + q.height / 2,
                text,
                ha="center",
                va="center",
                fontsize=7,
                rotation=90 if tall else 0,
            )
        ax.set_xlim(-5, data.strip_width + 5)
        ax.set_ylim(-5, longest + 5)
        ax.set_aspect("equal")
        ax.set_title(f"{label}: {result.length} cm of roll")
        ax.set_xlabel("across the roll (cm)")
        ax.grid(False)
    axes[0].set_ylabel("along the roll (cm)")
    fig.suptitle("Hatched panels are turned by 90 degrees")
    return save(fig, FIGURES / "layouts.png")


def _short_label(name: str) -> str:
    return name.replace("price tag ", "tag ").replace("door sign ", "door ")


def main() -> None:
    """Parse arguments, solve both parts, verify, and report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plot", action="store_true", help="write PNG figures")
    parser.add_argument(
        "--compare-grid",
        action="store_true",
        help="also solve the 2D part on a 1 cm grid (slower)",
    )
    args = parser.parse_args()

    errors = part_1()
    print("\n" + "=" * 72 + "\n")
    order = sign_order()
    results, strip_errors = part_2(order, args.compare_grid)
    errors += strip_errors

    if errors:
        print("\nVerification FAILED:", *errors, sep="\n  ")
    else:
        print(
            "\nVerification: every packing obeys all rules, no result beats its"
            "\nlower bound, and CP-SAT proved every result optimal."
        )
    if args.plot:
        for path in (plot_vans(crates("Mon")), plot_layouts(order, results)):
            print("Wrote", path)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
