"""Roster nurses for two weeks with CP-SAT and print the result.

Run from the repository root:

    uv run python -m examples.ex09_shift_scheduling.main
    uv run python -m examples.ex09_shift_scheduling.main --plot
"""

import argparse
from pathlib import Path

from examples.common.report import format_table

from .check import hard_rule_errors, penalty_breakdown
from .data import RosterData, day_label, ward_two_weeks
from .model import (
    CATEGORIES,
    Roster,
    night_fairness_tradeoff,
    solve_lexicographic,
    solve_roster,
)

FIGURES = Path(__file__).parent / "figures"


def broken_requests(data: RosterData, roster: Roster) -> list:
    """Return the requests that the roster does not honor."""
    out = []
    for r in data.requests:
        shift = roster.assignment[r.nurse, r.day]
        if (r.shift is None and shift is not None) or (
            shift is not None and shift == r.shift
        ):
            out.append(r)
    return out


def print_roster(data: RosterData, roster: Roster) -> None:
    """Print the roster as a grid: one row per nurse, one column per day."""
    broken = {(r.nurse, r.day) for r in broken_requests(data, roster)}
    header = "nurse     " + " ".join(day_label(d)[:2] for d in range(data.num_days))
    print(header)
    for n in data.nurses:
        cells = []
        for d in range(data.num_days):
            shift = roster.assignment[n.name, d]
            cell = "L" if d in n.leave else (shift or ".")
            cells.append(cell + ("!" if (n.name, d) in broken else " "))
        tag = "*" if n.senior else " "
        print(f"{n.name:<8}{tag} " + " ".join(cells))
    print("\nD/E/N = day/evening/night, . = off, L = leave, * = senior,")
    print("! = a request on that day is not honored")


def print_report(data: RosterData, roster: Roster) -> None:
    """Print the roster, the penalties, and the workload per nurse."""
    status = "proven optimal" if roster.optimal else f"gap {roster.gap:.1%}"
    print(
        f"Roster for '{data.name}': {roster.objective} penalty points "
        f"({status}, best bound {roster.best_bound:g}, {roster.wall_time:.2f} s)\n"
    )
    print_roster(data, roster)

    print("\nPenalty points")
    rows = [(c, roster.penalty[c]) for c in CATEGORIES]
    rows.append(("total", roster.objective))
    print(format_table(["soft rule", "points"], rows))

    print("\nRequests not honored")
    for r in broken_requests(data, roster):
        what = "day off" if r.shift is None else f"no {r.shift} shift"
        points = f"{r.weight} point" + ("s" if r.weight != 1 else "")
        print(f"  * {r.nurse}: {what} on {day_label(r.day)} ({points})")

    print("\nWorkload")
    rows = []
    for n in data.nurses:
        pattern = [roster.assignment[n.name, d] for d in range(data.num_days)]
        rows.append(
            (
                n.name,
                "full" if n.full_time else "part",
                sum(s is not None for s in pattern),
                f"{n.min_shifts}-{n.max_shifts}",
                *(pattern.count(s.code) for s in data.shifts),
                sum(pattern[d] is not None for d in data.weekend_days),
            )
        )
    codes = [s.code for s in data.shifts]
    print(
        format_table(["nurse", "time", "shifts", "contract", *codes, "weekend"], rows)
    )


def print_strategies(data: RosterData, weighted: Roster) -> None:
    """Compare the weighted objective with two lexicographic orders."""
    _, fair_first = solve_lexicographic(data, ("night spread", "weekend spread"))
    _, wishes_first = solve_lexicographic(data, ("requests",))
    print("\nThree ways to rank the goals (penalty points per soft rule)")
    rows = [
        (label, *(r.penalty[c] for c in CATEGORIES), r.objective)
        for label, r in (
            ("weighted sum", weighted),
            ("fairness first", fair_first),
            ("wishes first", wishes_first),
        )
    ]
    print(format_table(["strategy", *CATEGORIES, "total"], rows))


def plot_roster(data: RosterData, roster: Roster) -> Path:
    """Draw the roster as a colored grid with leave and broken requests."""
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch, Rectangle

    from examples.common.plotting import COLORS, new_figure, save

    codes = [s.code for s in data.shifts]
    colors = ["#f2f2f2", COLORS[6], COLORS[1], COLORS[0]]  # off, D, E, N
    grid = [
        [
            0
            if roster.assignment[n.name, d] is None
            else 1 + codes.index(roster.assignment[n.name, d])
            for d in range(data.num_days)
        ]
        for n in data.nurses
    ]
    fig, ax = new_figure(9, 4.6)
    ax.imshow(grid, cmap=ListedColormap(colors), vmin=0, vmax=3, aspect="auto")
    ax.grid(False)
    for d in data.weekend_days:
        ax.add_patch(
            Rectangle(
                (d - 0.5, -0.5), 1, len(data.nurses), fill=False, ec="0.4", lw=1.5
            )
        )
    for i, n in enumerate(data.nurses):
        for d in range(data.num_days):
            shift = roster.assignment[n.name, d]
            if d in n.leave:
                ax.text(d, i, "L", ha="center", va="center", color="0.4")
            elif shift:
                text_color = "white" if shift == "N" else "black"
                ax.text(
                    d, i, shift, ha="center", va="center", color=text_color, fontsize=8
                )
    for r in broken_requests(data, roster):
        i = [n.name for n in data.nurses].index(r.nurse)
        ax.plot(r.day + 0.32, i - 0.28, marker="v", color=COLORS[3], ms=6)
    ax.set_xticks(
        range(data.num_days), [day_label(d) for d in range(data.num_days)], rotation=60
    )
    ax.set_yticks(
        range(len(data.nurses)),
        [n.name + (" *" if n.senior else "") for n in data.nurses],
    )
    handles = [
        Patch(color=c, label=label)
        for c, label in zip(colors, ["off", "day", "evening", "night"], strict=True)
    ]
    handles.append(
        ax.plot([], [], "v", color=COLORS[3], label="request not honored")[0]
    )
    ax.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False
    )
    ax.set_title(
        f"Two-week roster: {roster.objective} penalty points"
        " (boxes = weekends, * = senior)"
    )
    return save(fig, FIGURES / "roster.png")


def plot_tradeoff(data: RosterData) -> Path:
    """Plot the wish points that each level of night fairness costs."""
    from examples.common.plotting import COLORS, new_figure, save

    points = night_fairness_tradeoff(data, [0, 1, 2, 3, 4])
    ks = [k for k, _ in points]
    wishes = [r.penalty["requests"] + r.penalty["isolated days off"] for _, r in points]
    fig, ax = new_figure(6, 3.6)
    ax.plot(ks, wishes, "o-", color=COLORS[0], lw=2)
    for k, v in zip(ks, wishes, strict=True):
        ax.annotate(
            f"{v}", (k, v), textcoords="offset points", xytext=(0, 8), ha="center"
        )
    ax.set_xticks(ks)
    ax.set_xlabel("allowed night spread (most nights minus fewest, full-time nurses)")
    ax.set_ylabel("wish points\n(requests + isolated days off)")
    ax.set_ylim(0, max(wishes) + 3)
    ax.set_title("The price of fair nights")
    return save(fig, FIGURES / "fairness_tradeoff.png")


def main() -> None:
    """Parse arguments, solve, verify, and report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plot", action="store_true", help="write PNG figures")
    args = parser.parse_args()

    data = ward_two_weeks()
    roster = solve_roster(data)
    print_report(data, roster)
    print_strategies(data, roster)

    errors = hard_rule_errors(data, roster.assignment)
    recomputed = penalty_breakdown(data, roster.assignment)
    if recomputed != roster.penalty:
        errors.append(f"penalties differ: solver {roster.penalty}, check {recomputed}")
    if not roster.optimal:
        errors.append("the roster is not proven optimal")
    if errors:
        print("\nVerification FAILED:", *errors, sep="\n  ")
    else:
        print(
            "\nVerification: every hard rule holds, the penalties match an"
            "\nindependent count, and CP-SAT proved the roster optimal."
        )

    if args.plot:
        for path in (plot_roster(data, roster), plot_tradeoff(data)):
            print("Wrote", path)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
