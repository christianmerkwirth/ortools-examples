"""Plan a technician's round trip with the routing library, then grade it.

Run from the repository root:

    uv run python -m examples.ex12_tsp.main
    uv run python -m examples.ex12_tsp.main --plot
"""

import argparse
from pathlib import Path

from examples.common.report import format_table

from .check import tour_errors, tour_length
from .data import METERS_PER_KM, TspData, cell_towers
from .model import Tour, solve_exact, solve_with_routing

FIGURES = Path(__file__).parent / "figures"

FIRST_SOLUTION_STRATEGIES = [
    "PATH_CHEAPEST_ARC",
    "GLOBAL_CHEAPEST_ARC",
    "SAVINGS",
    "CHRISTOFIDES",
    "PARALLEL_CHEAPEST_INSERTION",
    "LOCAL_CHEAPEST_INSERTION",
]
GLS_SECONDS = 2.0
METAHEURISTICS = ["GUIDED_LOCAL_SEARCH", "TABU_SEARCH", "SIMULATED_ANNEALING"]
TIME_LIMITS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]


def km(meters: int) -> float:
    """Convert meters to km."""
    return meters / METERS_PER_KM


def gap(tour: Tour, best: Tour) -> float:
    """Return how much longer a tour is than the best one, in percent."""
    return 100 * (tour.length / best.length - 1)


def verify(data: TspData, tours: list[Tour], exact: Tour) -> list[str]:
    """Check every tour independently and against the exact lower bound."""
    errors = []
    for t in tours:
        errors += [f"{t.method}: {e}" for e in tour_errors(data, t.route)]
        if tour_length(data, t.route) != t.length:
            errors.append(f"{t.method}: reported length differs from the map")
        if t.length < exact.lower_bound:
            errors.append(f"{t.method}: shorter than the proven lower bound")
    return errors


def plot_routes(data: TspData, first: Tour, best: Tour, exact: Tour) -> Path:
    """Draw the first tour next to the improved tour."""
    from examples.common.plotting import COLORS, new_figure, save

    fig, axes = new_figure(10, 5, ncols=2, sharey=True)
    for ax, tour, color in zip(
        axes, (first, best), (COLORS[3], COLORS[0]), strict=True
    ):
        xs = [data.sites[k].x for k in tour.route]
        ys = [data.sites[k].y for k in tour.route]
        ax.plot(xs, ys, "-", color=color, lw=1.4)
        ax.plot(xs[1:-1], ys[1:-1], "o", color="0.25", ms=3)
        depot = data.sites[data.depot]
        ax.plot(depot.x, depot.y, "s", color="black", ms=9, label="depot")
        summary = f"{km(tour.length):,.1f} km, {gap(tour, exact):+.2f}% vs optimum"
        ax.set_title(f"{tour.method}\n{summary}", fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlabel("km")
    axes[0].set_ylabel("km")
    axes[0].legend(loc="upper left")
    return save(fig, FIGURES / "routes.png")


def plot_search(
    rows: list[tuple], curves: dict[str, list[tuple[float, float]]]
) -> Path:
    """Two panels: first-solution strategies, and metaheuristics over time."""
    import numpy as np

    from examples.common.plotting import COLORS, new_figure, save

    fig, (left, right) = new_figure(11, 4.2, ncols=2, width_ratios=[1.2, 1])
    names = [r[0] for r in rows]
    y = np.arange(len(rows))
    left.barh(y - 0.2, [r[2] for r in rows], 0.4, color=COLORS[3], label="first tour")
    left.barh(
        y + 0.2, [r[4] for r in rows], 0.4, color=COLORS[0], label="+ greedy descent"
    )
    left.set_yticks(y, [n.replace("_", " ").lower() for n in names])
    left.invert_yaxis()
    left.set_xlabel("% longer than the optimal tour")
    left.set_title("First-solution strategies")
    left.legend(loc="lower right")

    # Tabu search and simulated annealing can give identical tours; distinct
    # line styles keep both visible when they overlap.
    styles = ["o-", "s--", "^:"]
    for (name, points), color, style in zip(
        curves.items(), COLORS[:3], styles, strict=False
    ):
        right.plot(
            [t for t, _ in points],
            [g for _, g in points],
            style,
            color=color,
            ms=7,
            label=name.replace("_", " ").lower(),
        )
    right.set_xscale("log")
    right.set_xticks(TIME_LIMITS, [f"{t:g}" for t in TIME_LIMITS])
    right.minorticks_off()
    right.set_xlabel("time limit (s)")
    right.set_ylabel("% longer than the optimal tour")
    right.set_ylim(bottom=0)
    right.set_title("Metaheuristics (start: path cheapest arc)")
    right.legend()
    return save(fig, FIGURES / "search.png")


def metaheuristic_curves(
    data: TspData, exact: Tour
) -> dict[str, list[tuple[float, float]]]:
    """Run each metaheuristic with growing time limits; return (seconds, gap)."""
    curves = {}
    for name in METAHEURISTICS:
        curves[name] = [
            (
                limit,
                gap(
                    solve_with_routing(data, metaheuristic=name, time_limit=limit),
                    exact,
                ),
            )
            for limit in TIME_LIMITS
        ]
    return curves


def main() -> None:
    """Solve, compare strategies, verify, and report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plot", action="store_true", help="write PNG figures")
    args = parser.parse_args()

    data = cell_towers()
    print(f"Instance: {data.name}\n")

    exact = solve_exact(data)
    status = "proven optimal" if exact.proven_optimal else "best found"
    print(
        f"CP-SAT exact baseline: {km(exact.length):,.1f} km ({status}, "
        f"lower bound {km(exact.lower_bound):,.1f} km, {exact.seconds:.1f} s)\n"
    )

    rows, tours = [], [exact]
    for name in FIRST_SOLUTION_STRATEGIES:
        raw = solve_with_routing(data, name, first_solution_only=True)
        improved = solve_with_routing(data, name)
        tours += [raw, improved]
        rows.append(
            (
                name,
                km(raw.length),
                gap(raw, exact),
                km(improved.length),
                gap(improved, exact),
            )
        )
    print("First-solution strategies (gap = % longer than the CP-SAT tour)")
    print(
        format_table(
            ["strategy", "first km", "gap %", "after descent km", "gap %"], rows
        )
    )

    gls = solve_with_routing(
        data, metaheuristic="GUIDED_LOCAL_SEARCH", time_limit=GLS_SECONDS
    )
    tours.append(gls)
    print(
        f"\nPATH_CHEAPEST_ARC + guided local search ({GLS_SECONDS:g} s): "
        f"{km(gls.length):,.1f} km, gap {gap(gls, exact):.2f}%"
    )

    errors = verify(data, tours, exact)
    if errors:
        print("\nVerification FAILED:", *errors, sep="\n  ")
    else:
        print(
            f"\nVerification: all {len(tours)} tours are valid round trips, their"
            "\nlengths match the map, and none beats the CP-SAT lower bound."
        )

    if args.plot:
        first = solve_with_routing(data, first_solution_only=True)
        curves = metaheuristic_curves(data, exact)
        for path in (
            plot_routes(data, first, gls, exact),
            plot_search(rows, curves),
        ):
            print("Wrote", path)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
