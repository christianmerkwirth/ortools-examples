"""Load relief supplies onto a plane, then onto a convoy of trucks.

Run from the repository root:

    uv run python -m examples.ex03_knapsack.main
    uv run python -m examples.ex03_knapsack.main --plot
    uv run python -m examples.ex03_knapsack.main --benchmark
"""

import argparse
import time
from pathlib import Path

from examples.common.report import format_table

from .check import (
    best_single_vehicle_value,
    feasibility_errors,
    loading_value,
    pooled_bound,
)
from .data import LoadingData, cargo_plane, random_instance, truck_convoy
from .model import (
    Loading,
    solve_convoy,
    solve_with_cp_sat,
    solve_with_knapsack_solver,
)

FIGURES = Path(__file__).parent / "figures"


def vehicle_rows(data: LoadingData, loading: Loading) -> list[tuple]:
    """Return one table row per vehicle: items, weight, volume, value."""
    items = {i.name: i for i in data.items}
    rows = []
    for v in data.vehicles:
        load = [items[n] for n in loading.items_on(v.name)]
        rows.append(
            (
                v.name,
                len(load),
                f"{sum(i.weight for i in load):,} / {v.max_weight:,}",
                f"{sum(i.volume for i in load):.1f} / {v.max_volume:.1f}",
                sum(i.value for i in load),
            )
        )
    return rows


def part_a(data: LoadingData) -> tuple[Loading, list[str]]:
    """Solve the plane with both solvers and compare with the DP referee."""
    print("Part A: one cargo plane (4,000 kg, 15 m³)\n")
    rows, results = [], {}
    for name, solve in (
        ("knapsack solver", solve_with_knapsack_solver),
        ("CP-SAT", solve_with_cp_sat),
    ):
        start = time.perf_counter()
        loading = solve(data)
        seconds = time.perf_counter() - start
        results[name] = loading
        rows.append((name, loading.value, str(loading.proven_optimal), seconds))
    reference = best_single_vehicle_value(data)
    rows.append(("dynamic programming (check.py)", reference, "True", ""))
    print(format_table(["method", "score", "optimal", "seconds"], rows))

    loading = results["knapsack solver"]
    print()
    header = ["vehicle", "items", "kg used / max", "m³ used / max", "score"]
    print(format_table(header, vehicle_rows(data, loading)))
    left = [i for i in data.items if i.name not in loading.assignment]
    print("\nLeft behind:", ", ".join(i.name for i in left))

    errors = []
    for name, result in results.items():
        errors += [f"{name}: {e}" for e in feasibility_errors(data, result.assignment)]
        if loading_value(data, result.assignment) != result.value:
            errors.append(f"{name}: reported score does not match its items")
        if result.value != reference:
            errors.append(f"{name}: score {result.value} != DP optimum {reference}")
    return loading, errors


def part_b(data: LoadingData) -> tuple[Loading, list[str]]:
    """Solve the convoy with CP-SAT."""
    print("Part B: a convoy of three trucks with side rules\n")
    loading = solve_convoy(data)
    print(f"Total score: {loading.value}  (proven optimal: {loading.proven_optimal})\n")
    header = ["vehicle", "items", "kg used / max", "m³ used / max", "score"]
    print(format_table(header, vehicle_rows(data, loading)))
    for v in data.vehicles:
        print(f"\n{v.name}: " + ", ".join(loading.items_on(v.name)))
    left = [i.name for i in data.items if i.name not in loading.assignment]
    print("\nLeft behind:", ", ".join(left))

    errors = feasibility_errors(data, loading.assignment)
    if loading_value(data, loading.assignment) != loading.value:
        errors.append("convoy: reported score does not match its items")
    if not loading.proven_optimal:
        errors.append("convoy: CP-SAT did not prove optimality")
    bound = pooled_bound(data)
    print(f"\nUpper bound from one pooled truck (check.py): {bound}")
    if loading.value > bound:
        errors.append(f"convoy: score {loading.value} beats the bound {bound}")
    return loading, errors


def benchmark(sizes=(20, 40, 60, 80, 100, 200, 500), time_limit=5.0) -> list[tuple]:
    """Time both solvers on random one-vehicle instances of growing size."""
    rows = []
    for n in sizes:
        data = random_instance(n, 1, seed=1)
        row = [n]
        for solve in (solve_with_knapsack_solver, solve_with_cp_sat):
            start = time.perf_counter()
            loading = solve(data, time_limit=time_limit)
            row += [time.perf_counter() - start, loading.value, loading.proven_optimal]
        rows.append(tuple(row))
    return rows


def plot_loadings(plane: LoadingData, plane_load: Loading, convoy, convoy_load) -> Path:
    """Weight and volume use of every vehicle, as % of its limit."""
    from examples.common.plotting import COLORS, new_figure, save

    fig, ax = new_figure(7, 3.2)
    labels, weight, volume = [], [], []
    for data, loading in ((plane, plane_load), (convoy, convoy_load)):
        items = {i.name: i for i in data.items}
        for v in data.vehicles:
            load = [items[n] for n in loading.items_on(v.name)]
            labels.append(v.name)
            weight.append(100 * sum(i.weight for i in load) / v.max_weight)
            volume.append(100 * sum(i.volume for i in load) / v.max_volume)
    ys = range(len(labels))
    ax.barh([y - 0.2 for y in ys], weight, height=0.4, color=COLORS[0], label="weight")
    ax.barh([y + 0.2 for y in ys], volume, height=0.4, color=COLORS[1], label="volume")
    ax.axvline(100, color="black", lw=1)
    ax.set_yticks(list(ys), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 135)
    ax.set_xlabel("% of the vehicle's limit")
    ax.set_title("How full is each vehicle?")
    ax.legend(loc="lower right")
    return save(fig, FIGURES / "vehicle_use.png")


def plot_benchmark(rows: list[tuple], time_limit: float) -> Path:
    """Solve time versus number of items for both solvers."""
    from examples.common.plotting import COLORS, new_figure, save

    fig, ax = new_figure(7, 3.8)
    n = [r[0] for r in rows]
    for col, name, color in (
        (1, "knapsack solver (B&B)", COLORS[3]),
        (4, "CP-SAT", COLORS[0]),
    ):
        times = [r[col] for r in rows]
        proven = [r[col + 2] for r in rows]
        ax.plot(n, times, "-", color=color, label=name)
        ax.plot(
            [x for x, p in zip(n, proven, strict=True) if p],
            [t for t, p in zip(times, proven, strict=True) if p],
            "o",
            color=color,
        )
        ax.plot(
            [x for x, p in zip(n, proven, strict=True) if not p],
            [t for t, p in zip(times, proven, strict=True) if not p],
            "x",
            color=color,
            ms=9,
            mew=2,
        )
    ax.axhline(time_limit, color="0.5", ls="--", lw=1)
    ax.text(n[0], time_limit * 1.15, f"time limit {time_limit:g} s", color="0.4")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(n, [str(x) for x in n])
    ax.minorticks_off()
    ax.set_xlabel("number of items (one vehicle, weight + volume)")
    ax.set_ylabel("seconds")
    ax.set_title(
        "Two limits per item: CP-SAT scales, branch and bound does not\n"
        "(o = proven optimal, x = stopped at the time limit)"
    )
    ax.legend(loc="center right")
    return save(fig, FIGURES / "benchmark.png")


def main() -> None:
    """Parse arguments, solve both parts, verify, and report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plot", action="store_true", help="write PNG figures")
    parser.add_argument("--benchmark", action="store_true", help="time both solvers")
    args = parser.parse_args()

    plane, convoy = cargo_plane(), truck_convoy()
    plane_load, errors = part_a(plane)
    print("\n" + "=" * 72 + "\n")
    convoy_load, convoy_errors = part_b(convoy)
    errors += convoy_errors

    if args.benchmark:
        time_limit = 5.0
        print("\n" + "=" * 72 + "\n")
        print(f"Benchmark: random one-vehicle instances, time limit {time_limit:g} s\n")
        rows = benchmark(time_limit=time_limit)
        header = [
            "items",
            "B&B s",
            "B&B score",
            "B&B opt",
            "CP-SAT s",
            "CP-SAT score",
            "CP-SAT opt",
        ]
        print(
            format_table(
                header,
                [tuple(str(v) if isinstance(v, bool) else v for v in r) for r in rows],
            )
        )
        if args.plot:
            print("Wrote", plot_benchmark(rows, time_limit))

    if errors:
        print("\nVerification FAILED:", *errors, sep="\n  ")
    else:
        print(
            "\nVerification: all loadings obey every rule; both plane solvers"
            "\nmatch the dynamic-programming optimum; the convoy is proven optimal"
            "\nby CP-SAT and reaches the independent pooled-truck bound."
        )
    if args.plot:
        print("Wrote", plot_loadings(plane, plane_load, convoy, convoy_load))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
