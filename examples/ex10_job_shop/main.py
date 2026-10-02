"""Schedule a machine shop with CP-SAT, then solve two classic benchmarks.

Run from the repository root:

    uv run python -m examples.ex10_job_shop.main
    uv run python -m examples.ex10_job_shop.main --plot
"""

import argparse
from pathlib import Path

from examples.common.report import format_table

from .check import (
    critical_path,
    end_time,
    feasibility_errors,
    lower_bound,
    makespan,
    weighted_tardiness,
)
from .data import JobShopData, ft06, la01, machine_shop, random_instance
from .model import Schedule, solve, solve_lexicographic

FIGURES = Path(__file__).parent / "figures"


def check_schedule(data: JobShopData, schedule: Schedule) -> list[str]:
    """Run the independent checks on one schedule. Return errors."""
    errors = [f"{data.name}: {e}" for e in feasibility_errors(data, schedule.start)]
    if makespan(data, schedule.start) != schedule.makespan:
        errors.append(f"{data.name}: reported makespan {schedule.makespan} is wrong")
    if max(lower_bound(data).values()) > schedule.makespan:
        errors.append(f"{data.name}: makespan beats a proven lower bound")
    return errors


def print_shop(data: JobShopData, fast: Schedule, on_time: Schedule) -> list[str]:
    """Print the story instance: schedule, bounds, and the trade-off."""
    print(f"Shortest schedule for the {data.name}: {fast.makespan} minutes")
    print(f"(proven optimal: {fast.optimal})\n")
    rows = []
    for job in data.jobs:
        finish = end_time(data, fast.start, (job.name, len(job.operations) - 1))
        rows.append(
            (
                job.name,
                " > ".join(op.machine for op in job.operations),
                fast.start[job.name, 0],
                finish,
                job.due,
                max(0, finish - job.due),
            )
        )
    print(format_table(["job", "route", "start", "finish", "due", "late"], rows))

    print("\nMachine use")
    rows = []
    for m in data.machines:
        load = sum(
            op.duration for j in data.jobs for op in j.operations if op.machine == m
        )
        rows.append((m, load, f"{100 * load / fast.makespan:.0f}%"))
    print(format_table(["machine", "busy (min)", "utilization"], rows))

    print("\nLower bounds (no schedule can be shorter)")
    print(format_table(["bound", "minutes"], list(lower_bound(data).items())))

    path = critical_path(data, fast.start)
    print("\nCritical path (delay any of these and the whole schedule slips):")
    rows = []
    for job, k in path:
        op = data.job(job).operations[k]
        rows.append(
            (fast.start[job, k], fast.start[job, k] + op.duration, op.machine, job)
        )
    print(format_table(["from", "to", "machine", "job"], rows))

    print("\nTwo goals, two schedules")
    rows = []
    for label, s in (
        ("shortest makespan first", fast),
        ("least lateness first", on_time),
    ):
        rows.append(
            (label, s.makespan, weighted_tardiness(data, s.start), str(s.optimal))
        )
    print(format_table(["priority", "makespan", "weighted lateness", "optimal"], rows))

    errors = check_schedule(data, fast) + check_schedule(data, on_time)
    # A lexicographic result reports the objective of its second pass.
    if weighted_tardiness(data, fast.start) != fast.objective:
        errors.append("reported weighted tardiness is wrong")
    return errors


def print_benchmarks(instances: list[JobShopData]) -> list[str]:
    """Solve benchmarks with known optima and compare."""
    print("\nBenchmarks from the OR-Library")
    rows, errors = [], []
    for data in instances:
        s = solve(data, time_limit=30)
        rows.append(
            (
                data.name,
                f"{len(data.jobs)}x{len(data.machines)}",
                s.makespan,
                data.known_optimum,
                max(lower_bound(data).values()),
                str(s.optimal),
                f"{s.wall_time:.2f} s",
            )
        )
        errors += check_schedule(data, s)
        if s.makespan != data.known_optimum:
            errors.append(
                f"{data.name}: makespan {s.makespan} != known {data.known_optimum}"
            )
    headers = [
        "instance",
        "size",
        "makespan",
        "known optimum",
        "simple bound",
        "proven",
        "time",
    ]
    print(format_table(headers, rows))
    return errors


# ----------------------------------------------------------------- plots ---


def _gantt(ax, data: JobShopData, start, title: str, highlight=()) -> list:
    """Draw one Gantt chart: a row per machine, a color per job.

    Return legend handles, one colored patch per job.
    """
    from matplotlib.patches import Patch

    from examples.common.plotting import COLORS

    palette = [*COLORS[:7], "#999999"]
    color = {j.name: palette[k % len(palette)] for k, j in enumerate(data.jobs)}
    row = {m: k for k, m in enumerate(data.machines)}
    highlight = set(highlight)
    for job in data.jobs:
        for k, op in enumerate(job.operations):
            on_path = (job.name, k) in highlight
            ax.barh(
                row[op.machine],
                op.duration,
                left=start[job.name, k],
                color=color[job.name],
                edgecolor="black" if on_path else "white",
                linewidth=2.0 if on_path else 0.5,
                height=0.7,
            )
    ax.set_yticks(range(len(data.machines)), data.machines)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_title(title, loc="left")
    return [Patch(color=color[j.name], label=j.name) for j in data.jobs]


def plot_shop(data: JobShopData, fast: Schedule, on_time: Schedule) -> Path:
    """Two Gantt charts of the story instance, one per goal."""
    from examples.common.plotting import new_figure, save

    fig, (top, bottom) = new_figure(10, 6.2, nrows=2, sharex=True)
    handles = _gantt(
        top,
        data,
        fast.start,
        f"Shortest makespan: {fast.makespan} min, weighted lateness "
        f"{weighted_tardiness(data, fast.start)} (black outline = critical path)",
        critical_path(data, fast.start),
    )
    _gantt(
        bottom,
        data,
        on_time.start,
        f"Least weighted lateness: {weighted_tardiness(data, on_time.start)}, "
        f"makespan {on_time.makespan} min",
    )
    for job in data.jobs:
        bottom.axvline(job.due, color="0.6", lw=0.6, ls=":")
    bottom.set_xlabel("minutes (dotted lines: due times)")
    fig.legend(handles=handles, loc="outside right center", frameon=False)
    return save(fig, FIGURES / "machine_shop_gantt.png")


def plot_ft06(data: JobShopData) -> Path:
    """Gantt chart of an optimal ft06 schedule with its critical path."""
    from examples.common.plotting import new_figure, save

    s = solve(data)
    fig, ax = new_figure(9, 3.6)
    handles = _gantt(
        ax,
        data,
        s.start,
        f"ft06: optimal makespan {s.makespan} (black outline = critical path)",
        critical_path(data, s.start),
    )
    ax.set_xlabel("time")
    fig.legend(handles=handles, loc="outside right center", frameon=False)
    return save(fig, FIGURES / "ft06_gantt.png")


def plot_progress(data: JobShopData) -> Path:
    """Makespan of the best schedule and proven bound over solve time."""
    from examples.common.plotting import COLORS, new_figure, save

    s = solve(data, time_limit=30, record_progress=True)
    times = [t for t, _, _ in s.progress]
    fig, ax = new_figure(7, 4)
    ax.step(
        times,
        [v for _, v, _ in s.progress],
        where="post",
        color=COLORS[0],
        lw=2,
        label="best schedule found",
    )
    ax.step(
        times,
        [b for _, _, b in s.progress],
        where="post",
        color=COLORS[1],
        lw=2,
        label="proven lower bound",
    )
    ax.axhline(
        max(lower_bound(data).values()),
        color="0.5",
        ls="--",
        lw=1,
        label="simple bound (check.py)",
    )
    ax.set_xscale("log")
    ax.set_xlabel("solve time (seconds, log scale)")
    ax.set_ylabel("makespan")
    status = "proven optimal" if s.optimal else f"gap {s.makespan - s.best_bound}"
    ax.set_title(f"CP-SAT on a {data.name} instance: {s.makespan}, {status}")
    ax.legend()
    return save(fig, FIGURES / "solver_progress.png")


def main() -> None:
    """Solve, verify, and report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plot", action="store_true", help="write PNG figures")
    args = parser.parse_args()

    shop = machine_shop()
    fast = solve_lexicographic(shop, "makespan", "tardiness")
    on_time = solve_lexicographic(shop, "tardiness", "makespan")
    errors = print_shop(shop, fast, on_time)
    errors += print_benchmarks([ft06(), la01()])

    if errors:
        print("\nVerification FAILED:", *errors, sep="\n  ")
    else:
        print(
            "\nVerification: every schedule is feasible, and both benchmarks"
            "\nmatch their known optimal makespan."
        )
    if args.plot:
        paths = (
            plot_shop(shop, fast, on_time),
            plot_ft06(ft06()),
            plot_progress(random_instance(15, 10, seed=1)),
        )
        for path in paths:
            print("Wrote", path)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
