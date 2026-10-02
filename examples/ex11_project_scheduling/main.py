"""Schedule two houses that share crews, and find the bottleneck crew.

Run from the repository root:

    uv run python -m examples.ex11_project_scheduling.main
    uv run python -m examples.ex11_project_scheduling.main --plot
"""

import argparse
import re
from pathlib import Path

from examples.common.report import format_table

from .check import (
    critical_path_length,
    energy_bound,
    feasibility_errors,
    latest_finish_rule,
    makespan,
    resource_profile,
)
from .data import ProjectData, housing_row
from .model import ProjectSchedule, extra_unit_gains, solve_schedule

FIGURES = Path(__file__).parent / "figures"
N_HOUSES = 2


def _split(name: str) -> tuple[str, int]:
    """Split "drywall [2]" into ("drywall", 2)."""
    match = re.fullmatch(r"(.*) \[(\d+)\]", name)
    assert match, name
    return match.group(1), int(match.group(2))


def print_report(data: ProjectData, schedule: ProjectSchedule) -> list[str]:
    """Print bounds, the schedule, and a what-if study. Return check errors."""
    heuristic = latest_finish_rule(data)
    cpl, energy = critical_path_length(data), energy_bound(data)
    print(f"Project: {data.name}, {len(data.activities)} activities\n")
    rows = [
        ("critical path (unlimited crews)", cpl, "lower bound"),
        ("resource energy bound", energy, "lower bound"),
        ("priority-rule heuristic", makespan(data, heuristic), "feasible schedule"),
        (
            "CP-SAT",
            schedule.makespan,
            "proven optimal" if schedule.proven_optimal else "best found",
        ),
    ]
    print(format_table(["method", "days", "meaning"], rows))
    stretch = schedule.makespan - cpl
    print(
        f"\nShared crews stretch the project by {stretch} days ({stretch / cpl:.0%})."
    )

    # Schedule table: one row per activity, one column pair per house.
    print("\nSchedule (start day - end day)")
    base_names = list(dict.fromkeys(_split(a.name)[0] for a in data.activities))
    rows = []
    for base in base_names:
        cells = []
        for h in range(1, N_HOUSES + 1):
            a = data.activity(f"{base} [{h}]")
            s = schedule.start[a.name]
            cells.append(f"{s:>3} - {s + a.duration:<3}")
        rows.append((base, *cells))
    print(
        format_table(
            ["activity", *(f"house {h}" for h in range(1, N_HOUSES + 1))], rows
        )
    )

    print("\nWhat if we add one unit of a resource?")
    gains = extra_unit_gains(data, schedule.makespan)
    rows = [(r.name, r.capacity, r.capacity + 1, gains[r.name]) for r in data.resources]
    print(format_table(["resource", "now", "with one more", "days saved"], rows))

    errors = feasibility_errors(data, schedule.start)
    errors += [f"heuristic: {e}" for e in feasibility_errors(data, heuristic)]
    if schedule.makespan < max(cpl, energy):
        errors.append("makespan is below a lower bound")
    return errors


def plot_gantt(data: ProjectData, schedule: ProjectSchedule) -> Path:
    """Gantt chart: one row per activity, one bar per house."""
    from examples.common.plotting import COLORS, new_figure, save

    base_names = list(dict.fromkeys(_split(a.name)[0] for a in data.activities))
    fig, ax = new_figure(10, 8.5)
    height = 0.8 / N_HOUSES
    for a in data.activities:
        base, h = _split(a.name)
        y = base_names.index(base) - 0.4 + (h - 1) * height
        ax.broken_barh(
            [(schedule.start[a.name], a.duration)],
            (y, height * 0.9),
            color=COLORS[h - 1],
            label=f"house {h}" if base == base_names[0] else None,
        )
    ax.set_yticks(range(len(base_names)), base_names)
    ax.invert_yaxis()
    ax.set_xlabel("working day")
    ax.set_xlim(0, schedule.makespan + 1)
    ax.axvline(schedule.makespan, color="black", lw=1)
    ax.set_title(f"Optimal schedule: {schedule.makespan} working days")
    ax.legend(loc="upper right")
    return save(fig, FIGURES / "gantt.png")


def plot_resources(data: ProjectData, schedule: ProjectSchedule) -> Path:
    """Daily use of every resource, against its capacity."""
    from examples.common.plotting import COLORS, new_figure, save

    n = len(data.resources)
    fig, axes = new_figure(10, 1.4 * n, nrows=n, sharex=True)
    days = range(schedule.makespan + 1)
    for ax, r in zip(axes, data.resources, strict=True):
        profile = resource_profile(data, schedule.start, r.name)
        ax.stairs(profile, days, fill=True, color=COLORS[5], alpha=0.8)
        ax.axhline(r.capacity, color=COLORS[3], lw=1.5, ls="--")
        load = sum(profile) / (r.capacity * schedule.makespan)
        ax.set_ylabel(r.name, rotation=0, ha="right", va="center")
        ax.set_ylim(0, r.capacity + 0.6)
        ax.set_yticks(range(r.capacity + 1))
        ax.text(
            schedule.makespan + 0.5, r.capacity / 2, f"{load:.0%} busy", va="center"
        )
    axes[0].set_title("Resource use per day (dashed line: capacity)")
    axes[-1].set_xlabel("working day")
    axes[-1].set_xlim(0, schedule.makespan + 6)
    return save(fig, FIGURES / "resources.png")


def main() -> None:
    """Parse arguments, solve, verify, and report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plot", action="store_true", help="write PNG figures")
    args = parser.parse_args()

    data = housing_row(N_HOUSES)
    schedule = solve_schedule(data)
    errors = print_report(data, schedule)

    if errors:
        print("\nVerification FAILED:", *errors, sep="\n  ")
    else:
        print(
            "\nVerification: the schedule meets every precedence and every"
            "\ndaily crew limit; CP-SAT proved that no shorter schedule exists."
            if schedule.proven_optimal
            else "\nVerification: the schedule is feasible (optimality not proven)."
        )
    if args.plot:
        for path in (plot_gantt(data, schedule), plot_resources(data, schedule)):
            print("Wrote", path)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
