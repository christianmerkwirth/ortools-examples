"""Assign repair technicians to jobs: first one job each, then a full day.

Run from the repository root:

    uv run python -m examples.ex04_assignment.main
    uv run python -m examples.ex04_assignment.main --plot
    uv run python -m examples.ex04_assignment.main --benchmark
"""

import argparse
import time
from pathlib import Path

from examples.common.report import format_table

from .check import certificate_errors, feasibility_errors, hungarian, plan_minutes
from .data import AssignmentData, full_day, morning, random_square
from .model import Plan, solve_day, solve_morning, solve_morning_cp_sat

FIGURES = Path(__file__).parent / "figures"


def report_morning(data: AssignmentData) -> tuple[Plan, list[str]]:
    """Solve Part A with both solvers, print the plan, and verify it."""
    plan = solve_morning(data)
    cp_plan = solve_morning_cp_sat(data)
    techs = {t.name: t for t in data.technicians}
    print(f"Part A: morning, one job each. Total: {plan.minutes} minutes\n")
    rows = []
    for job in data.jobs:
        (name,) = plan.crew[job.name]
        rows.append((name, job.name, job.trade, data.cost(techs[name], job)))
    rows.sort()
    print(format_table(["technician", "job", "trade", "minutes"], rows))
    print(
        f"\nLinear sum assignment: {plan.minutes} min."
        f"  CP-SAT: {cp_plan.minutes} min (proven optimal: {cp_plan.proven_optimal})."
    )

    errors = feasibility_errors(data, plan.crew, one_job_each=True)
    if plan_minutes(data, plan.crew) != plan.minutes:
        errors.append("reported minutes do not match the plan")
    matrix = data.cost_matrix()
    _, _, u, v = hungarian(matrix)
    errors += certificate_errors(matrix, plan.minutes, u, v)
    if cp_plan.minutes != plan.minutes:
        errors.append(
            f"CP-SAT found {cp_plan.minutes}, assignment solver {plan.minutes}"
        )
    return plan, errors


def report_day(data: AssignmentData) -> tuple[Plan, list[str]]:
    """Solve Part B, print each technician's day, and show what each rule costs."""
    plan = solve_day(data)
    jobs = {j.name: j for j in data.jobs}
    print(f"Part B: full day, {len(data.jobs)} jobs. Total: {plan.minutes} minutes\n")
    rows = []
    for tech in data.technicians:
        mine = plan.jobs_of(tech.name)
        load = sum(data.cost(tech, jobs[j]) for j in mine)
        rows.append((tech.name, load, tech.shift, ", ".join(mine)))
    print(format_table(["technician", "minutes", "shift", "jobs"], rows))

    print("\nWhat each rule costs")
    variants = [
        ("no shift limits, no team jobs", dict(use_shifts=False, use_teams=False)),
        ("+ team jobs need two people", dict(use_shifts=False, use_teams=True)),
        ("+ shift limits (the real plan)", dict(use_shifts=True, use_teams=True)),
    ]
    rows, previous = [], None
    for label, kw in variants:
        minutes = solve_day(data, **kw).minutes
        rows.append(
            (label, minutes, "" if previous is None else f"+{minutes - previous}")
        )
        previous = minutes
    print(format_table(["rules", "minutes", "extra"], rows))

    errors = feasibility_errors(data, plan.crew)
    if plan_minutes(data, plan.crew) != plan.minutes:
        errors.append("reported minutes do not match the plan")
    if not plan.proven_optimal:
        errors.append("CP-SAT did not prove optimality")
    return plan, errors


def benchmark(sizes=(50, 100, 200, 400)) -> None:
    """Time both solvers on random square instances."""
    rows = []
    for n in sizes:
        data = random_square(n, seed=n)
        start = time.perf_counter()
        a = solve_morning(data)
        t_lsa = time.perf_counter() - start
        start = time.perf_counter()
        b = solve_morning_cp_sat(data, time_limit=60)
        t_cp = time.perf_counter() - start
        rows.append(
            (
                n,
                a.minutes,
                f"{t_lsa:.3f}",
                b.minutes,
                f"{t_cp:.2f}",
                str(b.proven_optimal),
            )
        )
    print(
        format_table(
            [
                "n",
                "LSA minutes",
                "LSA s",
                "CP-SAT minutes",
                "CP-SAT s",
                "CP-SAT optimal",
            ],
            rows,
        )
    )


def plot_morning(data: AssignmentData, plan: Plan) -> Path:
    """Heatmap of the cost matrix with the chosen pairs outlined."""
    import numpy as np
    from matplotlib.patches import Rectangle

    from examples.common.plotting import COLORS, new_figure, save

    matrix = np.array(
        [[np.nan if c is None else c for c in row] for row in data.cost_matrix()],
        dtype=float,
    )
    fig, ax = new_figure(8, 4.8)
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=np.nanmax(matrix))
    ax.grid(False)
    tech_index = {t.name: k for k, t in enumerate(data.technicians)}
    for j, job in enumerate(data.jobs):
        (name,) = plan.crew[job.name]
        t = tech_index[name]
        ax.add_patch(
            Rectangle((j - 0.5, t - 0.5), 1, 1, fill=False, ec=COLORS[3], lw=3)
        )
    for t in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isnan(matrix[t, j]):
                ax.text(j, t, "–", ha="center", va="center", color="0.6")
            else:
                dark = matrix[t, j] > 0.6 * np.nanmax(matrix)
                ax.text(
                    j,
                    t,
                    f"{matrix[t, j]:.0f}",
                    ha="center",
                    va="center",
                    color="white" if dark else "black",
                    fontsize=9,
                )
    ax.set_xticks(
        range(len(data.jobs)), [j.name for j in data.jobs], rotation=35, ha="right"
    )
    ax.set_yticks(range(len(data.technicians)), [t.name for t in data.technicians])
    fig.colorbar(image, ax=ax, label="minutes (travel + work)")
    ax.set_title(f"Morning plan: {plan.minutes} minutes (boxed; – = not qualified)")
    return save(fig, FIGURES / "morning_matrix.png")


def plot_day(data: AssignmentData, plan: Plan) -> Path:
    """Each technician's day as stacked bars, with their shift limit."""
    from examples.common.plotting import COLORS, new_figure, save

    jobs = {j.name: j for j in data.jobs}
    trades = sorted({j.trade for j in data.jobs})
    color = {trade: COLORS[k] for k, trade in enumerate(trades)}
    fig, ax = new_figure(9, 5)
    for k, tech in enumerate(data.technicians):
        start = 0
        for name in plan.jobs_of(tech.name):
            job = jobs[name]
            minutes = data.cost(tech, job)
            ax.barh(k, minutes, left=start, color=color[job.trade], ec="white")
            label = name + (" (team)" if job.team_size > 1 else "")
            if minutes >= 45:
                ax.text(start + 4, k, label, va="center", fontsize=7, color="white")
            start += minutes
        ax.plot([tech.shift, tech.shift], [k - 0.4, k + 0.4], color="black", lw=2)
    ax.set_yticks(range(len(data.technicians)), [t.name for t in data.technicians])
    ax.invert_yaxis()
    ax.set_xlabel("minutes (black bar = end of shift)")
    for trade in trades:
        ax.barh(0, 0, color=color[trade], label=trade)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncols=len(trades),
        frameon=False,
    )
    ax.set_xlim(0, 400)
    ax.set_title(f"Full-day plan: {plan.minutes} minutes in total")
    return save(fig, FIGURES / "day_plan.png")


def main() -> None:
    """Solve both parts, verify them, and print reports."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plot", action="store_true", help="write PNG figures")
    parser.add_argument("--benchmark", action="store_true", help="time both solvers")
    args = parser.parse_args()

    morning_data, day_data = morning(), full_day()
    morning_plan, errors = report_morning(morning_data)
    print("\n" + "=" * 72 + "\n")
    day_plan, day_errors = report_day(day_data)
    errors += day_errors

    if errors:
        print("\nVerification FAILED:", *errors, sep="\n  ")
    else:
        print(
            "\nVerification: both plans meet every rule. The morning plan is"
            "\nproven optimal by dual potentials; the day plan by CP-SAT."
        )
    if args.benchmark:
        print()
        benchmark()
    if args.plot:
        for path in (
            plot_morning(morning_data, morning_plan),
            plot_day(day_data, day_plan),
        ):
            print("Wrote", path)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
