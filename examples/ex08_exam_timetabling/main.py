"""Schedule a school's exams: fewest slots first, then a kinder week.

Run from the repository root:

    uv run python -m examples.ex08_exam_timetabling.main
    uv run python -m examples.ex08_exam_timetabling.main --plot   # ~1 min
"""

import argparse
import math
from pathlib import Path

from examples.common.report import format_table

from .bounds import dsatur, max_clique
from .check import (
    chromatic_number,
    coloring_errors,
    is_clique,
    num_colors,
    penalty_lower_bound,
    student_penalties,
    timetable_errors,
)
from .data import DEPARTMENTS, ExamData, Graph, mycielski_graph, queen_graph, school
from .model import Coloring, Timetable, color_boolean, color_integer, spread_exams

FIGURES = Path(__file__).parent / "figures"
DAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri")
VARIANTS = (
    ("integer", False),
    ("integer", True),
    ("boolean", False),
    ("boolean", True),
)


def part1(graph: Graph) -> tuple[Coloring, list[str]]:
    """Find the fewest slots and print the bounds. Return check errors."""
    clique = max_clique(graph)
    greedy = num_colors(dsatur(graph))
    density = 2 * len(graph.edges) / (len(graph.nodes) * (len(graph.nodes) - 1))
    print(
        f"Conflict graph: {len(graph.nodes)} exams, {len(graph.edges)} conflicting "
        f"pairs ({density:.0%} of all pairs)\n"
    )
    print(f"Lower bound, largest clique: {len(clique)} exams")
    print(f"  {', '.join(clique)}")
    print(f"Upper bound, DSATUR greedy:  {greedy} slots")

    coloring = color_boolean(graph)
    status = "optimal" if coloring.optimal else "best found"
    print(
        f"CP-SAT:                      {coloring.num_colors} slots ({status}, "
        f"bound {coloring.lower_bound}, {coloring.seconds:.2f} s)"
    )

    errors = coloring_errors(graph, coloring.color)
    if not is_clique(graph, clique):
        errors.append("the clique bound is not a clique")
    exact = chromatic_number(graph)
    if exact != coloring.num_colors:
        errors.append(
            f"backtracking needs {exact} slots, CP-SAT used {coloring.num_colors}"
        )
    return coloring, errors


def part2(data: ExamData, greedy: dict[str, int], time_limit: float):
    """Spread the exams over the week and print the result. Return errors."""
    week = spread_exams(data, time_limit=time_limit)
    naive = student_penalties(data, greedy)
    spread = student_penalties(data, week.slot)
    floor = penalty_lower_bound(data)
    status = "optimal" if week.optimal else f"stopped after {week.seconds:.0f} s"
    print("Penalty (3 per same-day pair, 1 per next-day pair, per student):")
    rows = [
        ("DSATUR slots, in order", sum(naive)),
        (f"CP-SAT spread ({status})", week.penalty),
        ("CP-SAT lower bound", week.lower_bound),
        ("Sum of best cases per student", floor),
    ]
    print(format_table(["timetable", "penalty"], rows))
    print(
        f"\nSo the best possible timetable has a penalty between {week.lower_bound:,}"
        f" and {week.penalty:,}.\n"
    )

    rows = [
        (label, _count_students(data, slots, 2), _count_students(data, slots, 1))
        for label, slots in (
            ("DSATUR, in order", greedy),
            ("CP-SAT spread", week.slot),
        )
    ]
    print(
        format_table(
            ["students with ...", "2 exams on 1 day", "exams on 2 days in a row"], rows
        )
    )
    print()
    print_week(data, week)

    errors = timetable_errors(data, week.slot)
    if sum(spread) != week.penalty:
        errors.append(f"penalty {week.penalty} != recomputed {sum(spread)}")
    if week.lower_bound > week.penalty:
        errors.append("lower bound above the penalty")
    return week, errors


def _count_students(data: ExamData, slot: dict[str, int], kind: int) -> int:
    """Count students with two exams on one day (kind 2) or on next days (1)."""
    count = 0
    for exams in data.enrollments:
        days = sorted(slot[e] // data.slots_per_day for e in exams)
        gaps = [b - a for a, b in zip(days, days[1:], strict=False)]
        if (kind == 2 and 0 in gaps) or (kind == 1 and 0 not in gaps and 1 in gaps):
            count += 1
    return count


def print_week(data: ExamData, week: Timetable) -> None:
    """Print the timetable as a day-by-slot grid."""
    rows = []
    for d in range(data.days):
        for k, label in enumerate(("am", "pm")[: data.slots_per_day]):
            s = d * data.slots_per_day + k
            exams = [e for e in data.exams if week.slot[e] == s]
            rows.append((DAY_NAMES[d] if k == 0 else "", label, ", ".join(exams)))
    print(format_table(["day", "slot", "exams"], rows))


# ------------------------------------------------------------------ plots ---


def benchmark(time_limit: float) -> list[tuple[str, str, Coloring]]:
    """Solve three graphs with all four model variants."""
    runs = []
    for graph in (school().conflict_graph(), queen_graph(6), mycielski_graph(5)):
        for encoding, symmetry in VARIANTS:
            solve = color_integer if encoding == "integer" else color_boolean
            result = solve(graph, symmetry_breaking=symmetry, time_limit=time_limit)
            runs.append(
                (graph.name, f"{encoding}{' + symmetry' if symmetry else ''}", result)
            )
    return runs


def plot_benchmark(runs, time_limit: float) -> Path:
    """Plot the time to prove the optimum, per graph and model variant."""
    from matplotlib.patches import Patch

    from examples.common.plotting import COLORS, new_figure, save

    graphs = list(dict.fromkeys(g for g, _, _ in runs))
    variants = list(dict.fromkeys(v for _, v, _ in runs))
    fig, ax = new_figure(8, 4)
    width = 0.8 / len(variants)
    for k, variant in enumerate(variants):
        for j, graph in enumerate(graphs):
            (r,) = [r for g, v, r in runs if g == graph and v == variant]
            x = j + (k - (len(variants) - 1) / 2) * width
            ax.bar(
                x,
                r.seconds,
                width,
                color=COLORS[k],
                hatch=None if r.optimal else "//",
                edgecolor="white" if r.optimal else "black",
            )
    ax.set_yscale("log")
    ax.set_ylim(0.001, time_limit * 2)
    ax.axhline(time_limit, color="black", lw=1, ls="--")
    ax.set_xticks(range(len(graphs)), graphs)
    ax.set_ylabel("seconds to prove the optimum (log scale)")
    ax.set_title(f"Model choice matters (time limit {time_limit:g} s, 8 workers)")
    handles = [Patch(color=COLORS[k], label=v) for k, v in enumerate(variants)]
    handles.append(Patch(fc="white", ec="black", hatch="//", label="not proven"))
    fig.legend(handles=handles, ncols=5, loc="outside lower center", fontsize=8)
    return save(fig, FIGURES / "benchmark.png")


def plot_conflict_graph(graph: Graph, coloring: Coloring) -> Path:
    """Draw the exams on a circle by department, colored by slot."""
    import matplotlib.pyplot as plt

    from examples.common.plotting import new_figure, save

    fig, ax = new_figure(7.5, 7.5)
    n = len(graph.nodes)
    pos = {
        e: (math.cos(2 * math.pi * k / n), math.sin(2 * math.pi * k / n))
        for k, e in enumerate(graph.nodes)
    }
    for a, b in graph.edges:
        (x1, y1), (x2, y2) = pos[a], pos[b]
        ax.plot([x1, x2], [y1, y2], color="0.6", lw=0.4, alpha=0.5, zorder=1)
    palette = plt.get_cmap("tab10")
    for e, (x, y) in pos.items():
        ax.scatter(x, y, s=140, color=palette(coloring.color[e]), zorder=2, ec="white")
        angle = math.degrees(math.atan2(y, x))
        flip = 90 < angle % 360 < 270
        ax.text(
            1.09 * x,
            1.09 * y,
            e,
            rotation=angle + 180 if flip else angle,
            ha="right" if flip else "left",
            va="center",
            rotation_mode="anchor",
            fontsize=7,
        )
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"{len(graph.edges)} conflicts, {coloring.num_colors} slots: "
        "no line joins two dots of one color"
    )
    return save(fig, FIGURES / "conflict_graph.png")


def plot_week(data: ExamData, week: Timetable) -> Path:
    """Draw the timetable grid with exams colored by department."""
    from matplotlib.patches import Patch

    from examples.common.plotting import COLORS, new_figure, save

    dept_of = {c: d for d, courses in DEPARTMENTS.items() for c in courses}
    dept_color = {d: COLORS[k] for k, d in enumerate(DEPARTMENTS)}
    fig, ax = new_figure(10, 4.2)
    per_day = data.slots_per_day
    for s in range(data.num_slots):
        d, k = divmod(s, per_day)
        exams = [e for e in data.exams if week.slot[e] == s]
        for j, e in enumerate(exams):
            ax.text(
                d + 0.04,
                -(k + 0.12 + j * 0.105),
                e,
                fontsize=7.5,
                va="top",
                color="white",
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "fc": dept_color[dept_of[e]],
                    "ec": "none",
                },
            )
    for d in range(data.days + 1):
        ax.axvline(d, color="0.7", lw=1)
    for k in range(per_day + 1):
        ax.axhline(-k, color="0.7", lw=1)
    ax.set_xlim(0, data.days)
    ax.set_ylim(-per_day, 0)
    ax.set_xticks([d + 0.5 for d in range(data.days)], DAY_NAMES[: data.days])
    ax.xaxis.tick_top()
    ax.set_yticks(
        [-k - 0.5 for k in range(per_day)], ["morning", "afternoon"][:per_day]
    )
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    handles = [Patch(color=c, label=d) for d, c in dept_color.items()]
    ax.legend(
        handles=handles,
        ncols=6,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        fontsize=8,
        frameon=False,
    )
    ax.set_title(f"Exam week: penalty {week.penalty:,}", pad=24)
    return save(fig, FIGURES / "exam_week.png")


def main() -> None:
    """Run both parts, verify, report, and optionally plot."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plot", action="store_true", help="write PNG figures")
    parser.add_argument("--time-limit", type=float, default=20.0, help="Part 2 seconds")
    args = parser.parse_args()

    data = school()
    graph = data.conflict_graph()
    print(f"{len(data.exams)} exams, {len(data.enrollments)} students\n")
    print("Part 1: the fewest slots\n")
    coloring, errors = part1(graph)

    week_size = f"{data.days} days x {data.slots_per_day} slots"
    print(f"\nPart 2: spread the exams over {week_size}\n")
    # Baseline: the DSATUR slots in order (slot 0 = Monday morning, ...).
    # Unlike a CP-SAT coloring, it is the same on every run.
    week, more = part2(data, dsatur(graph), args.time_limit)
    errors += more

    if errors:
        print("\nVerification FAILED:", *errors, sep="\n  ")
    else:
        print(
            "\nVerification: both timetables have no clashes; backtracking confirms"
            f"\nthat {coloring.num_colors} slots are the minimum."
        )

    if args.plot:
        limit = 10.0
        runs = benchmark(limit)
        print()
        print(
            format_table(
                ["graph", "model", "colors", "bound", "seconds", "proven"],
                [
                    (g, v, r.num_colors, r.lower_bound, r.seconds, str(r.optimal))
                    for g, v, r in runs
                ],
            )
        )
        for path in (
            plot_conflict_graph(graph, coloring),
            plot_week(data, week),
            plot_benchmark(runs, limit),
        ):
            print("Wrote", path)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
