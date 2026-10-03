"""Schedule a ten-club league with as few breaks as possible.

Run from the repository root:

    uv run python -m examples.ex16_sports_scheduling.main
    uv run python -m examples.ex16_sports_scheduling.main --plot
    uv run python -m examples.ex16_sports_scheduling.main --benchmark
"""

import argparse
from pathlib import Path

from examples.common.report import format_table

from .check import patterns, schedule_errors, team_breaks, total_breaks
from .circle import (
    circle_method,
    count_breaks,
    home_away,
    inverted,
    lower_bound,
    mirrored,
)
from .data import League, plain_league, regional_league
from .model import LeagueSchedule, solve_league, solve_monolithic

FIGURES = Path(__file__).parent / "figures"


def print_theory(n: int) -> list[str]:
    """Compare the proven bounds with the circle-method schedules."""
    first = circle_method(n)
    rows, errors = [], []
    for label, scheme, season in (
        ("single round robin", "single", first),
        ("double, mirrored", "mirrored", mirrored(first)),
        ("double, inverted", "phased", inverted(first)),
    ):
        built = sum(count_breaks(p) for p in home_away(season, n))
        rows.append((label, len(season), lower_bound(n, scheme), built))
        if built != lower_bound(n, scheme):
            errors.append(f"circle method, {label}: {built} breaks")
    print(f"Theory for {n} teams: fewest possible breaks, and the circle method\n")
    print(format_table(["schedule", "rounds", "lower bound", "circle method"], rows))
    return errors


def print_season(league: League, season: LeagueSchedule) -> None:
    """Print the draw, the fixtures, and each team's home/away pattern."""
    short = {t.name: t.short for t in league.teams}
    bound = lower_bound(len(league.teams), "phased")
    print(f"\nThe {league.name}: {season.breaks} breaks", end="")
    print(f" (bound {bound}, solved in {season.wall_time:.2f} s)")
    if season.globally_optimal:
        print("This meets the lower bound, so no schedule at all has fewer breaks.")

    print("\nFixtures (home team first)")
    rows = [
        (r, "  ".join(f"{short[h]}-{short[a]}" for h, a in games))
        for r, games in enumerate(season.rounds, start=1)
    ]
    print(format_table(["round", "games"], rows))

    print("\nHome/away patterns (| = turn of the season)")
    half = len(league.teams) - 1
    breaks = team_breaks(league, season.rounds)
    rows = [
        (
            t.name,
            season.draw[t.name],
            (p := patterns(league, season.rounds)[t.name])[:half] + "|" + p[half:],
            len(breaks[t.name]),
            ", ".join(map(str, breaks[t.name])) or "-",
        )
        for t in league.teams
    ]
    print(format_table(["team", "slot", "pattern", "breaks", "in rounds"], rows))


def benchmark(sizes=(6, 8, 10, 12), time_limit: float = 10.0) -> list[tuple]:
    """Time the direct model against the template model (no local rules)."""
    rows = []
    for n in sizes:
        plain = solve_monolithic(n, "phased", add_bound=False, time_limit=time_limit)
        bounded = solve_monolithic(n, "phased", add_bound=True, time_limit=time_limit)
        template = solve_league(plain_league(n), time_limit=time_limit)
        rows.append(
            (
                n,
                lower_bound(n, "phased"),
                plain.breaks,
                plain.status == "OPTIMAL",
                plain.wall_time,
                bounded.breaks,
                bounded.status == "OPTIMAL",
                bounded.wall_time,
                template.breaks,
                template.wall_time,
            )
        )
    for n in (16, 20):
        template = solve_league(plain_league(n), time_limit=time_limit)
        rows.append(
            (
                n,
                lower_bound(n, "phased"),
                None,
                None,
                None,
                None,
                None,
                None,
                template.breaks,
                template.wall_time,
            )
        )
    return rows


def plot_fixtures(league: League, season: LeagueSchedule) -> Path:
    """Teams x rounds grid: opponent, venue, and breaks."""
    from matplotlib.patches import Rectangle

    from examples.common.plotting import COLORS, new_figure, save

    short = {t.name: t.short for t in league.teams}
    names = [t.name for t in league.teams]
    num_rounds = len(season.rounds)
    breaks = team_breaks(league, season.rounds)
    fig, ax = new_figure(12, 5.2)
    for r, games in enumerate(season.rounds):
        for h, a in games:
            for team, other, at_home in ((h, a, True), (a, h, False)):
                y = names.index(team)
                color = COLORS[0] if at_home else "#F2F2F2"
                ax.add_patch(Rectangle((r, y), 1, 1, fc=color, ec="white", lw=1.5))
                ax.text(
                    r + 0.5,
                    y + 0.5,
                    short[other],
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if at_home else "0.25",
                )
    for y, team in enumerate(names):
        for rnd in breaks[team]:
            ax.add_patch(
                Rectangle((rnd - 1, y), 1, 1, fill=False, ec=COLORS[3], lw=2.5)
            )
    half = len(names) - 1
    ax.axvline(half, color="black", lw=2)
    ax.set_xlim(0, num_rounds)
    ax.set_ylim(len(names), 0)
    ax.set_xticks(
        [r + 0.5 for r in range(num_rounds)], [str(r + 1) for r in range(num_rounds)]
    )
    ax.set_yticks([y + 0.5 for y in range(len(names))], names)
    ax.tick_params(length=0)
    ax.grid(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("round")
    ax.set_title(
        f"Season fixtures: {season.breaks} breaks (blue = home, gray = away, "
        "red box = break, black line = turn of the season)"
    )
    return save(fig, FIGURES / "fixtures.png")


def plot_benchmark(rows: list[tuple], time_limit: float) -> Path:
    """Solve time versus league size for the direct and the template model."""
    from examples.common.plotting import COLORS, new_figure, save

    fig, ax = new_figure(7, 4.4)
    series = (
        ("direct model", 4, 3, COLORS[3]),
        ("direct model + bound", 7, 6, COLORS[1]),
    )
    for label, t_col, opt_col, color in series:
        pts = [(r[0], r[t_col], r[opt_col]) for r in rows if r[t_col] is not None]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", color=color, label=label)
        for n, t, ok in pts:
            ax.plot(n, t, "o" if ok else "x", color=color, ms=8, mew=2)
    ax.plot(
        [r[0] for r in rows],
        [r[9] for r in rows],
        "-o",
        color=COLORS[0],
        label="template model",
    )
    ax.axhline(time_limit, color="0.5", ls="--", lw=1)
    ax.text(rows[0][0], time_limit * 1.2, f"time limit {time_limit:g} s", color="0.4")
    ax.set_yscale("log")
    ax.set_xticks([r[0] for r in rows])
    ax.set_xlabel("number of teams")
    ax.set_ylabel("seconds")
    ax.set_title(
        "Fewest breaks for a phased double round robin\n"
        "(o = proven optimal, x = stopped at the time limit)"
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    return save(fig, FIGURES / "benchmark.png")


def main() -> None:
    """Parse arguments, solve, verify, and report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plot", action="store_true", help="write PNG figures")
    parser.add_argument(
        "--benchmark", action="store_true", help="compare the two models"
    )
    args = parser.parse_args()

    league = regional_league()
    errors = print_theory(len(league.teams))
    season = solve_league(league)
    print_season(league, season)

    errors += schedule_errors(league, season.rounds)
    if total_breaks(league, season.rounds) != season.breaks:
        errors.append("the reported break count does not match the schedule")
    if season.breaks < lower_bound(len(league.teams), "phased"):
        errors.append(
            "fewer breaks than the proven bound: the checker or the bound is wrong"
        )
    if not season.globally_optimal:
        errors.append(
            "the season does not reach the lower bound; it may not be optimal"
        )

    if args.benchmark:
        time_limit = 10.0
        print(f"\nBenchmark: no local rules, time limit {time_limit:g} s per solve\n")
        rows = benchmark(time_limit=time_limit)
        header = [
            "teams",
            "bound",
            "direct",
            "opt",
            "s",
            "+bound",
            "opt",
            "s",
            "template",
            "s",
        ]
        shown = [
            tuple("" if v is None else str(v) if isinstance(v, bool) else v for v in r)
            for r in rows
        ]
        print(format_table(header, shown))
        if args.plot:
            print("Wrote", plot_benchmark(rows, time_limit))

    if errors:
        print("\nVerification FAILED:", *errors, sep="\n  ")
    else:
        print(
            "\nVerification: the season obeys every rule, its breaks match an"
            "\nindependent count, and the count equals the proven lower bound."
        )
    if args.plot:
        print("Wrote", plot_fixtures(league, season))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
