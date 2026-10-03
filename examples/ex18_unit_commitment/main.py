"""Plan tomorrow's power plants hour by hour (unit commitment).

Run from the repository root:

    uv run python -m examples.ex18_unit_commitment.main
    uv run python -m examples.ex18_unit_commitment.main --plot
    uv run python -m examples.ex18_unit_commitment.main --benchmark
"""

import argparse
import time
from pathlib import Path

from ortools.math_opt.python import mathopt

from examples.common.report import format_table

from .check import cost_breakdown, feasibility_errors, starts_and_stops
from .data import GridData, cloudy_day, large_grid, sunny_day
from .model import (
    STRONG,
    WEAK,
    Prices,
    Schedule,
    hourly_prices,
    lp_bound,
    solve_commitment,
)

FIGURES = Path(__file__).parent / "figures"


def verify(data: GridData, s: Schedule) -> list[str]:
    """Run the independent checks on one schedule."""
    errors = feasibility_errors(data, s.on, s.power, s.solar_used, s.shed)
    total = cost_breakdown(data, s.on, s.power, s.shed)["total"]
    if abs(total - s.cost) > 1e-6 * s.cost:
        errors.append(f"{data.name}: cost {s.cost:.2f} != recomputed {total:.2f}")
    return errors


def print_schedule(data: GridData, s: Schedule) -> None:
    """Print the on/off grid and a summary per plant."""
    print(f"On/off plan, '{data.name}' (# = on; hours 0-23)\n")
    rows = []
    for g in data.units:
        on, power = s.on[g.name], s.power[g.name]
        rows.append(
            (
                g.name,
                g.kind,
                "".join("#" if x else "." for x in on),
                len(starts_and_stops(g, on)[0]),
                round(sum(power)),
                f"{max(0.0, sum(power)) / (g.p_max * data.hours):.0%}",
            )
        )
    header = ["plant", "type", "on/off by hour", "starts", "MWh", "load factor"]
    print(format_table(header, rows))


def print_costs(data: GridData, s: Schedule) -> None:
    """Print the cost split and the solver's proof."""
    parts = cost_breakdown(data, s.on, s.power, s.shed)
    rows = [(k, round(v)) for k, v in parts.items()]
    print(format_table(["cost part", "$"], rows))
    print(
        f"\nSolver: cost ${s.cost:,.0f}, bound ${s.bound:,.0f}, gap {s.gap:.4%}, "
        f"{'proven optimal' if s.proven_optimal else 'not proven'} "
        f"in {s.solve_time:.2f} s"
    )


def print_prices(data: GridData, s: Schedule, prices: Prices) -> None:
    """Print the hourly balance and the marginal prices."""
    rows = []
    for t in range(data.hours):
        curtailed = data.solar[t] - s.solar_used[t]
        rows.append(
            (
                t,
                data.demand[t],
                round(s.solar_used[t]),
                round(curtailed),
                round(data.demand[t] - s.solar_used[t]),
                round(prices.energy[t], 2),
            )
        )
    header = ["hour", "demand MW", "solar MW", "curtailed", "plants MW", "price $/MWh"]
    print(format_table(header, rows))
    for dam, value in prices.water_value.items():
        print(f"\nWater value of {dam}: ${value:.2f} per MWh.")


def compare_formulations(data: GridData) -> list[tuple]:
    """Return LP bound, MIP time, and proof for both formulations."""
    rows = []
    for formulation in (WEAK, STRONG):
        bound = lp_bound(data, formulation)
        s = solve_commitment(data, formulation)
        rows.append(
            (
                formulation,
                round(bound),
                f"{(s.cost - bound) / s.cost:.2%}",
                round(s.cost),
                str(s.proven_optimal),
            )
        )
    return rows


def compare_days(days: list[tuple[GridData, Schedule, Prices]]) -> None:
    """Print one row per day: cost, solar, prices, gas use."""
    rows = []
    for data, s, prices in days:
        gas = sum(
            sum(s.power[g.name])
            for g in data.units
            if g.kind.startswith(("gas", "diesel"))
        )
        rows.append(
            (
                data.name,
                round(s.cost),
                round(sum(s.solar_used)),
                round(sum(data.solar) - sum(s.solar_used)),
                round(gas),
                round(
                    sum(p * d for p, d in zip(prices.energy, data.demand, strict=True))
                    / sum(data.demand),
                    2,
                ),
                f"{min(prices.energy):.0f}-{max(prices.energy):.0f}",
            )
        )
    header = [
        "day",
        "cost $",
        "solar MWh",
        "curtailed",
        "gas+diesel MWh",
        "avg price",
        "price range",
    ]
    print(format_table(header, rows))


def benchmark(time_limit: float = 30.0) -> list[tuple]:
    """Compare weak and strong formulations on larger random fleets."""
    rows = []
    for n_units, days, seed in ((12, 1, 1), (30, 2, 1), (40, 2, 0)):
        data = large_grid(n_units, days, seed)
        for formulation in (WEAK, STRONG):
            bound = lp_bound(data, formulation)
            start = time.perf_counter()
            s = solve_commitment(data, formulation, time_limit=time_limit)
            seconds = time.perf_counter() - start
            rows.append(
                (
                    f"{n_units} units x {24 * days} h",
                    formulation,
                    f"{(s.cost - bound) / s.cost:.2%}",
                    seconds,
                    f"{s.gap:.3%}",
                    str(s.proven_optimal),
                )
            )
    return rows


def plot_dispatch(days: list[tuple[GridData, Schedule, Prices]]) -> Path:
    """Stacked output by plant type for each day, with demand on top."""
    from examples.common.plotting import COLORS, new_figure, save

    kinds = ["nuclear", "coal", "hydro", "gas CCGT", "gas peaker", "diesel"]
    colors = {
        "nuclear": COLORS[4],
        "coal": "0.35",
        "hydro": COLORS[5],
        "gas CCGT": COLORS[1],
        "gas peaker": COLORS[3],
        "diesel": COLORS[7],
        "solar": COLORS[6],
    }
    fig, axes = new_figure(11, 4.2, ncols=len(days), sharey=True)
    for ax, (data, s, _) in zip(axes, days, strict=True):
        hours = list(range(data.hours))
        layers = [
            [sum(s.power[g.name][t] for g in data.units if g.kind == k) for t in hours]
            for k in kinds
        ]
        layers.append(list(s.solar_used))
        x = [t + 0.5 for t in hours]
        ax.stackplot(
            x,
            *layers,
            labels=[*kinds, "solar"],
            step=None,
            colors=[colors[k] for k in [*kinds, "solar"]],
            alpha=0.9,
        )
        ax.plot(x, data.demand, color="black", lw=2, label="demand")
        net = [d - so for d, so in zip(data.demand, data.solar, strict=True)]
        ax.plot(x, net, color="black", lw=1, ls="--", label="demand minus solar")
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 3))
        ax.set_xlabel("hour of day")
        ax.set_title(f"{data.name}: ${s.cost:,.0f}")
    axes[0].set_ylabel("MW")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    return save(fig, FIGURES / "dispatch.png")


def plot_commitment(data: GridData, s: Schedule) -> Path:
    """Draw the on/off grid: one row per plant, one column per hour."""
    from matplotlib.colors import ListedColormap

    from examples.common.plotting import COLORS, new_figure, save

    fig, ax = new_figure(9, 3.6)
    grid = [[s.on[g.name][t] for t in range(data.hours)] for g in data.units]
    ax.imshow(
        grid,
        aspect="auto",
        cmap=ListedColormap(["0.93", COLORS[0]]),
        extent=(0, data.hours, len(data.units), 0),
    )
    for k, g in enumerate(data.units):
        for t in range(data.hours):
            if s.on[g.name][t]:
                ax.text(
                    t + 0.5,
                    k + 0.5,
                    f"{s.power[g.name][t]:.0f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white",
                )
        for t in starts_and_stops(g, s.on[g.name])[0]:
            ax.plot(t + 0.08, k + 0.5, ">", color=COLORS[1], ms=6)
    ax.set_yticks(
        [k + 0.5 for k in range(len(data.units))], [g.name for g in data.units]
    )
    ax.set_xticks(range(0, 25, 3))
    ax.grid(False)
    ax.set_xlabel("hour of day (numbers: MW; orange arrow: start)")
    ax.set_title(f"Which plants run, '{data.name}'")
    return save(fig, FIGURES / "commitment.png")


def plot_prices(days: list[tuple[GridData, Schedule, Prices]]) -> Path:
    """Hourly marginal price for each day."""
    from examples.common.plotting import COLORS, new_figure, save

    fig, ax = new_figure(8, 3.4)
    for k, (data, _, prices) in enumerate(days):
        ax.step(
            range(data.hours + 1),
            [*prices.energy, prices.energy[-1]],
            where="post",
            color=COLORS[k],
            lw=2,
            label=data.name,
        )
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xlabel("hour of day")
    ax.set_ylabel("$ per MWh")
    ax.set_title("Marginal price of electricity (LP with fixed on/off plan)")
    ax.legend(loc="lower left")
    return save(fig, FIGURES / "prices.png")


def main() -> None:
    """Solve both days, verify, and report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plot", action="store_true", help="write PNG figures")
    parser.add_argument(
        "--benchmark", action="store_true", help="weak vs strong on big fleets"
    )
    args = parser.parse_args()

    days, errors = [], []
    for data in (sunny_day(), cloudy_day()):
        s = solve_commitment(data)
        days.append((data, s, hourly_prices(data, s)))
        errors += verify(data, s)

    data, s, prices = days[0]
    print_schedule(data, s)
    print()
    print_costs(data, s)
    print("\nOther solvers on the same model:")
    rows = []
    for solver in (mathopt.SolverType.GSCIP, mathopt.SolverType.CP_SAT):
        other = solve_commitment(data, solver=solver)
        rows.append(
            (
                solver.name,
                round(other.cost),
                str(other.proven_optimal),
                other.solve_time,
            )
        )
        if abs(other.cost - s.cost) > 1e-6 * s.cost:
            errors.append(f"{solver.name} cost {other.cost} != HiGHS cost {s.cost}")
    print(format_table(["solver", "cost $", "optimal", "seconds"], rows))

    print("\nWeak vs strong minimum up/down rows (LP relaxation):\n")
    header = ["formulation", "LP bound $", "LP gap", "MIP cost $", "optimal"]
    print(format_table(header, compare_formulations(data)))

    print("\n" + "=" * 72 + "\n")
    print_prices(data, s, prices)
    print("\n" + "=" * 72 + "\n")
    print("Sunny day vs cloudy day (30% of the solar power):\n")
    compare_days(days)

    if args.benchmark:
        print("\n" + "=" * 72 + "\n")
        print("Weak vs strong on random fleets (time limit 30 s):\n")
        header = [
            "instance",
            "formulation",
            "LP gap",
            "seconds",
            "final gap",
            "optimal",
        ]
        print(format_table(header, benchmark()))

    if errors:
        print("\nVerification FAILED:", *errors, sep="\n  ")
    else:
        print(
            "\nVerification: both schedules meet every hourly rule, ramp, and"
            "\nminimum up/down time; the costs match an independent recount;"
            "\nHiGHS, SCIP, and CP-SAT agree on the optimum."
        )
    if args.plot:
        for path in (plot_dispatch(days), plot_commitment(data, s), plot_prices(days)):
            print("Wrote", path)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
