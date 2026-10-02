"""Choose distribution centers and assign towns to them (facility location).

Run from the repository root:

    uv run python -m examples.ex06_facility_location.main
    uv run python -m examples.ex06_facility_location.main --plot
"""

import argparse
from pathlib import Path

from ortools.math_opt.python import mathopt

from examples.common.report import format_table

from .check import brute_force_optimum, feasibility_errors, plan_cost
from .data import LocationData, random_region, small_region, wholesaler
from .model import STRONG, WEAK, Plan, lp_bound, solve_location

FIGURES = Path(__file__).parent / "figures"


def print_plan(data: LocationData, plan: Plan) -> None:
    """Print the open sites, their loads, and the cost split."""
    fixed, transport = plan_cost(data, plan.open_sites, plan.assignment)
    print(f"Optimal weekly cost: ${plan.cost:,.2f}")
    print(f"  fixed cost of {len(plan.open_sites)} open DCs: ${fixed:,.2f}")
    print(f"  transport cost:            ${transport:,.2f}\n")
    rows = []
    for s in data.sites:
        if s.name not in plan.open_sites:
            continue
        towns = [c for c in data.customers if plan.assignment[c.name] == s.name]
        load = sum(c.demand for c in towns)
        rows.append(
            (
                s.name,
                len(towns),
                load,
                s.capacity,
                100 * load / s.capacity,
                s.fixed_cost,
            )
        )
    print(
        format_table(
            ["open site", "towns", "load", "capacity", "used %", "fixed $"], rows
        )
    )
    closed = [s.name for s in data.sites if s.name not in plan.open_sites]
    print(f"\nClosed: {', '.join(closed)}")


def print_formulations(data: LocationData, optimum: float) -> None:
    """Compare the weak and strong formulations: bounds, time, and nodes."""
    print("Weak vs strong formulation")
    rows = []
    for label, inst in (
        ("tight capacity", data),
        ("loose capacity", data.with_capacity_factor(10)),
    ):
        best = optimum if inst is data else solve_location(inst).cost
        for f in (WEAK, STRONG):
            bound = lp_bound(inst, f)
            gap = max(0.0, (best - bound) / best)  # Clip -0.0 from round-off.
            rows.append((label, f, bound, best, f"{gap:.1%}"))
    print(
        format_table(
            ["instance", "formulation", "LP bound", "MIP optimum", "root gap"], rows
        )
    )

    print()
    rows = []
    for solver in (mathopt.SolverType.HIGHS, mathopt.SolverType.GSCIP):
        for f in (WEAK, STRONG):
            p = solve_location(data, f, solver)
            rows.append((solver.name, f, p.cost, p.solve_time, p.nodes))
    print(format_table(["solver", "formulation", "optimum", "seconds", "nodes"], rows))


def print_large_run(time_limit: float = 10.0, gap: float = 0.01) -> Plan:
    """Solve a larger instance with a gap limit, and show both bounds."""
    data = random_region(25, 150, seed=6, name="large region")
    plan = solve_location(data, time_limit=time_limit, relative_gap=gap)
    print(f"Large instance: 25 sites, 150 towns. Stop at {gap:.0%} gap", end="")
    print(f" or after {time_limit:g} s.")
    rows = [
        ("best plan found (primal bound)", plan.cost),
        ("proven lower bound (dual bound)", plan.bound),
    ]
    print(format_table(["", "$ per week"], rows))
    if plan.proven_optimal:
        reason = f"OPTIMAL, which here means 'within {gap:.0%} of the best possible'"
    else:
        reason = "FEASIBLE: the time limit stopped the search"
    print(f"gap {plan.gap:.2%} after {plan.solve_time:.1f} s")
    print(f"termination: {reason}")
    return plan


def verify(data: LocationData, plan: Plan) -> list[str]:
    """Run the independent checks. Return a list of problems."""
    errors = feasibility_errors(data, plan.open_sites, plan.assignment)
    fixed, transport = plan_cost(data, plan.open_sites, plan.assignment)
    if abs(fixed + transport - plan.cost) > 1e-6 * plan.cost:
        errors.append(f"reported cost {plan.cost} != recomputed {fixed + transport}")
    if not plan.proven_optimal:
        errors.append("the solver did not prove optimality")

    small = small_region()
    exact, _ = brute_force_optimum(small)
    found = solve_location(small)
    if abs(found.cost - exact) > 1e-6 * exact:
        errors.append(f"small instance: MIP {found.cost} != brute force {exact}")
    return errors


def plot_map(data: LocationData, plan: Plan) -> Path:
    """Map of towns, open and closed sites, and who serves whom."""
    from examples.common.plotting import COLORS, new_figure, save

    fig, ax = new_figure(6.4, 6.0)
    sites = {s.name: s for s in data.sites}
    color = {name: COLORS[k % 6] for k, name in enumerate(plan.open_sites)}
    for c in data.customers:
        s = sites[plan.assignment[c.name]]
        ax.plot(
            [c.x, s.x], [c.y, s.y], color=color[s.name], lw=0.8, alpha=0.6, zorder=1
        )
    ax.scatter(
        [c.x for c in data.customers],
        [c.y for c in data.customers],
        s=[c.demand * 2 for c in data.customers],
        c=[color[plan.assignment[c.name]] for c in data.customers],
        edgecolors="white",
        linewidths=0.5,
        zorder=2,
    )
    for s in data.sites:
        is_open = s.name in plan.open_sites
        ax.scatter(
            s.x,
            s.y,
            marker="s",
            s=140,
            c=color[s.name] if is_open else "white",
            edgecolors="black",
            linewidths=1.5 if is_open else 1.0,
            zorder=3,
        )
        ax.annotate(
            s.name.removeprefix("site "),
            (s.x, s.y),
            xytext=(7, 5),
            textcoords="offset points",
            fontweight="bold" if is_open else "normal",
            color="black" if is_open else "0.45",
        )
    ax.set_xlim(-3, 103)
    ax.set_ylim(-3, 103)
    ax.set_aspect("equal")
    ax.set_xlabel("km")
    ax.set_ylabel("km")
    ax.set_title(
        f"{len(plan.open_sites)} of {len(data.sites)} sites open "
        "(squares; white = closed; dot size = demand)"
    )
    return save(fig, FIGURES / "map.png")


def plot_tradeoff(data: LocationData, plan: Plan) -> Path:
    """Weekly cost when exactly k sites must open, split into fixed and transport."""
    from examples.common.plotting import COLORS, new_figure, save

    ks, fixed, transport = [], [], []
    for k in range(5, 10):
        p = solve_location(data, force_open_count=k)
        f, t = plan_cost(data, p.open_sites, p.assignment)
        ks.append(k)
        fixed.append(f)
        transport.append(t)

    total = [f + t for f, t in zip(fixed, transport, strict=True)]
    fig, (top, bottom) = new_figure(6.4, 5.2, nrows=2, sharex=True)
    top.plot(ks, [v / 1000 for v in total], "o-", color=COLORS[0], lw=2)
    best = len(plan.open_sites)
    top.plot(best, plan.cost / 1000, "o", color=COLORS[3], ms=12, zorder=3)
    top.annotate(
        f"optimum: {best} DCs, ${plan.cost / 1000:.1f}k",
        (best, plan.cost / 1000),
        xytext=(0, -22),
        textcoords="offset points",
        ha="center",
        color=COLORS[3],
    )
    top.set_ylim(min(total) / 1000 - 2.5, max(total) / 1000 + 1)
    top.set_ylabel("total ($k per week)")
    top.set_title("More DCs: less driving, more fixed cost")

    bottom.plot(
        ks, [v / 1000 for v in transport], "o-", color=COLORS[0], label="transport"
    )
    bottom.plot(
        ks, [v / 1000 for v in fixed], "s-", color=COLORS[1], label="fixed (open DCs)"
    )
    bottom.set_ylabel("$k per week")
    bottom.set_xlabel("number of open DCs (forced)")
    bottom.set_xticks(ks)
    bottom.set_ylim(0, None)
    bottom.legend(loc="center right")
    return save(fig, FIGURES / "tradeoff.png")


def main() -> None:
    """Solve, compare formulations, run a time-limited solve, and verify."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plot", action="store_true", help="write PNG figures")
    args = parser.parse_args()

    data = wholesaler()
    plan = solve_location(data)
    print_plan(data, plan)
    print("\n" + "=" * 72 + "\n")
    print_formulations(data, plan.cost)
    print("\n" + "=" * 72 + "\n")
    print_large_run()

    errors = verify(data, plan)
    if errors:
        print("\nVerification FAILED:", *errors, sep="\n  ")
    else:
        print(
            "\nVerification: the plan is feasible, its cost checks out, and the"
            "\nsolver proved it optimal. On a small instance, the MIP matches a"
            "\nbrute-force search over all assignments."
        )
    if args.plot:
        for path in (plot_map(data, plan), plot_tradeoff(data, plan)):
            print("Wrote", path)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
