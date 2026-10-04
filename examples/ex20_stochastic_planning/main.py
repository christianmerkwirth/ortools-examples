"""Plan a season under uncertain demand: point forecast vs stochastic program.

Run from the repository root:

    uv run python -m examples.ex20_stochastic_planning.main
    uv run python -m examples.ex20_stochastic_planning.main --plot
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from examples.common.report import format_table

from .check import best_plan, plan_errors, profits
from .data import PlanningData, Scenarios, point_forecast, sample
from .model import Plan, Solution, average_plan, solve_each, solve_plan

FIGURES = Path(__file__).parent / "figures"
SPREAD = 0.3  # Main case: demand about +-30% around the forecast.
SPREADS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
N_TRAIN = 300  # Scenarios the solver sees.
N_TEST = 20_000  # Fresh scenarios to judge each plan ("the real future").
TRAIN_SEED, TEST_SEED = 1, 2


@dataclass
class Approach:
    """One way to make a plan, and how that plan does."""

    name: str
    plan: Plan | None  # None for perfect information: it is not one plan.
    promised: float  # What the method expects to earn.
    actual: np.ndarray | None  # Profit in each test scenario.
    solves: int
    seconds: float


def best_candidate(
    data: PlanningData, candidates: list[Solution], train: Scenarios
) -> Plan:
    """Score every what-if plan on all training scenarios; keep the best.

    This is the bridge between "rerun" and "reformulate": the runs propose
    plans, and a scenario test picks one. It is only as good as its
    candidates, though, and no single run looks for a hedge.
    """
    scores = [
        profits(data, c.plan.lines, c.plan.make, train).mean() for c in candidates
    ]
    return candidates[int(np.argmax(scores))].plan


def compare(data: PlanningData, spread: float) -> dict[str, Approach]:
    """Make a plan with each approach; play each out on the test scenarios."""
    train = sample(data, spread, N_TRAIN, TRAIN_SEED)
    test = sample(data, spread, N_TEST, TEST_SEED)

    def play(plan: Plan) -> np.ndarray:
        return profits(data, plan.lines, plan.make, test)

    out = {}
    start = time.perf_counter()
    det = solve_plan(data, point_forecast(data))
    out["point forecast"] = Approach(
        "point forecast", det.plan, det.promised, play(det.plan), 1, 0.0
    )
    out["point forecast"].seconds = time.perf_counter() - start

    start = time.perf_counter()
    runs = solve_each(data, train)
    what_if = time.perf_counter() - start
    mean_plan = average_plan(data, runs)
    out["what-if, average"] = Approach(
        "what-if, average",
        mean_plan,
        float(profits(data, mean_plan.lines, mean_plan.make, train).mean()),
        play(mean_plan),
        N_TRAIN,
        what_if,
    )
    start = time.perf_counter()
    pick = best_candidate(data, runs, train)
    out["what-if, best"] = Approach(
        "what-if, best",
        pick,
        float(profits(data, pick.lines, pick.make, train).mean()),
        play(pick),
        N_TRAIN,
        what_if + time.perf_counter() - start,
    )

    start = time.perf_counter()
    sto = solve_plan(data, train)
    out["stochastic"] = Approach(
        "stochastic", sto.plan, sto.promised, play(sto.plan), 1, 0.0
    )
    out["stochastic"].seconds = time.perf_counter() - start

    # Perfect information: in each scenario, the plan we would have made had
    # we known that future. No one can do better; it is a bound, not a plan.
    ws = np.array([r.promised for r in runs])
    out["perfect information"] = Approach(
        "perfect information", None, float(ws.mean()), None, N_TRAIN, what_if
    )
    out["_runs"] = runs  # Kept for the report and the plots.
    out["_train"] = train
    return out


def k(x: float) -> str:
    """Format euros in thousands."""
    return f"{x / 1000:,.1f}k"


def paired(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Return the mean of a - b and its standard error (same scenarios)."""
    diff = a - b
    return float(diff.mean()), float(diff.std(ddof=1) / np.sqrt(len(diff)))


def print_main_case(data: PlanningData, res: dict) -> list[str]:
    """Print the five approaches at the main spread. Return check errors."""
    names = [p.name for p in data.products]
    print(
        "Season plan. Forecast: "
        + ", ".join(f"{p.name} {p.mean_demand:,.0f}" for p in data.products)
        + f" units.\nDemand spread {SPREAD:.0%}. The solver sees {N_TRAIN} scenarios; "
        f"each plan is then\nplayed out on {N_TEST:,} fresh ones.\n"
    )
    rows = []
    for key in ("point forecast", "what-if, average", "what-if, best", "stochastic"):
        a = res[key]
        rows.append(
            (
                a.name,
                a.plan.lines,
                *[f"{x:,.0f}" for x in a.plan.make],
                k(a.promised),
                k(a.actual.mean()),
                k(np.percentile(a.actual, 5)),
                f"{(a.actual < 0).mean():.1%}",
                a.solves,
                a.seconds,
            )
        )
    ws = res["perfect information"]
    rows.append((ws.name, "", "", "", "", k(ws.promised), "", "", "", ws.solves, ""))
    header = ["approach", "lines", *names, "promised", "actual", "bad year", "loss"]
    print(format_table([*header, "solves", "seconds"], rows))
    print(
        "\npromised: average profit the method expects; actual: average over the"
        "\ntest scenarios; bad year: the 5% quantile; loss: chance of a loss."
    )

    det, sto = res["point forecast"].actual, res["stochastic"].actual
    vss, vss_se = paired(sto, det)
    evpi = ws.promised - sto.mean()
    ws_se = np.std([r.promised for r in res["_runs"]], ddof=1) / np.sqrt(N_TRAIN)
    gap = res["point forecast"].promised - det.mean()
    print(
        f"\nFlaw of averages: the point forecast promises {k(gap)} more than it earns."
        f"\nValue of the stochastic solution (VSS): {k(vss)} +- {k(vss_se)}"
        f" ({vss / sto.mean():.1%})."
        f"\nValue of perfect information (EVPI):    {k(evpi)} +- {k(ws_se)}"
        f" ({evpi / sto.mean():.1%})."
    )
    lines = [r.plan.lines for r in res["_runs"]]
    print(
        f"The {N_TRAIN} what-if runs rent between {min(lines)} and {max(lines)} lines"
        f" (point forecast: {res['point forecast'].plan.lines})."
    )
    return verify(data, res)


def verify(data: PlanningData, res: dict) -> list[str]:
    """Check plans and compare every solver optimum with the referee."""
    errors = []
    for key in ("point forecast", "what-if, average", "what-if, best", "stochastic"):
        plan = res[key].plan
        errors += [f"{key}: {e}" for e in plan_errors(data, plan.lines, plan.make)]
    checks = [("point forecast", point_forecast(data), res["point forecast"].promised)]
    checks.append(("stochastic", res["_train"], res["stochastic"].promised))
    for name, scen, promised in checks:
        _, _, ref = best_plan(data, scen)
        if abs(ref - promised) > 1e-6 * abs(ref):
            errors.append(f"{name}: solver {promised:.2f} != referee {ref:.2f}")
    sto = res["stochastic"].plan
    direct = profits(data, sto.lines, sto.make, res["_train"]).mean()
    if abs(direct - res["stochastic"].promised) > 1e-6 * abs(direct):
        errors.append("stochastic: objective does not match the direct playout")
    for s in range(0, N_TRAIN, 30):  # A sample of what-if runs.
        _, _, ref = best_plan(data, res["_train"].subset([s]))
        if abs(ref - res["_runs"][s].promised) > 1e-6 * abs(ref):
            errors.append(f"what-if run {s}: solver != referee")
    return errors


def sweep(data: PlanningData) -> dict[float, dict]:
    """Run the comparison for each spread."""
    return {s: compare(data, s) for s in SPREADS}


def print_sweep(table: dict[float, dict]) -> None:
    """Print how the gap between approaches grows with the spread."""
    rows = []
    for s, res in table.items():
        det = res["point forecast"].actual.mean()
        sto = res["stochastic"]
        rows.append(
            (
                f"{s:.0%}",
                k(res["point forecast"].promised),
                k(det),
                k(res["what-if, average"].actual.mean()),
                k(res["what-if, best"].actual.mean()),
                k(sto.actual.mean()),
                k(res["perfect information"].promised),
                f"{(sto.actual.mean() - det) / sto.actual.mean():.1%}",
                sto.plan.lines,
                " / ".join(f"{x:,.0f}" for x in sto.plan.make),
            )
        )
    header = ["spread", "promised", "forecast", "avg what-if", "best what-if"]
    header += ["stochastic", "perfect", "VSS", "lines", "stochastic plan"]
    print("\nAs the forecast gets less certain (actual average profit per plan):")
    print(format_table(header, rows))


def convergence(data: PlanningData, sizes=(10, 30, 100, 300, 1000), reps=8):
    """Solve the stochastic model with more and more scenarios.

    Return {N: (promised per rep, actual per rep)}. With few scenarios, the
    model fits their quirks: it promises too much and earns too little.
    """
    test = sample(data, SPREAD, N_TEST, TEST_SEED)
    out = {}
    for n in sizes:
        promised, actual = [], []
        for r in range(reps):
            sol = solve_plan(data, sample(data, SPREAD, n, 100 + r))
            promised.append(sol.promised)
            actual.append(profits(data, sol.plan.lines, sol.plan.make, test).mean())
        out[n] = (np.array(promised), np.array(actual))
    return out


def print_convergence(conv) -> None:
    """Print the in-sample promise and the real result per scenario count."""
    rows = [
        (n, k(p.mean()), k(a.mean()), k(a.min()), k(a.max()), k(p.mean() - a.mean()))
        for n, (p, a) in conv.items()
    ]
    print(f"\nHow many scenarios? ({len(next(iter(conv.values()))[0])} samples each)")
    header = ["scenarios", "promised", "actual", "worst", "best", "over-promise"]
    print(format_table(header, rows))


# ---------------------------------------------------------------- plots ---


STYLE = {
    "point forecast": (0, "o"),
    "what-if, average": (4, "v"),
    "what-if, best": (1, "^"),
    "stochastic": (2, "s"),
    "perfect information": (7, "D"),
}


def plot_sweep(table) -> Path:
    """Plot the expected profit of each approach against the demand spread."""
    from examples.common.plotting import COLORS, new_figure, save

    xs = [100 * s for s in table]
    fig, ax = new_figure(7.5, 4.4)
    promise = [table[s]["point forecast"].promised / 1000 for s in table]
    ax.plot(xs, promise, "--", color=COLORS[0], lw=1, label="point forecast (promise)")
    for name, (c, m) in STYLE.items():
        ys = [
            (r[name].promised if r[name].actual is None else r[name].actual.mean())
            / 1000
            for r in table.values()
        ]
        ls = ":" if name == "perfect information" else "-"
        ax.plot(xs, ys, ls, marker=m, ms=5, color=COLORS[c], label=name)
    sto = [r["stochastic"].actual.mean() / 1000 for r in table.values()]
    det = [r["point forecast"].actual.mean() / 1000 for r in table.values()]
    ax.fill_between(xs, det, sto, color=COLORS[2], alpha=0.15, label="VSS")
    ax.set_xlabel("demand spread (%)")
    ax.set_ylabel("expected season profit (k€)")
    ax.set_title("The less certain the forecast, the more a stochastic model pays")
    ax.legend(loc="lower left", fontsize=8)
    return save(fig, FIGURES / "value_of_uncertainty.png")


def plot_plans(data: PlanningData, table) -> Path:
    """Left: stochastic plan vs spread. Right: the what-if cloud at SPREAD."""
    from examples.common.plotting import COLORS, new_figure, save

    fig, (left, right) = new_figure(10, 4.2, ncols=2)
    mean = data.column("mean_demand")
    xs = [100 * s for s in table]
    for i, p in enumerate(data.products):
        ys = [100 * r["stochastic"].plan.make[i] / mean[i] for r in table.values()]
        left.plot(xs, ys, "o-", color=COLORS[i], label=p.name)
    left.axhline(100, color="0.5", lw=1, ls="--")
    left.set_xlabel("demand spread (%)")
    left.set_ylabel("stochastic plan: made in-house (% of forecast)")
    left.set_title("Hedging: keep the parkas, cut gloves and fleece")
    left.legend()

    res = table[SPREAD]
    runs = np.array([r.plan.make for r in res["_runs"]])
    lines = np.array([r.plan.lines for r in res["_runs"]])
    sc = right.scatter(runs[:, 0], runs[:, 2], c=lines, cmap="viridis", s=14, alpha=0.7)
    fig.colorbar(sc, ax=right, label="lines rented")
    for name in ("point forecast", "what-if, average", "what-if, best", "stochastic"):
        c, m = STYLE[name]
        mk = res[name].plan.make
        right.scatter(
            mk[0],
            mk[2],
            marker=m,
            s=90,
            color=COLORS[c],
            edgecolor="black",
            zorder=3,
            label=name,
        )
    right.set_xlabel("parkas made")
    right.set_ylabel("gloves made")
    right.set_title(f"{N_TRAIN} what-if runs at {SPREAD:.0%} spread: no consensus")
    right.legend(loc="upper left", fontsize=8)
    return save(fig, FIGURES / "plans.png")


def plot_distribution(table) -> Path:
    """Profit histograms of the point-forecast and stochastic plans."""
    from examples.common.plotting import COLORS, new_figure, save

    res = table[SPREAD]
    fig, ax = new_figure(7.5, 3.8)
    det, sto = res["point forecast"].actual / 1000, res["stochastic"].actual / 1000
    bins = np.linspace(min(det.min(), sto.min()), max(det.max(), sto.max()), 60)
    for name, vals, c in (("point forecast", det, 0), ("stochastic", sto, 2)):
        ax.hist(vals, bins=bins, alpha=0.55, color=COLORS[c], label=name)
        ax.axvline(vals.mean(), color=COLORS[c], lw=1.5)
        ax.axvline(np.percentile(vals, 5), color=COLORS[c], lw=1.5, ls=":")
    ax.axvline(
        res["point forecast"].promised / 1000,
        color="black",
        ls="--",
        lw=1,
        label="point forecast promise",
    )
    ax.set_xlabel("season profit (k€); solid = mean, dotted = 5% quantile")
    ax.set_ylabel(f"scenarios (of {N_TEST:,})")
    ax.set_title(f"Profit over {N_TEST:,} futures at {SPREAD:.0%} spread")
    ax.legend(loc="upper left", fontsize=8)
    return save(fig, FIGURES / "profit_distribution.png")


def plot_convergence(conv) -> Path:
    """Promised vs actual profit as the scenario count grows."""
    from examples.common.plotting import COLORS, new_figure, save

    fig, ax = new_figure(7, 3.8)
    ns = list(conv)
    for idx, (label, c) in enumerate((("promised (in-sample)", 3), ("actual", 2))):
        vals = np.array([conv[n][idx] for n in ns]) / 1000
        ax.plot(ns, vals.mean(axis=1), "o-", color=COLORS[c], label=label)
        ax.fill_between(
            ns, vals.min(axis=1), vals.max(axis=1), color=COLORS[c], alpha=0.15
        )
    ax.set_xscale("log")
    ax.set_xlabel("scenarios in the stochastic model")
    ax.set_ylabel("expected season profit (k€)")
    ax.set_title("Few scenarios: the model over-promises and under-delivers")
    ax.legend()
    return save(fig, FIGURES / "scenarios.png")


def main() -> None:
    """Parse arguments, solve, verify, and report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plot", action="store_true", help="write PNG figures")
    parser.add_argument(
        "--convergence", action="store_true", help="also vary the scenario count"
    )
    args = parser.parse_args()

    data = PlanningData()
    table = sweep(data)
    errors = print_main_case(data, table[SPREAD])
    print_sweep(table)
    conv = None
    if args.convergence or args.plot:
        conv = convergence(data)
        print_convergence(conv)

    if errors:
        print("\nVerification FAILED:", *errors, sep="\n  ")
    else:
        print(
            "\nVerification: every plan fits the sewing lines; the solver optima for"
            "\nthe point forecast, the stochastic model, and sampled what-if runs"
            "\nmatch an exact greedy referee in check.py (no OR-Tools)."
        )
    if args.plot:
        for path in (
            plot_sweep(table),
            plot_plans(data, table),
            plot_distribution(table),
            plot_convergence(conv),
        ):
            print("Wrote", path)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
