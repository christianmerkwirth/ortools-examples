"""Pick a low-risk portfolio with CVaR, then limit the number of assets.

Run from the repository root:

    uv run python -m examples.ex17_portfolio.main
    uv run python -m examples.ex17_portfolio.main --plot

A teaching example with synthetic data, not investment advice.
"""

import argparse
import time
from pathlib import Path

import numpy as np

from examples.common.report import format_table

from .check import (
    cvar,
    feasibility_errors,
    lp_lower_bound,
    value_at_risk,
    zeta_range,
)
from .data import PortfolioData, Rules, generate
from .model import Portfolio, frontier, return_range, solve_portfolio

FIGURES = Path(__file__).parent / "figures"
TARGET = 0.010  # 1.0% mean return per month.

RULE_SETS = {
    "no limit (LP)": Rules(),
    "max 6 assets": Rules(min_weight=0.03, max_assets=6),
    "max 4 assets": Rules(min_weight=0.03, max_assets=4),
}


def pct(x: float) -> str:
    """Format a fraction as a percentage with two decimals."""
    return f"{100 * x:.2f}%"


def solve_all(data: PortfolioData) -> dict[str, tuple[Portfolio, float]]:
    """Solve each rule set at the target return. Return portfolio and time."""
    out = {}
    for name, rules in RULE_SETS.items():
        start = time.perf_counter()
        portfolio = solve_portfolio(data, rules, TARGET)
        out[name] = (portfolio, time.perf_counter() - start)
    return out


def print_report(data: PortfolioData, results) -> list[str]:
    """Print the portfolios and verify them. Return check errors."""
    print(
        f"{data.n_assets} assets, {data.n_scenarios} scenarios of monthly returns."
        f"\nGoal: the lowest CVaR (average loss in the worst 5% of months) for a"
        f"\nmean return of at least {pct(TARGET)} per month.\n"
    )
    rows = []
    for name, (p, seconds) in results.items():
        rows.append(
            (
                name,
                int((p.weights > 0).sum()),
                pct(p.expected_return),
                pct(value_at_risk(data, p.weights, RULE_SETS[name].alpha)),
                pct(p.cvar),
                "yes" if p.proven_optimal else f"gap {p.gap:.2%}",
                seconds,
            )
        )
    header = ["rules", "assets", "mean", "VaR 95%", "CVaR 95%", "optimal", "seconds"]
    print(format_table(header, rows))

    print("\nWeights (assets held by at least one portfolio)")
    names = list(results)
    weights = np.array([results[n][0].weights for n in names])
    rows = [
        (a.name, *[pct(w) if w > 0 else "" for w in weights[:, i]])
        for i, a in enumerate(data.assets)
        if weights[:, i].max() > 0
    ]
    print(format_table(["asset", *names], rows))

    lp = results["no limit (LP)"][0]
    step = 0.001
    print(
        f"\nShadow price of the target: {lp.return_price:.2f}. Raising the target"
        f"\nby {pct(step)} per month costs about {pct(lp.return_price * step)} of CVaR."
    )

    errors = []
    for name, (p, _) in results.items():
        rules = RULE_SETS[name]
        errors += [
            f"{name}: {e}" for e in feasibility_errors(data, rules, p.weights, TARGET)
        ]
        direct = cvar(data, p.weights, rules.alpha)
        if abs(direct - p.cvar) > 1e-7:
            errors.append(f"{name}: solver CVaR {p.cvar} != sorted tail {direct}")
        lo, hi = zeta_range(data, p.weights, rules.alpha)
        if not lo - 1e-7 <= p.zeta <= hi + 1e-7:
            errors.append(f"{name}: zeta {p.zeta} outside the optimal range")
        if not p.proven_optimal:
            errors.append(f"{name}: not proven optimal")
    bound = lp_lower_bound(
        data,
        RULE_SETS["no limit (LP)"],
        TARGET,
        lp.scenario_prices,
        lp.budget_price,
        lp.return_price,
        lp.sector_prices,
    )
    print(f"LP dual bound (check.py): {pct(bound)}; LP CVaR: {pct(lp.cvar)}")
    if abs(bound - lp.cvar) > 1e-7:
        errors.append(f"LP: dual bound {bound} does not match CVaR {lp.cvar}")
    return errors


def plot_frontier(data: PortfolioData) -> Path:
    """Efficient frontiers: lowest CVaR for each target return, per rule set."""
    from examples.common.plotting import COLORS, new_figure, save

    low, high = return_range(data, RULE_SETS["no limit (LP)"])
    targets = list(np.linspace(low, high, 21))
    fig, ax = new_figure(7, 4.2)
    for color, (name, rules) in zip(COLORS, RULE_SETS.items(), strict=False):
        points = [
            (t, p.cvar)
            for t, p in zip(targets, frontier(data, rules, targets), strict=True)
            if p
        ]
        ax.plot(
            [100 * c for _, c in points],
            [100 * t for t, _ in points],
            "o-",
            ms=3,
            color=color,
            label=name,
        )
    ax.axhline(100 * TARGET, color="0.5", ls="--", lw=1)
    ax.text(ax.get_xlim()[0], 100 * TARGET, " target", va="bottom", color="0.4")
    ax.set_xlabel("CVaR 95%: average loss in the worst 5% of months (%)")
    ax.set_ylabel("mean monthly return (%)")
    ax.set_title("Efficient frontiers: a limit on the number of assets costs risk")
    ax.legend(loc="lower right")
    return save(fig, FIGURES / "frontier.png")


def plot_weights(data: PortfolioData, results) -> Path:
    """Horizontal bars: weight of each asset in each portfolio."""
    from examples.common.plotting import COLORS, new_figure, save

    names = list(results)
    weights = np.array([results[n][0].weights for n in names])
    used = [i for i in range(data.n_assets) if weights[:, i].max() > 0]
    fig, ax = new_figure(7, 0.35 * len(used) + 1.2)
    height = 0.8 / len(names)
    for k, (name, color) in enumerate(zip(names, COLORS, strict=False)):
        ys = [j + (k - 1) * height for j in range(len(used))]
        ax.barh(
            ys,
            [100 * weights[k, i] for i in used],
            height=height,
            color=color,
            label=name,
        )
    ax.set_yticks(range(len(used)), [data.assets[i].name for i in used])
    ax.invert_yaxis()
    ax.set_xlim(0, 36)
    ax.set_xlabel("share of the money (%)")
    ax.set_title(f"Portfolios for a {pct(TARGET)} monthly target")
    ax.legend(loc="lower right")
    return save(fig, FIGURES / "weights.png")


def plot_tail(data: PortfolioData, portfolio: Portfolio, alpha: float) -> Path:
    """Histogram of monthly returns with VaR and CVaR marked."""
    from examples.common.plotting import COLORS, new_figure, save

    returns = 100 * (data.returns @ portfolio.weights)
    var = -100 * value_at_risk(data, portfolio.weights, alpha)
    cv = -100 * portfolio.cvar
    fig, ax = new_figure(7, 3.6)
    bins = np.linspace(returns.min(), returns.max(), 50)
    ax.hist(
        returns[returns > var],
        bins=bins,
        color=COLORS[0],
        alpha=0.8,
        label="other months",
    )
    ax.hist(
        returns[returns <= var],
        bins=bins,
        color=COLORS[3],
        alpha=0.9,
        label=f"worst {1 - alpha:.0%} of months",
    )
    ax.axvline(var, color="black", ls="--", lw=1)
    ax.text(var, ax.get_ylim()[1] * 0.92, f" VaR: {var:.2f}%", ha="left")
    ax.axvline(cv, color=COLORS[3], lw=1.5, ls="-.")
    ax.text(
        cv, ax.get_ylim()[1] * 0.80, f"CVaR: {cv:.2f}% ", ha="right", color=COLORS[3]
    )
    ax.set_xlabel("monthly return of the LP portfolio (%)")
    ax.set_ylabel("months (of 400)")
    ax.set_title("CVaR is the average of the red tail; VaR is where the tail starts")
    ax.legend(loc="upper right")
    return save(fig, FIGURES / "tail.png")


def main() -> None:
    """Parse arguments, solve, verify, and report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plot", action="store_true", help="write PNG figures")
    args = parser.parse_args()

    data = generate()
    results = solve_all(data)
    errors = print_report(data, results)

    if errors:
        print("\nVerification FAILED:", *errors, sep="\n  ")
    else:
        print(
            "\nVerification: every portfolio meets all rules; each solver CVaR"
            "\nmatches a direct sort of the scenario losses; the LP portfolio is"
            "\nproven optimal by its dual bound, the MIPs by HiGHS."
        )
    if args.plot:
        lp = results["no limit (LP)"][0]
        for path in (
            plot_frontier(data),
            plot_weights(data, results),
            plot_tail(data, lp, RULE_SETS["no limit (LP)"].alpha),
        ):
            print("Wrote", path)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
