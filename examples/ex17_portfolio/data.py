"""Instance data for the portfolio example.

An investor picks a portfolio from 30 assets in six sectors. We do not
know future returns, so we use scenarios: 400 simulated months of returns
for all assets at once. A good portfolio earns a target return on average
and loses little in the bad months.

All data is synthetic and seeded. The asset names are generic labels, not
real companies. This is a teaching example, not investment advice.
"""

from dataclasses import dataclass, field, replace

import numpy as np

SECTORS = ("Utility", "Consumer", "Health", "Finance", "Energy", "Tech")

# Per sector: (mean monthly return range, market beta, own volatility).
# Utilities are calm and earn little; tech swings more and earns more.
_SECTOR_PROFILE = {
    "Utility": ((0.0030, 0.0050), 0.4, 0.020),
    "Consumer": ((0.0045, 0.0065), 0.8, 0.030),
    "Health": ((0.0050, 0.0075), 0.7, 0.035),
    "Finance": ((0.0055, 0.0080), 1.2, 0.040),
    "Energy": ((0.0050, 0.0090), 1.1, 0.060),
    "Tech": ((0.0080, 0.0130), 1.4, 0.070),
}


@dataclass(frozen=True)
class Asset:
    """One asset the investor can buy."""

    name: str
    sector: str


@dataclass(frozen=True)
class Rules:
    """The investor's rules for a portfolio."""

    alpha: float = 0.95  # CVaR level: average loss in the worst 5% of months.
    max_weight: float = 0.25  # No asset above 25% of the money.
    min_weight: float = 0.0  # If an asset is held, at least this share.
    max_assets: int | None = None  # Cardinality limit; None means no limit.
    sector_cap: float = 0.40  # No sector above 40%.


@dataclass(frozen=True)
class PortfolioData:
    """Assets plus a scenario matrix of monthly returns."""

    assets: tuple[Asset, ...]
    # returns[s, i] is the return of asset i in scenario s (0.01 = +1%).
    returns: np.ndarray = field(compare=False, repr=False)

    @property
    def n_assets(self) -> int:
        """Return the number of assets."""
        return len(self.assets)

    @property
    def n_scenarios(self) -> int:
        """Return the number of scenarios."""
        return self.returns.shape[0]

    @property
    def mean_returns(self) -> np.ndarray:
        """Return each asset's average monthly return over the scenarios."""
        return self.returns.mean(axis=0)

    def subset(self, indices: list[int]) -> "PortfolioData":
        """Return a copy with only the given assets."""
        assets = tuple(self.assets[i] for i in indices)
        return replace(self, assets=assets, returns=self.returns[:, indices])


def generate(
    n_per_sector: int = 5, n_scenarios: int = 400, seed: int = 17
) -> PortfolioData:
    """Simulate monthly returns with a simple factor model.

    return = mean + beta * market + sector factor + own noise

    The market factor has fat tails (Student t with 4 degrees of freedom),
    so crashes happen more often than a normal distribution would suggest.
    This is where CVaR earns its keep.
    """
    rng = np.random.default_rng(seed)
    market = 0.035 * rng.standard_t(4, size=n_scenarios) / np.sqrt(2.0)
    assets, columns = [], []
    for sector in SECTORS:
        (lo, hi), beta, vol = _SECTOR_PROFILE[sector]
        sector_factor = 0.015 * rng.standard_normal(n_scenarios)
        for k in range(n_per_sector):
            assets.append(Asset(f"{sector} {chr(ord('A') + k)}", sector))
            mean = rng.uniform(lo, hi)
            b = beta * rng.uniform(0.8, 1.2)
            noise = vol * rng.standard_normal(n_scenarios)
            columns.append(mean + b * market + sector_factor + noise)
    return PortfolioData(tuple(assets), np.column_stack(columns))
