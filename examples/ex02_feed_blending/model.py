"""Feed blending as a linear program, plus tools to diagnose infeasibility.

Decision: the share of each ingredient in the blend.
Goal: minimize the cost of one batch.
Limits: every nutrient must stay inside its range, and some ingredients
have a maximum share.

A blending spec written by people often asks for something impossible.
Then the solver only says "infeasible". This module shows two ways to
find out why:

* An elastic model lets every nutrient requirement bend, at a penalty.
  It finds the smallest change to the spec that makes it feasible.
* A deletion filter finds a minimal set of limits that conflict with each
  other (an irreducible infeasible subsystem, IIS).
"""

import math
from dataclasses import dataclass

from ortools.math_opt.python import mathopt

from .data import BlendData

OPTIMAL = mathopt.TerminationReason.OPTIMAL
INFEASIBLE = mathopt.TerminationReason.INFEASIBLE


class InfeasibleSpecError(RuntimeError):
    """Raised when no blend can meet the specification."""


@dataclass
class BlendModel:
    """A MathOpt model plus handles to its variables and constraints."""

    model: mathopt.Model
    share: dict[str, mathopt.Variable]  # Ingredient -> fraction of blend.
    total: mathopt.LinearConstraint  # The shares add up to 1.
    nutrient: dict[str, mathopt.LinearConstraint]  # Nutrient -> range row.


@dataclass(frozen=True)
class Blend:
    """An optimal blend and its dual information."""

    cost: float  # Dollars per batch.
    share: dict[str, float]  # Ingredient -> fraction of the blend.
    level: dict[str, float]  # Nutrient -> level in the blend.
    # Dollars per batch per unit of nutrient level. Positive: a binding
    # minimum (raising it costs money). Negative: a binding maximum.
    nutrient_price: dict[str, float]
    total_price: float  # Dual value of the "shares add up to 1" row.


@dataclass(frozen=True)
class Limit:
    """One relaxable limit: a nutrient bound or an ingredient's max share."""

    kind: str  # "nutrient" or "ingredient".
    name: str
    side: str  # "min" or "max".
    value: float

    def __str__(self) -> str:
        op = ">=" if self.side == "min" else "<="
        if self.kind == "ingredient":
            return f"{self.name} share {op} {self.value:.0%}"
        return f"{self.name} {op} {self.value:g}"


@dataclass(frozen=True)
class Relaxation:
    """How much the elastic model moves one nutrient bound."""

    limit: Limit
    new_value: float

    @property
    def change(self) -> float:
        """Relative change of the bound, for example -0.05 for -5%."""
        return (self.new_value - self.limit.value) / abs(self.limit.value)


@dataclass(frozen=True)
class ElasticResult:
    """The smallest spec change that makes the blend feasible."""

    total_violation: float  # Sum of relative changes; 0 means feasible.
    relaxations: list[Relaxation]
    blend_share: dict[str, float]  # A blend that meets the relaxed spec.
    nutrient_price: dict[str, float]  # Duals, used as an infeasibility proof.
    total_price: float


def build_model(data: BlendData) -> BlendModel:
    """Build the cost-minimizing blend LP."""
    model = mathopt.Model(name=data.name)
    share = {
        i.name: model.add_variable(lb=0.0, ub=i.max_share, name=i.name)
        for i in data.ingredients
    }
    total = model.add_linear_constraint(
        mathopt.fast_sum(share.values()) == 1.0, name="total"
    )
    # A ranged constraint min <= level <= max is one row with two bounds.
    # (Python cannot chain `lb <= expr <= ub` here, so pass lb and ub.)
    nutrient = {
        r.nutrient: model.add_linear_constraint(
            lb=r.min,
            ub=r.max,
            expr=_level_expr(data, share, r.nutrient),
            name=r.nutrient,
        )
        for r in data.requirements
    }
    model.minimize(
        data.batch_kg
        * mathopt.fast_sum(i.cost * share[i.name] for i in data.ingredients)
    )
    return BlendModel(model, share, total, nutrient)


def _level_expr(data, share, nutrient: str) -> mathopt.LinearExpression:
    """Nutrient level of the blend: the share-weighted ingredient levels."""
    return mathopt.fast_sum(
        i.nutrients.get(nutrient, 0.0) * share[i.name] for i in data.ingredients
    )


def solve_blend(data: BlendData) -> Blend:
    """Find the cheapest blend. Raise InfeasibleSpecError if none exists."""
    bm = build_model(data)
    result = mathopt.solve(bm.model, mathopt.SolverType.GLOP)
    if result.termination.reason == INFEASIBLE:
        raise InfeasibleSpecError(f"No blend meets the '{data.name}' spec.")
    if result.termination.reason != OPTIMAL:
        raise RuntimeError(f"Unexpected solver result: {result.termination}")

    share = {n: result.variable_values(v) for n, v in bm.share.items()}
    level = {
        r.nutrient: sum(
            i.nutrients.get(r.nutrient, 0.0) * share[i.name] for i in data.ingredients
        )
        for r in data.requirements
    }
    return Blend(
        cost=result.objective_value(),
        share=share,
        level=level,
        nutrient_price={n: result.dual_values(c) for n, c in bm.nutrient.items()},
        total_price=result.dual_values(bm.total),
    )


def relaxable_limits(data: BlendData) -> list[Limit]:
    """List every finite, non-trivial limit in the spec."""
    limits = []
    for r in data.requirements:
        if r.min > 0:
            limits.append(Limit("nutrient", r.nutrient, "min", r.min))
        if math.isfinite(r.max):
            limits.append(Limit("nutrient", r.nutrient, "max", r.max))
    for i in data.ingredients:
        if i.max_share < 1:
            limits.append(Limit("ingredient", i.name, "max", i.max_share))
    return limits


def elastic_blend(data: BlendData) -> ElasticResult:
    """Find the smallest relative change of the nutrient bounds that works.

    Each nutrient row gets two slack variables: `below` lets the level fall
    under the minimum, `above` lets it rise over the maximum. We minimize
    the sum of slacks, each divided by its bound, so that a 1% change counts
    the same for protein (%) and energy (kcal/kg). Ingredient limits stay
    hard: they protect animal health, so we do not bargain over them.
    """
    bm = build_model(data)
    model = bm.model
    penalty = []
    slack: dict[Limit, mathopt.Variable] = {}
    for r in data.requirements:
        row = bm.nutrient[r.nutrient]
        for side, value, sign in (("min", r.min, +1.0), ("max", r.max, -1.0)):
            if value == 0 or not math.isfinite(value):
                continue  # Nothing to relax.
            limit = Limit("nutrient", r.nutrient, side, value)
            s = model.add_variable(lb=0.0, name=f"{r.nutrient}_{side}_slack")
            # row: level + below - above in [min, max].
            row.set_coefficient(s, sign)
            penalty.append(s / abs(value))
            slack[limit] = s
    model.minimize(mathopt.fast_sum(penalty))  # Replaces the cost objective.

    result = mathopt.solve(model, mathopt.SolverType.GLOP)
    if result.termination.reason != OPTIMAL:
        raise RuntimeError(f"Elastic model failed: {result.termination}")

    relaxations = []
    for limit, s in slack.items():
        amount = result.variable_values(s)
        if amount > 1e-9:
            new = limit.value - amount if limit.side == "min" else limit.value + amount
            relaxations.append(Relaxation(limit, new))
    return ElasticResult(
        total_violation=result.objective_value(),
        relaxations=relaxations,
        blend_share={n: result.variable_values(v) for n, v in bm.share.items()},
        nutrient_price={n: result.dual_values(c) for n, c in bm.nutrient.items()},
        total_price=result.dual_values(bm.total),
    )


def find_conflict(data: BlendData) -> list[Limit]:
    """Return a minimal set of limits that cannot hold together.

    This is the deletion filter (Chinneck, 2008). Go through the limits one
    by one. Drop a limit for good if the model stays infeasible without it.
    Keep it if dropping it makes the model feasible. At the end, every kept
    limit is needed for the conflict, so the set is minimal.

    We switch limits off by setting bounds to infinity, so the model is
    built only once. Each check is a fast LP solve.
    """
    bm = build_model(data)
    if _is_feasible(bm):
        return []

    conflict = []
    for limit in relaxable_limits(data):
        restore = _drop(bm, limit)
        if _is_feasible(bm):
            restore()  # This limit is part of the conflict.
            conflict.append(limit)
    return conflict


def _is_feasible(bm: BlendModel) -> bool:
    # Only feasibility matters, so a zero objective is enough and faster.
    saved = bm.model.objective.as_linear_expression()
    bm.model.minimize(0.0)
    result = mathopt.solve(bm.model, mathopt.SolverType.GLOP)
    bm.model.minimize(saved)
    if result.termination.reason not in (OPTIMAL, INFEASIBLE):
        raise RuntimeError(f"Unexpected solver result: {result.termination}")
    return result.termination.reason == OPTIMAL


def _drop(bm: BlendModel, limit: Limit):
    """Switch one limit off. Return a function that switches it back on."""
    if limit.kind == "ingredient":
        var = bm.share[limit.name]
        var.upper_bound = 1.0
        return lambda: setattr(var, "upper_bound", limit.value)
    row = bm.nutrient[limit.name]
    attr = "lower_bound" if limit.side == "min" else "upper_bound"
    setattr(row, attr, -math.inf if limit.side == "min" else math.inf)
    return lambda: setattr(row, attr, limit.value)


def frontier(
    data: BlendData, x_nutrient: str, y_nutrient: str, x_values: list[float]
) -> list[tuple[float, float | None]]:
    """Return the highest reachable y level for each minimum x level.

    This traces the trade-off between two nutrients (for example protein
    and energy). Every spec above the curve is infeasible. Both nutrients'
    own limits are switched off; all other limits stay.
    """
    bm = build_model(data)
    x_row, y_row = bm.nutrient[x_nutrient], bm.nutrient[y_nutrient]
    x_row.upper_bound = math.inf
    y_row.lower_bound, y_row.upper_bound = -math.inf, math.inf
    bm.model.maximize(_level_expr(data, bm.share, y_nutrient))
    points = []
    for x in x_values:
        x_row.lower_bound = x
        result = mathopt.solve(bm.model, mathopt.SolverType.GLOP)
        best = (
            result.objective_value() if result.termination.reason == OPTIMAL else None
        )
        points.append((x, best))
    return points
