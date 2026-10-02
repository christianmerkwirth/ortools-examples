"""Production planning as a linear program (LP), solved with MathOpt and GLOP.

Decision: how many units of each product to make per week.
Goal: maximize total profit.
Limits: resource capacities, market demand, and promised orders.

Besides the plan itself, an LP solver gives two kinds of economic insight
for free:

* Shadow prices (dual values): how much more profit one extra unit of a
  resource would bring.
* Reduced costs: how much a product's profit must change before the
  optimal plan changes how much of it to make.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from ortools.math_opt.python import mathopt

from .data import ProductionData


class NoOptimalPlanError(RuntimeError):
    """Raised when the solver does not prove an optimal plan."""


@dataclass
class ProductionModel:
    """A MathOpt model plus handles to its variables and constraints.

    Keep the handles so that we can read results and change the model
    (for example a capacity) without building it again.
    """

    model: mathopt.Model
    quantity: dict[str, mathopt.Variable]  # Product name -> variable.
    capacity: dict[str, mathopt.LinearConstraint]  # Resource name -> row.


@dataclass(frozen=True)
class ProductionPlan:
    """The optimal plan and its sensitivity information."""

    profit: float
    quantity: dict[str, float]  # Units per product.
    resource_used: dict[str, float]  # Amount used per resource.
    shadow_price: dict[str, float]  # $ per extra unit of each resource.
    reduced_cost: dict[str, float]  # $ per unit, see module docstring.


def build_model(data: ProductionData) -> ProductionModel:
    """Build the LP for one instance."""
    model = mathopt.Model(name=data.name)

    # One continuous variable per product. Demand and promised orders are
    # simple bounds, so we put them on the variable instead of adding extra
    # constraints. Their effect then shows up in the reduced costs.
    quantity = {
        p.name: model.add_variable(lb=p.min_order, ub=p.max_demand, name=p.name)
        for p in data.products
    }

    # One capacity constraint per resource: total usage <= capacity.
    capacity = {
        r.name: model.add_linear_constraint(
            mathopt.fast_sum(
                p.usage.get(r.name, 0.0) * quantity[p.name] for p in data.products
            )
            <= r.capacity,
            name=r.name,
        )
        for r in data.resources
    }

    model.maximize(mathopt.fast_sum(p.profit * quantity[p.name] for p in data.products))
    return ProductionModel(model, quantity, capacity)


def solve_model(
    pm: ProductionModel,
    data: ProductionData,
    solver_type: mathopt.SolverType = mathopt.SolverType.GLOP,
) -> ProductionPlan:
    """Solve a built model and collect the plan and its dual information."""
    result = mathopt.solve(pm.model, solver_type)
    if result.termination.reason != mathopt.TerminationReason.OPTIMAL:
        raise NoOptimalPlanError(f"No optimal plan: {result.termination}")

    quantity = {name: result.variable_values(v) for name, v in pm.quantity.items()}
    resource_used = {
        r.name: sum(p.usage.get(r.name, 0.0) * quantity[p.name] for p in data.products)
        for r in data.resources
    }
    return ProductionPlan(
        profit=result.objective_value(),
        quantity=quantity,
        resource_used=resource_used,
        # dual_values() and reduced_costs() need a dual solution. LP solvers
        # such as GLOP, PDLP, and HiGHS give one; MIP solvers do not.
        shadow_price={n: _clean(result.dual_values(c)) for n, c in pm.capacity.items()},
        reduced_cost={
            n: _clean(result.reduced_costs(v)) for n, v in pm.quantity.items()
        },
    )


def _clean(value: float, tol: float = 1e-9) -> float:
    """Map tiny round-off values (and -0.0) to 0.0 for readable output."""
    return 0.0 if abs(value) < tol else value


def solve_production_plan(
    data: ProductionData,
    solver_type: mathopt.SolverType = mathopt.SolverType.GLOP,
) -> ProductionPlan:
    """Build and solve the model in one call."""
    return solve_model(build_model(data), data, solver_type)


def profit_curve(
    data: ProductionData,
    resource: str,
    capacities: Iterable[float],
) -> list[tuple[float, float | None, float | None]]:
    """Solve the LP for a range of capacities of one resource.

    This is parametric analysis. We build the model once and only change
    the right-hand side of one constraint between solves.

    Returns:
        A list of (capacity, profit, shadow price) tuples. Profit and shadow
        price are None where no feasible plan exists.

    """
    pm = build_model(data)
    row = pm.capacity[resource]
    points = []
    for cap in capacities:
        row.upper_bound = cap
        try:
            plan = solve_model(pm, data)
        except NoOptimalPlanError:
            points.append((cap, None, None))
        else:
            points.append((cap, plan.profit, plan.shadow_price[resource]))
    return points
