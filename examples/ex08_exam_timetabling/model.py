"""Exam timetabling as graph coloring, solved with CP-SAT.

Part 1 finds the fewest slots for an exam week. That number is the
chromatic number of the conflict graph. We show two ways to write the
model and measure how much symmetry breaking helps:

* `color_integer`: one integer variable per exam (its slot), and
  `slot[a] != slot[b]` for each conflict.
* `color_boolean`: one true/false variable per (exam, slot) pair.

Part 2 (`spread_exams`) fixes the week (5 days, 2 slots a day) and makes
it kinder for students: it avoids two exams on one day and, less
strongly, exams on two days in a row.
"""

import functools
import itertools
import time
from dataclasses import dataclass

from ortools.sat.python import cp_model

from .bounds import dsatur, max_clique, maximal_cliques
from .data import ExamData, Graph

# Penalty per student for each pair of their exams.
SAME_DAY_PENALTY = 3
NEXT_DAY_PENALTY = 1


@dataclass(frozen=True)
class Coloring:
    """A coloring found by CP-SAT, with its proof status."""

    color: dict[str, int]  # Node -> color (slot) number, from 0.
    num_colors: int
    lower_bound: int  # No coloring with fewer colors exists.
    optimal: bool
    seconds: float


@dataclass(frozen=True)
class Timetable:
    """An exam week found by CP-SAT."""

    slot: dict[str, int]  # Exam -> slot number, from 0.
    penalty: int
    lower_bound: int  # No timetable has a smaller penalty.
    optimal: bool
    seconds: float


# ---------------------------------------------------------------- Part 1 ---


def color_integer(
    graph: Graph, symmetry_breaking: bool = True, time_limit: float = 10.0
) -> Coloring:
    """Minimize the number of colors with one integer variable per node."""
    clique = max_clique(graph)
    # DSATUR gives a coloring with `upper` colors, so we never need more.
    # A small domain makes the model much smaller.
    upper = max(dsatur(graph).values()) + 1
    model = cp_model.CpModel()
    color = {n: model.new_int_var(0, upper - 1, n) for n in graph.nodes}
    for a, b in graph.edges:
        model.add(color[a] != color[b])

    # The number of colors is the highest color used, plus one. A clique of
    # size q needs q colors; telling the solver this helps it prove the
    # optimum.
    num_colors = model.new_int_var(len(clique), upper, "num_colors")
    model.add_max_equality(num_colors - 1, list(color.values()))
    model.minimize(num_colors)

    if symmetry_breaking:
        # Any coloring stays valid if we swap two colors. So we may fix the
        # colors of the clique nodes: they must all differ anyway.
        for k, node in enumerate(clique):
            model.add(color[node] == k)

    solver, status, seconds = _solve(model, time_limit)
    return Coloring(
        color={n: solver.value(v) for n, v in color.items()},
        num_colors=round(solver.objective_value),
        lower_bound=round(solver.best_objective_bound),
        optimal=status == cp_model.OPTIMAL,
        seconds=seconds,
    )


def color_boolean(
    graph: Graph, symmetry_breaking: bool = True, time_limit: float = 10.0
) -> Coloring:
    """Minimize the number of colors with one Boolean per (node, color)."""
    clique = max_clique(graph)
    upper = max(dsatur(graph).values()) + 1
    colors = range(upper)
    model = cp_model.CpModel()
    x = {(n, c): model.new_bool_var(f"{n}={c}") for n in graph.nodes for c in colors}
    used = [model.new_bool_var(f"used {c}") for c in colors]

    for n in graph.nodes:
        model.add_exactly_one(x[n, c] for c in colors)
        for c in colors:
            model.add_implication(x[n, c], used[c])
    for a, b in graph.edges:
        for c in colors:
            # Stronger than "x[a,c] + x[b,c] <= 1": it also says that an
            # edge in color c only exists if color c is used.
            model.add(x[a, c] + x[b, c] <= used[c])

    model.add(sum(used) >= len(clique))
    model.minimize(sum(used))

    if symmetry_breaking:
        for k, node in enumerate(clique):
            model.add(x[node, k] == 1)
        # Use the colors in order: color c + 1 only if color c is used.
        for c in colors[:-1]:
            model.add_implication(used[c + 1], used[c])

    solver, status, seconds = _solve(model, time_limit)
    return Coloring(
        color={
            n: next(c for c in colors if solver.boolean_value(x[n, c]))
            for n in graph.nodes
        },
        num_colors=round(solver.objective_value),
        lower_bound=round(solver.best_objective_bound),
        optimal=status == cp_model.OPTIMAL,
        seconds=seconds,
    )


# ---------------------------------------------------------------- Part 2 ---


def spread_exams(
    data: ExamData,
    time_limit: float = 20.0,
    student_cuts: bool = True,
    clique_cuts: bool = True,
) -> Timetable:
    """Place exams into a fixed week and spread each student's exams.

    For every pair of exams that share students we add two Boolean flags:
    `same_day` and `next_day`. The objective counts, per shared student,
    3 points for a same-day pair and 1 point for a next-day pair.

    `student_cuts` and `clique_cuts` switch on redundant constraints that
    raise the lower bound. They do not change the best timetable.
    """
    shared = data.shared_students()
    days, per_day = range(data.days), data.slots_per_day
    slots = range(data.num_slots)
    model = cp_model.CpModel()
    x = {(e, s): model.new_bool_var(f"{e}@{s}") for e in data.exams for s in slots}
    for e in data.exams:
        model.add_exactly_one(x[e, s] for s in slots)

    def on_day(e: str, d: int) -> cp_model.LinearExpr:
        # 1 if exam e is on day d. A sum, not a new variable: exactly one
        # slot is chosen, so the sum is 0 or 1.
        return sum(x[e, s] for s in range(d * per_day, (d + 1) * per_day))

    penalty_terms = []
    pair_penalty = {}
    for (a, b), students in shared.items():
        for s in slots:
            model.add_at_most_one(x[a, s], x[b, s])  # The hard rule.
        same_day = model.new_bool_var(f"{a}|{b} same day")
        next_day = model.new_bool_var(f"{a}|{b} next day")
        for d in days:
            # If both are on day d, then same_day must be true.
            model.add(on_day(a, d) + on_day(b, d) - 1 <= same_day)
            if d + 1 < data.days:
                model.add(on_day(a, d) + on_day(b, d + 1) - 1 <= next_day)
                model.add(on_day(a, d + 1) + on_day(b, d) - 1 <= next_day)
        pair_penalty[a, b] = SAME_DAY_PENALTY * same_day + NEXT_DAY_PENALTY * next_day
        penalty_terms.append(students * pair_penalty[a, b])

    # Redundant constraints: they cut off no timetable, but they give the
    # solver a much better lower bound. Exams that all conflict with each
    # other (a clique) need different slots. Spread over the week, k such
    # exams always cause at least min_student_penalty(k) penalty points
    # (counted per pair, before weighting by students). Each student's
    # exams form a clique, and so do most courses of one department.
    cliques = []
    if student_cuts:
        cliques += [list(exams) for exams in set(data.enrollments)]
    if clique_cuts:
        cliques += maximal_cliques(data.conflict_graph(), min_size=4)
    order = {e: k for k, e in enumerate(data.exams)}
    for clique in cliques:
        floor = min_student_penalty(len(clique), data.days, per_day)
        if floor > 0:
            pairs = itertools.combinations(sorted(clique, key=order.get), 2)
            model.add(sum(pair_penalty[p] for p in pairs) >= floor)
    model.minimize(sum(penalty_terms))

    solver, status, seconds = _solve(model, time_limit)
    slot = {
        e: next(s for s in slots if solver.boolean_value(x[e, s])) for e in data.exams
    }
    # The flags only have lower limits ("if both on day d, then same_day").
    # Minimizing pushes them down, but a solution found before the time
    # limit can still have a flag switched on without need. So we count the
    # true penalty from the slots. It is never above the objective value.
    day = {e: s // per_day for e, s in slot.items()}
    penalty = sum(n * _day_penalty(day[a], day[b]) for (a, b), n in shared.items())
    return Timetable(
        slot=slot,
        penalty=penalty,
        lower_bound=round(solver.best_objective_bound),
        optimal=status == cp_model.OPTIMAL,
        seconds=seconds,
    )


@functools.cache
def min_student_penalty(num_exams: int, days: int, per_day: int) -> int:
    """Return the lowest penalty one student with `num_exams` exams can get.

    Try every way to spread the exams over the days (at most `per_day` on
    one day). This is quick for the small numbers here (5^5 = 3125 cases).
    """
    best = None
    for plan in itertools.product(range(days), repeat=num_exams):
        if max(plan.count(d) for d in plan) > per_day:
            continue
        penalty = sum(_day_penalty(a, b) for a, b in itertools.combinations(plan, 2))
        best = penalty if best is None else min(best, penalty)
    if best is None:
        raise ValueError(f"{num_exams} exams do not fit into {days} days.")
    return best


# --------------------------------------------------------------- helpers ---


def _day_penalty(day_a: int, day_b: int) -> int:
    """Return the penalty for two exams of one student on these days."""
    if day_a == day_b:
        return SAME_DAY_PENALTY
    return NEXT_DAY_PENALTY if abs(day_a - day_b) == 1 else 0


def _solve(model: cp_model.CpModel, time_limit: float):
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 8
    start = time.perf_counter()
    status = solver.solve(model)
    seconds = time.perf_counter() - start
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"CP-SAT found no solution ({solver.status_name(status)}).")
    return solver, status, seconds
