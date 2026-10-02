"""Independent checks for colorings and timetables. Uses no OR-Tools code.

* `coloring_errors` and `timetable_errors` check the hard rules. The
  timetable check reads the student enrollments directly, not the
  conflict graph, so a bug in the graph would show up here.
* `is_clique` checks a clique, which proves a lower bound on the colors.
* `chromatic_number` finds the exact number of colors by backtracking.
  It is slow on big graphs, but simple enough to trust.
* `student_penalties` recomputes the Part 2 objective student by student.
* `penalty_lower_bound` adds up the best case for every student. No
  timetable can do better.
"""

import functools
import itertools

from .data import ExamData, Graph

SAME_DAY_PENALTY = 3
NEXT_DAY_PENALTY = 1


# ---------------------------------------------------------------- Part 1 ---


def coloring_errors(graph: Graph, color: dict[str, int]) -> list[str]:
    """Return a list of broken rules. An empty list means a valid coloring."""
    errors = [f"{n} has no color" for n in graph.nodes if n not in color]
    errors += [
        f"{a} and {b} share color {color[a]}"
        for a, b in graph.edges
        if a in color and b in color and color[a] == color[b]
    ]
    return errors


def num_colors(color: dict[str, int]) -> int:
    """Return the number of distinct colors used."""
    return len(set(color.values()))


def is_clique(graph: Graph, nodes: list[str]) -> bool:
    """Return True if every pair of these nodes is joined by an edge."""
    adj = graph.neighbors()
    return all(b in adj[a] for a, b in itertools.combinations(nodes, 2))


def chromatic_number(graph: Graph) -> int:
    """Return the exact chromatic number by backtracking.

    Try k = 1, 2, ... colors. For each k, color the nodes one by one (the
    nodes with most neighbors first) and undo a choice when a node has no
    free color left. A new color is only tried once, which removes the
    symmetry between unused colors.
    """
    adj = graph.neighbors()
    order = sorted(graph.nodes, key=lambda n: -len(adj[n]))

    def colorable(k: int) -> bool:
        color: dict[str, int] = {}

        def place(i: int, used: int) -> bool:
            if i == len(order):
                return True
            node = order[i]
            taken = {color[m] for m in adj[node] if m in color}
            for c in range(min(used + 1, k)):
                if c not in taken:
                    color[node] = c
                    if place(i + 1, max(used, c + 1)):
                        return True
                    del color[node]
            return False

        return place(0, 0)

    return next(k for k in range(1, len(graph.nodes) + 1) if colorable(k))


# ---------------------------------------------------------------- Part 2 ---


def timetable_errors(data: ExamData, slot: dict[str, int]) -> list[str]:
    """Return a list of broken rules. An empty list means a valid timetable."""
    errors = [f"{e} has no slot" for e in data.exams if e not in slot]
    errors += [
        f"{e} is in slot {s}, outside 0..{data.num_slots - 1}"
        for e, s in slot.items()
        if not 0 <= s < data.num_slots
    ]
    for k, exams in enumerate(data.enrollments):
        slots = [slot[e] for e in exams if e in slot]
        if len(slots) != len(set(slots)):
            errors.append(f"student {k} has two exams at the same time")
    return errors


def pair_penalty(day_a: int, day_b: int) -> int:
    """Return the penalty for two exams of one student on these days."""
    if day_a == day_b:
        return SAME_DAY_PENALTY
    if abs(day_a - day_b) == 1:
        return NEXT_DAY_PENALTY
    return 0


def student_penalties(data: ExamData, slot: dict[str, int]) -> list[int]:
    """Return the penalty of every student, in enrollment order."""
    day = {e: s // data.slots_per_day for e, s in slot.items()}
    return [
        sum(pair_penalty(day[a], day[b]) for a, b in itertools.combinations(exams, 2))
        for exams in data.enrollments
    ]


@functools.cache
def best_case_penalty(num_exams: int, days: int, per_day: int) -> int:
    """Return the lowest possible penalty for one student with this many exams."""
    plans = (
        p
        for p in itertools.product(range(days), repeat=num_exams)
        if all(p.count(d) <= per_day for d in p)
    )
    return min(
        sum(pair_penalty(a, b) for a, b in itertools.combinations(p, 2)) for p in plans
    )


def penalty_lower_bound(data: ExamData) -> int:
    """Return a lower bound: every student gets at least their best case."""
    return sum(
        best_case_penalty(len(exams), data.days, data.slots_per_day)
        for exams in data.enrollments
    )
