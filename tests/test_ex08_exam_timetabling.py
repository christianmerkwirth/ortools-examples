"""Tests for example 08: exam timetabling as graph coloring."""

import itertools

import pytest

from examples.ex08_exam_timetabling.bounds import dsatur, max_clique, maximal_cliques
from examples.ex08_exam_timetabling.check import (
    best_case_penalty,
    chromatic_number,
    coloring_errors,
    is_clique,
    num_colors,
    penalty_lower_bound,
    student_penalties,
    timetable_errors,
)
from examples.ex08_exam_timetabling.data import (
    ExamData,
    complete_graph,
    cycle_graph,
    mycielski_graph,
    petersen_graph,
    queen_graph,
    random_graph,
    school,
)
from examples.ex08_exam_timetabling.model import (
    color_boolean,
    color_integer,
    min_student_penalty,
    spread_exams,
)

# Graphs with a known chromatic number (from the literature).
KNOWN = [
    (petersen_graph(), 3),
    (mycielski_graph(3), 4),
    (mycielski_graph(4), 5),
    (cycle_graph(7), 3),
    (cycle_graph(8), 2),
    (complete_graph(6), 6),
    (queen_graph(5), 5),
]
SOLVERS = [color_integer, color_boolean]


@pytest.fixture(scope="module")
def data():
    return school()


@pytest.fixture(scope="module")
def graph(data):
    return data.conflict_graph()


# ----------------------------------------------------------- data, bounds ---


def test_school_instance_shape(data, graph):
    assert len(data.exams) == 42
    assert len(data.enrollments) == 300
    assert len(graph.edges) == 291
    # Every pair of exams of one student is an edge.
    edges = {frozenset(e) for e in graph.edges}
    for exams in data.enrollments:
        for a, b in itertools.combinations(exams, 2):
            assert frozenset((a, b)) in edges


def test_mycielski_graphs_are_triangle_free():
    for k, size in ((3, 11), (4, 23), (5, 47)):
        g = mycielski_graph(k)
        assert len(g.nodes) == size
        assert len(max_clique(g)) == 2


@pytest.mark.parametrize("seed", range(5))
def test_bounds_on_random_graphs(seed):
    g = random_graph(11, 0.45, seed)
    clique = max_clique(g)
    assert is_clique(g, clique)
    # Brute force: no larger clique exists.
    assert not any(
        is_clique(g, list(c)) for c in itertools.combinations(g.nodes, len(clique) + 1)
    )
    assert coloring_errors(g, dsatur(g)) == []
    assert max(len(c) for c in maximal_cliques(g)) == len(clique)


@pytest.mark.parametrize("seed", range(3))
def test_backtracking_matches_brute_force(seed):
    g = random_graph(7, 0.5, seed)
    brute = min(
        k
        for k in range(1, 8)
        for colors in itertools.product(range(k), repeat=7)
        if not coloring_errors(g, dict(zip(g.nodes, colors, strict=True)))
    )
    assert chromatic_number(g) == brute


# ---------------------------------------------------------------- Part 1 ---


@pytest.mark.parametrize("solve", SOLVERS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("g, chi", KNOWN, ids=lambda x: getattr(x, "name", str(x)))
def test_known_chromatic_numbers(solve, g, chi):
    result = solve(g, time_limit=10)
    assert coloring_errors(g, result.color) == []
    assert result.optimal
    assert result.num_colors == num_colors(result.color) == chi
    assert chromatic_number(g) == chi


@pytest.mark.parametrize("solve", SOLVERS, ids=lambda f: f.__name__)
def test_school_needs_nine_slots(graph, solve):
    result = solve(graph, symmetry_breaking=True, time_limit=10)
    assert coloring_errors(graph, result.color) == []
    assert result.optimal
    assert result.num_colors == result.lower_bound == 9
    assert chromatic_number(graph) == 9
    # The bounds sandwich the answer: clique 8 <= 9 <= DSATUR 9.
    assert len(max_clique(graph)) == 8
    assert num_colors(dsatur(graph)) == 9


@pytest.mark.parametrize("seed", range(3))
def test_symmetry_breaking_keeps_the_optimum(seed):
    g = random_graph(16, 0.4, seed)
    chi = chromatic_number(g)
    for solve in SOLVERS:
        for symmetry in (False, True):
            result = solve(g, symmetry_breaking=symmetry, time_limit=10)
            assert result.optimal
            assert result.num_colors == chi
            assert coloring_errors(g, result.color) == []


# ---------------------------------------------------------------- Part 2 ---


def test_student_floor_agrees_with_checker():
    for k in range(1, 7):
        assert min_student_penalty(k, 5, 2) == best_case_penalty(k, 5, 2)
    assert [best_case_penalty(k, 5, 2) for k in range(1, 7)] == [0, 0, 0, 2, 4, 8]


TINY = ExamData(
    exams=("A", "B", "C", "D", "E", "F"),
    enrollments=(
        ("A", "B", "C"),
        ("A", "B", "D"),
        ("C", "D", "E"),
        ("B", "E", "F"),
        ("A", "F"),
        ("A", "C", "E", "F"),
    ),
    days=3,
    slots_per_day=2,
)


def test_tiny_week_matches_brute_force():
    best = None
    for slots in itertools.product(range(TINY.num_slots), repeat=len(TINY.exams)):
        slot = dict(zip(TINY.exams, slots, strict=True))
        if not timetable_errors(TINY, slot):
            penalty = sum(student_penalties(TINY, slot))
            best = penalty if best is None else min(best, penalty)
    for cuts in (False, True):
        week = spread_exams(TINY, time_limit=10, student_cuts=cuts, clique_cuts=cuts)
        assert week.optimal
        assert week.penalty == best == sum(student_penalties(TINY, week.slot))
        assert timetable_errors(TINY, week.slot) == []


def test_school_week_is_valid_and_better_than_greedy(data, graph):
    week = spread_exams(data, time_limit=5)
    assert timetable_errors(data, week.slot) == []
    assert week.penalty == sum(student_penalties(data, week.slot))
    assert penalty_lower_bound(data) <= week.lower_bound <= week.penalty
    greedy = dsatur(graph)  # DSATUR slots, used in order.
    assert sum(student_penalties(data, greedy)) == 1663
    assert week.penalty < 0.75 * 1663
