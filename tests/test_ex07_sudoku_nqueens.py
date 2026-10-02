"""Tests for example 07: Sudoku and N-Queens."""

import pytest

from examples.ex07_sudoku_nqueens.check import (
    count_queens_backtracking,
    count_sudoku_backtracking,
    queens_errors,
    sudoku_errors,
)
from examples.ex07_sudoku_nqueens.data import (
    INKALA_2012,
    INKALA_2012_SOLUTION,
    KNOWN_QUEENS_COUNTS,
    parse,
    to_text,
    without_cell,
)
from examples.ex07_sudoku_nqueens.model import (
    all_queens,
    count_queens,
    count_sudoku_solutions,
    find_second_solution,
    is_unique,
    make_puzzle,
    solve_queens,
    solve_sudoku,
)

PUZZLE = INKALA_2012.grid


# ---------------------------------------------------------------- Sudoku ---


def test_published_solution_is_valid():
    # The referee itself must accept the published answer.
    assert sudoku_errors(INKALA_2012_SOLUTION, PUZZLE) == []


def test_solver_finds_the_published_solution():
    solution = solve_sudoku(PUZZLE)
    assert sudoku_errors(solution, PUZZLE) == []
    assert solution == INKALA_2012_SOLUTION


def test_solution_is_unique_two_ways():
    first = solve_sudoku(PUZZLE)
    assert find_second_solution(PUZZLE, first) is None
    assert count_sudoku_solutions(PUZZLE) == 1
    assert count_sudoku_backtracking(PUZZLE) == 1


def test_puzzle_is_minimal():
    """Every clue is needed: removing any one gives more than one solution."""
    for r, row in enumerate(PUZZLE):
        for c, v in enumerate(row):
            if v:
                assert count_sudoku_solutions(without_cell(PUZZLE, r, c), limit=2) == 2


def test_missing_clue_counts_agree():
    smaller = without_cell(PUZZLE, 0, 0)
    assert count_sudoku_solutions(smaller) == 292
    assert count_sudoku_backtracking(smaller) == 292
    second = find_second_solution(smaller, INKALA_2012_SOLUTION)
    assert second is not None and second != INKALA_2012_SOLUTION
    assert sudoku_errors(second, smaller) == []
    assert not is_unique(smaller)


def test_contradictory_puzzle_has_no_solution():
    # Two 8s in the first row.
    bad = parse("88" + "0" * 79)
    assert solve_sudoku(bad) is None
    assert count_sudoku_solutions(bad) == 0
    assert count_sudoku_backtracking(bad) == 0


def test_unsolvable_puzzle_without_obvious_clash():
    # No digit repeats in any row, column, or box, yet the top-left cell
    # cannot hold any digit: 1-6 are in its row, 7-9 in its column.
    text = ["0123456" + "00", "7" + "0" * 8, "8" + "0" * 8, "9" + "0" * 8]
    bad = parse("".join(text) + "0" * 45)
    assert solve_sudoku(bad) is None
    assert count_sudoku_backtracking(bad) == 0


def test_generated_puzzle_is_unique_and_minimal():
    puzzle = make_puzzle(INKALA_2012_SOLUTION, seed=7)
    assert count_sudoku_backtracking(puzzle, limit=2) == 1
    assert solve_sudoku(puzzle) == INKALA_2012_SOLUTION
    for r, row in enumerate(puzzle):
        for c, v in enumerate(row):
            if v:
                assert (
                    count_sudoku_backtracking(without_cell(puzzle, r, c), limit=2) == 2
                )


def test_checker_catches_errors():
    broken = [list(row) for row in INKALA_2012_SOLUTION]
    broken[0][1], broken[0][2] = broken[0][2], broken[0][1]  # Swap two cells.
    errors = sudoku_errors(tuple(map(tuple, broken)), PUZZLE)
    assert any("column" in e for e in errors)
    # Changing a given digit is reported even if the grid were valid.
    assert sudoku_errors(INKALA_2012_SOLUTION, parse("9" + "0" * 80))


def test_text_round_trip():
    assert parse(to_text(PUZZLE)) == PUZZLE
    with pytest.raises(ValueError):
        parse("123")


# -------------------------------------------------------------- N-Queens ---


@pytest.mark.parametrize("n", [1, 4, 5, 8, 20, 50])
def test_single_placement_is_valid(n):
    placement = solve_queens(n)
    assert queens_errors(placement, n) == []


@pytest.mark.parametrize("n", [2, 3])
def test_no_placement_exists(n):
    assert solve_queens(n) is None
    assert all_queens(n) == []


@pytest.mark.parametrize("n", range(1, 11))
def test_cp_sat_counts_match_oeis(n):
    assert count_queens(n) == KNOWN_QUEENS_COUNTS[n]


@pytest.mark.parametrize("n", range(1, 13))
def test_backtracking_counter_matches_oeis(n):
    assert count_queens_backtracking(n) == KNOWN_QUEENS_COUNTS[n]


def test_all_8_queens_solutions_are_distinct_and_valid():
    solutions = all_queens(8)
    assert len(solutions) == 92
    assert len(set(solutions)) == 92
    assert all(queens_errors(s, 8) == [] for s in solutions)


def test_enumeration_respects_limit():
    assert len(all_queens(8, limit=5)) == 5


def test_queens_checker_catches_attacks():
    assert queens_errors((0, 1, 2, 3), 4)  # All on one diagonal.
    assert queens_errors((1, 1, 3, 0), 4)  # Same column.
    assert queens_errors((1, 3, 0), 4)  # Too few queens.
    assert queens_errors((1, 3, 0, 2), 4) == []
