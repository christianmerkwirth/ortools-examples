"""Solve Sudoku and N-Queens with CP-SAT, and count their solutions.

Run from the repository root:

    uv run python -m examples.ex07_sudoku_nqueens.main
    uv run python -m examples.ex07_sudoku_nqueens.main --plot --max-n 12
"""

import argparse
import time
from pathlib import Path

from examples.common.report import format_table

from .check import (
    count_queens_backtracking,
    count_sudoku_backtracking,
    queens_errors,
    sudoku_errors,
)
from .data import (
    INKALA_2012,
    INKALA_2012_SOLUTION,
    KNOWN_QUEENS_COUNTS,
    Grid,
    without_cell,
)
from .model import (
    all_queens,
    count_queens,
    count_sudoku_solutions,
    find_second_solution,
    make_puzzle,
    solve_queens,
    solve_sudoku,
)

FIGURES = Path(__file__).parent / "figures"


def format_grid(grid: Grid) -> str:
    """Return a grid as text, with lines between the 3x3 boxes."""
    lines = []
    for r, row in enumerate(grid):
        if r in (3, 6):
            lines.append("------+-------+------")
        cells = [str(v) if v else "." for v in row]
        lines.append(" | ".join(" ".join(cells[k : k + 3]) for k in (0, 3, 6)))
    return "\n".join(lines)


def side_by_side(left: str, right: str, gap: str = "      ") -> str:
    """Join two blocks of text line by line."""
    return "\n".join(
        a + gap + b for a, b in zip(left.splitlines(), right.splitlines(), strict=True)
    )


def sudoku_part() -> tuple[list[str], dict]:
    """Solve, prove unique, test minimality, and generate. Return errors and data."""
    errors = []
    puzzle = INKALA_2012.grid
    start = time.perf_counter()
    solution = solve_sudoku(puzzle)
    seconds = time.perf_counter() - start
    print(f"Sudoku: '{INKALA_2012.name}' ({INKALA_2012.clues} clues)\n")
    print(side_by_side(format_grid(puzzle), format_grid(solution)))
    print(f"\nSolved in {seconds:.3f} s.")
    errors += sudoku_errors(solution, puzzle)
    if solution != INKALA_2012_SOLUTION:
        errors.append("solution differs from the published one")

    second = find_second_solution(puzzle, solution)
    print(
        "Uniqueness: the model 'solution must differ in some cell' is "
        + ("infeasible, so the solution is unique." if second is None else "FEASIBLE.")
    )
    if second is not None:
        errors.append("puzzle has a second solution")

    # Remove one clue at a time and count the solutions of what is left.
    counts = {}
    for r, row in enumerate(puzzle):
        for c, v in enumerate(row):
            if v:
                counts[r, c] = count_sudoku_solutions(without_cell(puzzle, r, c))
    print(
        f"Minimality: remove any one clue and the puzzle has between "
        f"{min(counts.values()):,} and {max(counts.values()):,} solutions."
    )
    (r0, c0) = (0, 0)
    backtrack = count_sudoku_backtracking(without_cell(puzzle, r0, c0))
    print(
        f"  Without the {puzzle[r0][c0]} in the top-left corner: "
        f"{counts[r0, c0]} solutions (CP-SAT), {backtrack} (backtracking check)."
    )
    if backtrack != counts[r0, c0]:
        errors.append("solution counts of CP-SAT and backtracking differ")
    if min(counts.values()) < 2:
        errors.append("puzzle is not minimal")

    start = time.perf_counter()
    new = make_puzzle(solution, seed=7)
    seconds = time.perf_counter() - start
    clues = sum(v != 0 for row in new for v in row)
    print(
        f"\nNew puzzle from the same solution "
        f"({clues} clues, built in {seconds:.1f} s):\n"
    )
    print(format_grid(new))
    if count_sudoku_backtracking(new, limit=2) != 1:
        errors.append("generated puzzle is not unique")
    return errors, {"puzzle": puzzle, "solution": solution, "counts": counts}


def queens_part(max_n: int) -> tuple[list[str], dict]:
    """Place 8 queens, then count all placements for each n. Return errors and data."""
    errors = []
    placement = solve_queens(8)
    print("\n8 queens, one solution (Q = queen):\n")
    for col in placement:
        print("  " + " ".join("Q" if c == col else "." for c in range(8)))
    errors += queens_errors(placement, 8)

    solutions = all_queens(8)
    bad = sum(1 for s in solutions if queens_errors(s, 8))
    print(
        f"\nAll 8-queens solutions: {len(solutions)} found, "
        f"{len(set(solutions))} distinct, {bad} invalid."
    )
    if bad or len(set(solutions)) != len(solutions):
        errors.append("enumeration returned invalid or repeated solutions")

    print("\nNumber of solutions by board size\n")
    rows, counts = [], {}
    for n in range(1, max_n + 1):
        start = time.perf_counter()
        cp = count_queens(n)
        seconds = time.perf_counter() - start
        bt = count_queens_backtracking(n)
        known = KNOWN_QUEENS_COUNTS.get(n)
        counts[n] = cp
        rows.append((n, cp, bt, known if known is not None else "?", round(seconds, 3)))
        if cp != bt or (known is not None and cp != known):
            errors.append(f"n={n}: counts disagree ({cp}, {bt}, {known})")
    print(
        format_table(["n", "CP-SAT", "backtracking", "OEIS A000170", "CP-SAT s"], rows)
    )
    return errors, {"placement": placement, "counts": counts}


def plot_sudoku(sudoku: dict) -> Path:
    """Two grids: the solved puzzle, and the effect of removing each clue."""
    import matplotlib.colors as mcolors

    from examples.common.plotting import COLORS, new_figure, save

    fig, (left, right) = new_figure(10, 5, ncols=2)
    puzzle, solution, counts = sudoku["puzzle"], sudoku["solution"], sudoku["counts"]

    _draw_grid(left)
    for r in range(9):
        for c in range(9):
            given = puzzle[r][c] != 0
            if given:
                left.add_patch(_cell_patch(r, c, "0.88"))
            left.text(
                c + 0.5,
                8.5 - r,
                str(solution[r][c]),
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold" if given else "normal",
                color="black" if given else COLORS[0],
            )
    left.set_title(f"Solved: {len(counts)} given clues (gray), the rest found")

    _draw_grid(right)
    norm = mcolors.LogNorm(vmin=min(counts.values()), vmax=max(counts.values()))
    cmap = mcolors.LinearSegmentedColormap.from_list("counts", ["#FFF4D6", COLORS[3]])
    for (r, c), n in counts.items():
        right.add_patch(_cell_patch(r, c, cmap(norm(n))))
        right.text(c + 0.5, 8.5 - r, f"{n:,}", ha="center", va="center", fontsize=7.5)
    right.set_title("Solutions left if this one clue is removed")
    return save(fig, FIGURES / "sudoku.png")


def _draw_grid(ax) -> None:
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 9)
    ax.set_aspect("equal")
    ax.axis("off")
    for k in range(10):
        lw = 2.2 if k % 3 == 0 else 0.6
        ax.plot([0, 9], [k, k], color="black", lw=lw)
        ax.plot([k, k], [0, 9], color="black", lw=lw)


def _cell_patch(r: int, c: int, color):
    from matplotlib.patches import Rectangle

    return Rectangle((c, 8 - r), 1, 1, facecolor=color, edgecolor="none", zorder=0)


def plot_queens(queens: dict) -> Path:
    """Plot an 8-queens board and the number of solutions by board size."""
    from examples.common.plotting import COLORS, new_figure, save

    fig, (board, bars) = new_figure(10, 4.5, ncols=2, width_ratios=[1, 1.4])
    placement = queens["placement"]
    n = len(placement)
    from matplotlib.patches import Rectangle

    for r in range(n):
        for c in range(n):
            shade = "#EEE8DC" if (r + c) % 2 == 0 else "#B9A88C"
            board.add_patch(Rectangle((c, n - 1 - r), 1, 1, facecolor=shade))
    for r, c in enumerate(placement):
        board.text(c + 0.5, n - 0.5 - r, "♛", ha="center", va="center", fontsize=24)
    board.set_xlim(0, n)
    board.set_ylim(0, n)
    board.set_aspect("equal")
    board.axis("off")
    board.set_title("One of the 92 ways to place 8 queens")

    counts = queens["counts"]
    sizes = list(counts)
    bars.bar(sizes, [counts[k] for k in sizes], color=COLORS[0], label="CP-SAT count")
    known = [
        k for k in sizes if k in KNOWN_QUEENS_COUNTS and KNOWN_QUEENS_COUNTS[k] > 0
    ]
    bars.plot(
        known,
        [KNOWN_QUEENS_COUNTS[k] for k in known],
        "o",
        color=COLORS[3],
        label="OEIS A000170",
    )
    for k in sizes:
        bars.text(
            k, max(counts[k], 1) * 1.15, f"{counts[k]:,}", ha="center", fontsize=8
        )
    bars.set_yscale("log")
    bars.set_ylim(0.8, max(counts.values()) * 4)
    bars.set_xticks(sizes)
    bars.set_xlabel("board size n")
    bars.set_ylabel("number of solutions (log scale)")
    bars.set_title("All solutions, found by enumeration")
    bars.legend(loc="upper left")
    return save(fig, FIGURES / "queens.png")


def main() -> None:
    """Run both parts, verify every result, and report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plot", action="store_true", help="write PNG figures")
    parser.add_argument(
        "--max-n",
        type=int,
        default=11,
        help="largest board for the queens count (default 11)",
    )
    args = parser.parse_args()

    errors, sudoku = sudoku_part()
    print("\n" + "=" * 72)
    queens_errors_, queens = queens_part(args.max_n)
    errors += queens_errors_

    if errors:
        print("\nVerification FAILED:", *errors, sep="\n  ")
    else:
        print("\nVerification: every solution obeys all rules, and all counts agree.")
    if args.plot:
        for path in (plot_sudoku(sudoku), plot_queens(queens)):
            print("Wrote", path)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
