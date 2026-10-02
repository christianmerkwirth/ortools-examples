"""Instance data for the Sudoku and N-Queens example.

A Sudoku grid is a 9x9 tuple of tuples. The digit 0 marks an empty cell.
Puzzles are written as 81-character strings, row by row, which is the
usual exchange format for Sudoku collections.
"""

from dataclasses import dataclass

Grid = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class Puzzle:
    """A Sudoku puzzle with a name and a note on where it comes from."""

    name: str
    grid: Grid
    source: str = ""

    @property
    def clues(self) -> int:
        """Return the number of given digits."""
        return sum(v != 0 for row in self.grid for v in row)


def parse(text: str) -> Grid:
    """Turn an 81-character string into a grid. '0' or '.' marks a blank."""
    digits = [0 if ch in "0." else int(ch) for ch in text if not ch.isspace()]
    if len(digits) != 81:
        raise ValueError(f"expected 81 cells, got {len(digits)}")
    return tuple(tuple(digits[9 * r : 9 * r + 9]) for r in range(9))


def to_text(grid: Grid) -> str:
    """Turn a grid back into an 81-character string."""
    return "".join(str(v) for row in grid for v in row)


def without_cell(grid: Grid, row: int, col: int) -> Grid:
    """Return a copy of the grid with one cell made blank."""
    return tuple(
        tuple(0 if (r, c) == (row, col) else v for c, v in enumerate(cells))
        for r, cells in enumerate(grid)
    )


# Published in 2012 by the Finnish mathematician Arto Inkala, and often
# called "the world's hardest Sudoku". It has 21 clues and one solution.
INKALA_2012 = Puzzle(
    "Inkala 2012",
    parse(
        """
        800 000 000
        003 600 000
        070 090 200
        050 007 000
        000 045 700
        000 100 030
        001 000 068
        008 500 010
        090 000 400
        """
    ),
    source="A. Inkala, 2012",
)

# The published solution of the puzzle above. The tests compare with it.
INKALA_2012_SOLUTION = parse(
    """
    812 753 649
    943 682 175
    675 491 283
    154 237 896
    369 845 721
    287 169 534
    521 974 368
    438 526 917
    796 318 452
    """
)

# Number of ways to place n non-attacking queens on an n x n board.
# OEIS sequence A000170.
KNOWN_QUEENS_COUNTS = {
    1: 1,
    2: 0,
    3: 0,
    4: 2,
    5: 10,
    6: 4,
    7: 40,
    8: 92,
    9: 352,
    10: 724,
    11: 2680,
    12: 14200,
}
