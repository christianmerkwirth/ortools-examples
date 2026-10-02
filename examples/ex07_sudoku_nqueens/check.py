"""Independent checks for Sudoku grids and queen placements. No OR-Tools.

* `sudoku_errors` and `queens_errors` check every rule of a solution.
* `count_sudoku_backtracking` and `count_queens_backtracking` count
  solutions with plain backtracking search. They use a different method
  from CP-SAT, so they serve as a second opinion on the solution counts.
"""

from .data import Grid

ALL_DIGITS = 0b1111111110  # Bits 1 to 9 set: every digit still possible.


def sudoku_errors(grid: Grid, puzzle: Grid | None = None) -> list[str]:
    """Return a list of broken rules. An empty list means a valid solution."""
    errors = []
    if len(grid) != 9 or any(len(row) != 9 for row in grid):
        return ["grid is not 9x9"]
    groups = {}
    for i in range(9):
        groups[f"row {i + 1}"] = [grid[i][c] for c in range(9)]
        groups[f"column {i + 1}"] = [grid[r][i] for r in range(9)]
        top, left = 3 * (i // 3), 3 * (i % 3)
        groups[f"box {i + 1}"] = [
            grid[top + dr][left + dc] for dr in range(3) for dc in range(3)
        ]
    for name, values in groups.items():
        if sorted(values) != list(range(1, 10)):
            errors.append(f"{name} holds {values}, not the digits 1 to 9")
    if puzzle is not None:
        for r in range(9):
            for c in range(9):
                if puzzle[r][c] and puzzle[r][c] != grid[r][c]:
                    errors.append(
                        f"cell ({r + 1},{c + 1}): given {puzzle[r][c]} "
                        f"changed to {grid[r][c]}"
                    )
    return errors


def queens_errors(placement: tuple[int, ...], n: int) -> list[str]:
    """Return a list of attacks. placement[i] is the queen's column in row i."""
    if len(placement) != n:
        return [f"{len(placement)} queens on an {n}x{n} board"]
    errors = [
        f"row {i}: column {q} is off the board"
        for i, q in enumerate(placement)
        if not 0 <= q < n
    ]
    for i in range(n):
        for j in range(i + 1, n):
            same_column = placement[i] == placement[j]
            same_diagonal = abs(placement[i] - placement[j]) == j - i
            if same_column or same_diagonal:
                errors.append(f"queens in rows {i} and {j} attack each other")
    return errors


def count_queens_backtracking(n: int) -> int:
    """Count placements of n queens, row by row, with bit masks.

    `cols`, `down`, and `up` mark attacked columns. The diagonal masks
    shift by one column per row, because a diagonal moves one step
    sideways each time it moves one row down.
    """
    full = (1 << n) - 1

    def place(cols: int, down: int, up: int) -> int:
        if cols == full:
            return 1
        count = 0
        free = full & ~(cols | down | up)
        while free:
            bit = free & -free  # Lowest free column.
            free ^= bit
            count += place(cols | bit, ((down | bit) << 1) & full, (up | bit) >> 1)
        return count

    return place(0, 0, 0)


def count_sudoku_backtracking(puzzle: Grid, limit: int = 10_000) -> int:
    """Count Sudoku solutions with backtracking. Stop at `limit`.

    At each step, fill the empty cell with the fewest possible digits. This
    "most constrained first" rule keeps the search small.
    """
    rows = [0] * 9  # Bit d set: digit d already used in this row.
    cols = [0] * 9
    boxes = [0] * 9
    empty = []
    for r in range(9):
        for c in range(9):
            d = puzzle[r][c]
            if d:
                bit, b = 1 << d, 3 * (r // 3) + c // 3
                if rows[r] & bit or cols[c] & bit or boxes[b] & bit:
                    return 0  # The givens already break a rule.
                rows[r] |= bit
                cols[c] |= bit
                boxes[b] |= bit
            else:
                empty.append((r, c, 3 * (r // 3) + c // 3))

    count = 0

    def search(todo: list) -> bool:
        """Return True when the limit is reached."""
        nonlocal count
        if not todo:
            count += 1
            return count >= limit
        # Pick the cell with the fewest candidates.
        best_k, best_free = 0, ALL_DIGITS
        best_n = 10
        for k, (r, c, b) in enumerate(todo):
            free = ALL_DIGITS & ~(rows[r] | cols[c] | boxes[b])
            n = bin(free).count("1")
            if n < best_n:
                best_k, best_free, best_n = k, free, n
                if n <= 1:
                    break
        r, c, b = todo[best_k]
        rest = todo[:best_k] + todo[best_k + 1 :]
        while best_free:
            bit = best_free & -best_free
            best_free ^= bit
            rows[r] |= bit
            cols[c] |= bit
            boxes[b] |= bit
            done = search(rest)
            rows[r] ^= bit
            cols[c] ^= bit
            boxes[b] ^= bit
            if done:
                return True
        return False

    search(empty)
    return count
