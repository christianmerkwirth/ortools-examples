"""Packing models with CP-SAT: bin packing (1D) and strip packing (2D).

Part 1, bin packing. Decision: which van carries which crate.
Goal: use as few vans as possible. Limit: the van capacity.

Part 2, strip packing. Decision: where to place each panel on the roll,
and whether to turn it by 90 degrees. Goal: use as little roll length as
possible. Limits: panels stay inside the roll width and do not overlap.
"""

import time
from dataclasses import dataclass

from ortools.sat.python import cp_model

from .bounds import first_fit_decreasing, grid_size
from .data import BinPackingData, StripData

# ---------------------------------------------------------------- Part 1 ---


@dataclass(frozen=True)
class BinPacking:
    """A bin packing solution."""

    bins: list[list[str]]  # Item names per used bin.
    bound: int  # Best lower bound that CP-SAT proved.
    proven_optimal: bool
    seconds: float
    work: float  # CP-SAT's deterministic time: the same on every machine.

    @property
    def n_bins(self) -> int:
        """Return the number of used bins."""
        return len(self.bins)


def solve_bin_packing(
    data: BinPackingData,
    symmetry_breaking: bool = True,
    time_limit: float = 10.0,
    workers: int = 8,
    work_limit: float | None = None,
) -> BinPacking:
    """Pack the items into as few bins as possible.

    Without symmetry breaking, every packing appears many times: swap the
    contents of two bins and you get a "new" solution with the same cost.
    To prove that no better packing exists, the solver must rule out all
    of these copies. Symmetry breaking keeps only one copy of each packing:

    * Sort the items from largest to smallest. Item k may only go into
      bins 0..k. (Number the bins by their largest item.)
    * Use bins in order: bin b+1 is only used if bin b is used.
    * Items larger than half the capacity can never share a bin, so the
      k-th such item goes straight into bin k.

    With `workers=1` and a `work_limit`, a run gives the same result on
    every machine. CP-SAT measures this "deterministic time" in work units
    that do not depend on the speed of the computer.
    """
    names = sorted(data.sizes, key=lambda n: -data.sizes[n])
    size = [data.sizes[n] for n in names]
    # First-fit decreasing gives an upper bound on the bins we need.
    n_bins = len(first_fit_decreasing(data.sizes, data.capacity))
    allowed = {
        (i, b)
        for i in range(len(names))
        for b in range(n_bins)
        if not symmetry_breaking or b <= i
    }

    model = cp_model.CpModel()
    x = {(i, b): model.new_bool_var(f"{names[i]} in bin {b}") for i, b in allowed}
    used = [model.new_bool_var(f"bin {b} used") for b in range(n_bins)]

    for i in range(len(names)):
        model.add_exactly_one(x[i, b] for b in range(n_bins) if (i, b) in allowed)
    for b in range(n_bins):
        # The load must fit, and an unused bin must stay empty.
        load = sum(size[i] * x[i, b] for i in range(len(names)) if (i, b) in allowed)
        model.add(load <= data.capacity * used[b])

    if symmetry_breaking:
        for b in range(n_bins - 1):
            model.add_implication(used[b + 1], used[b])
        for i, s in enumerate(size):
            if s > data.capacity / 2:
                model.add(x[i, i] == 1)

    model.minimize(sum(used))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = workers
    if work_limit is not None:
        solver.parameters.max_deterministic_time = work_limit
    start = time.perf_counter()
    status = solver.solve(model)
    seconds = time.perf_counter() - start
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"CP-SAT found no packing (status {status}).")

    bins = [
        [names[i] for i in range(len(names)) if (i, b) in x and solver.value(x[i, b])]
        for b in range(n_bins)
    ]
    return BinPacking(
        bins=[b for b in bins if b],
        bound=round(solver.best_objective_bound),
        proven_optimal=status == cp_model.OPTIMAL,
        seconds=seconds,
        work=solver.deterministic_time,
    )


# ---------------------------------------------------------------- Part 2 ---


@dataclass(frozen=True)
class Placement:
    """Where one panel goes on the strip, in cm."""

    x: int  # Distance from the left edge.
    y: int  # Distance from the start of the roll.
    width: int  # Size across the roll, after rotation.
    height: int  # Size along the roll, after rotation.
    rotated: bool


@dataclass(frozen=True)
class StripPacking:
    """A strip packing solution."""

    length: int  # Roll length used, in cm.
    placements: dict[str, Placement]
    bound: int  # Best lower bound that CP-SAT proved, in cm.
    proven_optimal: bool
    seconds: float


def solve_strip_packing(
    data: StripData,
    allow_rotation: bool = True,
    scale_to_grid: bool = True,
    time_limit: float = 10.0,
) -> StripPacking:
    """Place all panels on the strip and minimize the used length.

    Each panel gets one x and one y variable. For every orientation the
    panel may take, we create an *optional* interval on each axis. A
    Boolean literal decides if that orientation is present. Exactly one
    orientation is present, and `add_no_overlap_2d` ignores the absent
    ones. This is how CP-SAT models "choose one of several shapes".

    With `scale_to_grid`, all sizes are divided by their greatest common
    divisor first. A 10 cm grid gives the solver 10 times smaller domains
    and the same optimum (see `bounds.strip_lower_bound` for why).
    """
    panels = data.panels
    dims = [data.strip_width, *(d for p in panels for d in (p.width, p.height))]
    g = grid_size(dims) if scale_to_grid else 1
    strip_width = data.strip_width // g
    max_length = sum(max(p.width, p.height) for p in panels) // g

    model = cp_model.CpModel()
    length = model.new_int_var(0, max_length, "length")
    x_intervals, y_intervals = [], []
    pos, choice = {}, {}
    for p in panels:
        w, h = p.width // g, p.height // g
        x = model.new_int_var(0, strip_width, f"x {p.name}")
        y = model.new_int_var(0, max_length, f"y {p.name}")
        pos[p.name] = (x, y)
        shapes = [(w, h, False)]
        if allow_rotation and p.can_rotate and w != h:
            shapes.append((h, w, True))
        literals = []
        for sw, sh, rotated in shapes:
            if sw > strip_width:
                continue  # This orientation does not fit across the roll.
            present = model.new_bool_var(f"{p.name} rotated={rotated}")
            x_intervals.append(
                model.new_optional_fixed_size_interval_var(x, sw, present, "")
            )
            y_intervals.append(
                model.new_optional_fixed_size_interval_var(y, sh, present, "")
            )
            # Stay inside the roll width and below the used length.
            model.add(x + sw <= strip_width).only_enforce_if(present)
            model.add(y + sh <= length).only_enforce_if(present)
            literals.append(present)
            choice[p.name, rotated] = present
        model.add_exactly_one(literals)

    model.add_no_overlap_2d(x_intervals, y_intervals)
    model.minimize(length)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 8
    start = time.perf_counter()
    status = solver.solve(model)
    seconds = time.perf_counter() - start
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"CP-SAT found no packing (status {status}).")

    placements = {}
    for p in panels:
        rotated = (p.name, True) in choice and solver.boolean_value(
            choice[p.name, True]
        )
        x, y = pos[p.name]
        width, height = (p.height, p.width) if rotated else (p.width, p.height)
        placements[p.name] = Placement(
            solver.value(x) * g, solver.value(y) * g, width, height, rotated
        )
    return StripPacking(
        length=round(solver.objective_value) * g,
        placements=placements,
        bound=round(solver.best_objective_bound) * g,
        proven_optimal=status == cp_model.OPTIMAL,
        seconds=seconds,
    )
