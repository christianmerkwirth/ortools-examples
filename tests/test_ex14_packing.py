"""Tests for example 14: bin packing and 2D strip packing."""

from dataclasses import replace

import pytest

from examples.ex14_packing.bounds import (
    first_fit_decreasing,
    l1_bound,
    l2_bound,
    strip_lower_bound,
)
from examples.ex14_packing.check import (
    bin_packing_errors,
    min_bins_brute_force,
    strip_packing_errors,
)
from examples.ex14_packing.data import (
    BinPackingData,
    Panel,
    StripData,
    crates,
    guillotine_instance,
    random_bin_packing,
    sign_order,
    week,
)
from examples.ex14_packing.model import (
    Placement,
    solve_bin_packing,
    solve_strip_packing,
)

# ---------------------------------------------------------------- Part 1 ---

# Day -> (L1, L2, first-fit decreasing, optimum). See the README table.
WEEK = {
    "Mon": (14, 14, 15, 14),
    "Tue": (15, 15, 16, 16),
    "Wed": (16, 18, 18, 18),
    "Thu": (13, 13, 14, 14),
    "Fri": (15, 15, 15, 15),
}


@pytest.mark.parametrize("day", WEEK)
def test_week_bounds_and_optimum(day):
    data = crates(day)
    l1, l2, ffd, optimum = WEEK[day]
    assert l1_bound(data.sizes, data.capacity) == l1
    assert l2_bound(data.sizes, data.capacity) == l2
    ffd_bins = first_fit_decreasing(data.sizes, data.capacity)
    assert len(ffd_bins) == ffd
    assert bin_packing_errors(data, ffd_bins) == []

    result = solve_bin_packing(data)
    assert bin_packing_errors(data, result.bins) == []
    assert result.proven_optimal
    assert result.n_bins == result.bound == optimum


@pytest.mark.parametrize("day", ["Mon", "Tue", "Wed", "Thu"])
def test_symmetry_breaking_decides_the_proof(day):
    """One worker and a fixed work limit make this test deterministic."""
    data = crates(day)
    with_sb = solve_bin_packing(data, True, workers=1, work_limit=0.5)
    plain = solve_bin_packing(data, False, workers=1, work_limit=0.5)
    assert with_sb.proven_optimal
    assert with_sb.work < 0.1
    assert not plain.proven_optimal
    # The plain model either misses the optimum (Mon) or cannot prove it.
    assert plain.bound <= with_sb.n_bins <= plain.n_bins
    assert plain.bound < plain.n_bins


def test_week_has_five_different_days():
    assert [d.name for d in week()] == [f"{day} crates" for day in WEEK]
    assert all(len(d.sizes) == 36 for d in week())


@pytest.mark.parametrize("seed", range(8))
def test_cp_sat_matches_brute_force(seed):
    data = random_bin_packing(9, 100, seed)
    optimum = min_bins_brute_force(data.sizes, data.capacity)
    for sb in (True, False):
        result = solve_bin_packing(data, sb)
        assert bin_packing_errors(data, result.bins) == []
        assert result.n_bins == optimum
    assert l1_bound(data.sizes, data.capacity) <= l2_bound(data.sizes, data.capacity)
    assert l2_bound(data.sizes, data.capacity) <= optimum
    assert len(first_fit_decreasing(data.sizes, data.capacity)) >= optimum


@pytest.mark.parametrize(
    ("sizes", "l1", "l2", "optimum"),
    [
        ([6, 6, 6, 6], 3, 4, 4),  # No two items > 5 can share a bin.
        ([6, 6, 6, 4, 4, 4], 3, 3, 3),  # Each 6 pairs with a 4.
        ([7, 7, 3, 3, 3, 3], 3, 3, 3),
        ([5, 5, 5, 5, 5], 3, 3, 3),
    ],
)
def test_bounds_on_hand_worked_cases(sizes, l1, l2, optimum):
    named = {f"i{k}": s for k, s in enumerate(sizes)}
    assert l1_bound(named, 10) == l1
    assert l2_bound(named, 10) == l2
    assert min_bins_brute_force(named, 10) == optimum


def test_checker_catches_broken_bins():
    data = BinPackingData("tiny", {"a": 6, "b": 5, "c": 3}, 10)
    assert bin_packing_errors(data, [["a", "b"], ["c"]]) == [
        "bin 0: load 11 > capacity 10"
    ]
    assert bin_packing_errors(data, [["a"], ["b"]]) == ["not packed: ['c']"]
    assert "a packed 2 times" in bin_packing_errors(data, [["a", "c"], ["a", "b"]])


# ---------------------------------------------------------------- Part 2 ---


@pytest.fixture(scope="module")
def order():
    return sign_order()


@pytest.mark.parametrize(("rotate", "length"), [(False, 260), (True, 250)])
def test_sign_order(order, rotate, length):
    result = solve_strip_packing(order, allow_rotation=rotate)
    assert strip_packing_errors(order, result.placements, result.length, rotate) == []
    assert result.proven_optimal
    assert result.length == result.bound == length
    if not rotate:
        assert not any(q.rotated for q in result.placements.values())


def test_rotation_result_meets_our_own_bound(order):
    """250 cm equals the area-and-grid bound, so it is optimal without CP-SAT."""
    result = solve_strip_packing(order, allow_rotation=True)
    assert result.length == strip_lower_bound(order.panels, order.strip_width) == 250


def test_one_cm_grid_gives_the_same_layout_quality(order):
    result = solve_strip_packing(order, scale_to_grid=False, time_limit=20)
    assert strip_packing_errors(order, result.placements, result.length) == []
    assert result.length == 250


@pytest.mark.parametrize("seed", range(5))
def test_guillotine_instances_reach_the_known_optimum(seed):
    """Pieces cut from a 30 x 20 rectangle fit back into length 20."""
    data = guillotine_instance(30, 20, 8, seed)
    assert strip_lower_bound(data.panels, data.strip_width) == 20
    result = solve_strip_packing(data, time_limit=20)
    assert strip_packing_errors(data, result.placements, result.length) == []
    assert result.length == 20
    assert result.proven_optimal


def test_guillotine_pieces_fill_the_rectangle():
    data = guillotine_instance(120, 100, 12, seed=3)
    assert sum(p.width * p.height for p in data.panels) == 120 * 100
    assert len(data.panels) == 12


def test_fixed_panels_are_never_rotated(order):
    fixed = {"window display", "exhibition panel", "parking sign"}
    panels = tuple(replace(p, can_rotate=p.name not in fixed) for p in order.panels)
    data = replace(order, panels=panels)
    result = solve_strip_packing(data)
    assert strip_packing_errors(data, result.placements, result.length) == []
    assert not any(result.placements[name].rotated for name in fixed)


def test_panel_wider_than_the_roll():
    data = StripData("too wide", (Panel("banner", 130, 10, can_rotate=False),), 120)
    with pytest.raises(ValueError):
        strip_lower_bound(data.panels, data.strip_width)
    # The model checks the bound first, so it rejects the panel before
    # any search starts.
    with pytest.raises(ValueError):
        solve_strip_packing(data)
    # Turned by 90 degrees, the banner fits.
    turnable = StripData("turnable", (Panel("banner", 130, 10),), 120)
    assert solve_strip_packing(turnable).length == 130


def test_checker_catches_broken_layouts():
    data = StripData("two", (Panel("a", 10, 10), Panel("b", 20, 10)), 30)
    good = {"a": Placement(0, 0, 10, 10, False), "b": Placement(10, 0, 20, 10, False)}
    assert strip_packing_errors(data, good, 10) == []
    overlap = {**good, "b": Placement(5, 0, 20, 10, False)}
    assert strip_packing_errors(data, overlap, 10) == ["a overlaps b"]
    outside = {**good, "b": Placement(15, 0, 20, 10, False)}
    assert strip_packing_errors(data, outside, 10) == ["b: outside the strip"]
    wrong_size = {**good, "b": Placement(10, 0, 10, 20, False)}
    assert any("size" in e for e in strip_packing_errors(data, wrong_size, 20))
    assert any("length" in e for e in strip_packing_errors(data, good, 15))
