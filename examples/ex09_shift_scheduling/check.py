"""Independent checks for rosters. Uses no OR-Tools code.

* `hard_rule_errors` lists every broken hard rule.
* `penalty_breakdown` recomputes each soft-rule penalty from the roster.
* `brute_force_optimum` tries every roster of a tiny instance and returns
  the best score. It shares no code with the CP-SAT model.
"""

import itertools

from .data import RosterData

# A roster maps (nurse, day) to a shift code, or None for a day off.
Assignment = dict[tuple[str, int], str | None]


def hard_rule_errors(data: RosterData, roster: Assignment) -> list[str]:
    """Return a list of broken hard rules. An empty list means valid."""
    errors = []
    codes = {s.code for s in data.shifts}
    for n in data.nurses:
        errors += _personal_errors(
            data, n, [roster[n.name, d] for d in range(data.num_days)]
        )
        if any(roster[n.name, d] not in codes | {None} for d in range(data.num_days)):
            errors.append(f"{n.name}: unknown shift code")
    errors += _coverage_errors(data, roster)
    return errors


def _personal_errors(data: RosterData, nurse, pattern: list[str | None]) -> list[str]:
    """Check the rules that concern one nurse only."""
    errors = []
    name = nurse.name
    worked = [s is not None for s in pattern]
    for d in nurse.leave:
        if worked[d]:
            errors.append(f"{name}: works on leave day {d}")
    total = sum(worked)
    if not nurse.min_shifts <= total <= nurse.max_shifts:
        errors.append(
            f"{name}: {total} shifts, contract {nurse.min_shifts}-{nurse.max_shifts}"
        )
    forbidden = set(data.forbidden_sequences)
    for d in range(data.num_days - 1):
        if (pattern[d], pattern[d + 1]) in forbidden:
            errors.append(f"{name}: {pattern[d]} on day {d}, then {pattern[d + 1]}")
    run = 0
    for d, w in enumerate(worked):
        run = run + 1 if w else 0
        if run > data.max_consecutive_days:
            errors.append(f"{name}: {run} days in a row up to day {d}")
    return errors


def _coverage_errors(data: RosterData, roster: Assignment) -> list[str]:
    errors = []
    for d in range(data.num_days):
        for s in data.shifts:
            staff = [n for n in data.nurses if roster[n.name, d] == s.code]
            need = data.demand[d].get(s.code, 0)
            if len(staff) < need:
                errors.append(f"day {d} {s.code}: {len(staff)} nurses, needs {need}")
            if need > 0 and not any(n.senior for n in staff):
                errors.append(f"day {d} {s.code}: no senior nurse")
    return errors


def penalty_breakdown(data: RosterData, roster: Assignment) -> dict[str, int]:
    """Recompute the weighted penalty points of each soft-rule category."""
    w = data.weights
    requests = 0
    for r in data.requests:
        shift = roster[r.nurse, r.day]
        if (r.shift is None and shift is not None) or (
            r.shift is not None and shift == r.shift
        ):
            requests += r.weight

    isolated = 0
    for n in data.nurses:
        for d in range(1, data.num_days - 1):
            before, today, after = (roster[n.name, d + k] for k in (-1, 0, 1))
            if before is not None and today is None and after is not None:
                isolated += 1

    full_time = [n for n in data.nurses if n.full_time]
    nights = [
        sum(roster[n.name, d] == data.night_shift for d in range(data.num_days))
        for n in full_time
    ]
    weekends = [
        sum(roster[n.name, d] is not None for d in data.weekend_days) for n in full_time
    ]
    return {
        "requests": requests,
        "isolated days off": w.isolated_day_off * isolated,
        "night spread": w.night_spread * _spread(nights),
        "weekend spread": w.weekend_spread * _spread(weekends),
    }


def _spread(values: list[int]) -> int:
    return max(values) - min(values) if values else 0


def total_penalty(data: RosterData, roster: Assignment) -> int:
    """Return the total penalty points of a roster."""
    return sum(penalty_breakdown(data, roster).values())


def brute_force_optimum(data: RosterData) -> tuple[int, Assignment | None]:
    """Return the best total penalty and a roster that reaches it.

    First list every work pattern that each nurse may work on their own
    (leave, contract, rest, days in a row). Then try every combination of
    patterns and keep those that cover all shifts. Only for tiny instances:
    the number of combinations grows very fast.
    """
    choices = [None, *(s.code for s in data.shifts)]
    patterns = []
    for n in data.nurses:
        mine = [
            list(p)
            for p in itertools.product(choices, repeat=data.num_days)
            if not _personal_errors(data, n, list(p))
        ]
        patterns.append(mine)

    best, best_roster = None, None
    for combo in itertools.product(*patterns):
        roster = {
            (n.name, d): pattern[d]
            for n, pattern in zip(data.nurses, combo, strict=True)
            for d in range(data.num_days)
        }
        if _coverage_errors(data, roster):
            continue
        score = total_penalty(data, roster)
        if best is None or score < best:
            best, best_roster = score, roster
    return best, best_roster
