"""Plain-text tables for console reports."""

from collections.abc import Sequence


def format_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Return rows as an aligned text table.

    Numbers are right-aligned, everything else is left-aligned. Floats are
    shown with two decimals.
    """
    cells = [[_fmt(v) for v in row] for row in rows]
    widths = [
        max(len(h), *(len(r[i]) for r in cells)) if cells else len(h)
        for i, h in enumerate(headers)
    ]
    numeric = [
        bool(rows) and all(isinstance(r[i], int | float) for r in rows)
        for i in range(len(headers))
    ]

    def line(values: Sequence[str]) -> str:
        parts = [
            v.rjust(w) if num else v.ljust(w)
            for v, w, num in zip(values, widths, numeric, strict=True)
        ]
        return "  ".join(parts).rstrip()

    out = [line(headers), line(["-" * w for w in widths])]
    out += [line(r) for r in cells]
    return "\n".join(out)


def _fmt(value: object) -> str:
    if isinstance(value, float):
        if abs(value) < 0.005:
            value = 0.0  # Avoid "-0.00" from tiny solver round-off.
        return f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)
