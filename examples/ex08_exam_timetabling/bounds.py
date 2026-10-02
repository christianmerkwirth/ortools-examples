"""Fast bounds on the number of colors. Pure Python, no OR-Tools.

* `max_clique` gives a LOWER bound. Every node in a clique touches every
  other one, so a clique of size q needs q different colors.
* `dsatur` gives an UPPER bound. It is a greedy heuristic that colors a
  graph quickly, but not always with the fewest colors.

The model uses both: the upper bound limits the slot domain, and the
clique fixes some colors to break symmetry. The checker uses them too.
"""

from .data import Graph


def dsatur(graph: Graph) -> dict[str, int]:
    """Color a graph with the DSATUR heuristic (Brélaz, 1979).

    Repeat until all nodes have a color: pick the uncolored node that sees
    the most distinct colors among its neighbors (its *saturation*). Break
    ties by the most uncolored neighbors. Give it the smallest free color.
    """
    adj = graph.neighbors()
    color: dict[str, int] = {}
    while len(color) < len(graph.nodes):
        node = max(
            (n for n in graph.nodes if n not in color),
            key=lambda n: (
                len({color[m] for m in adj[n] if m in color}),
                sum(1 for m in adj[n] if m not in color),
            ),
        )
        used = {color[m] for m in adj[node] if m in color}
        color[node] = next(c for c in range(len(graph.nodes)) if c not in used)
    return color


def max_clique(graph: Graph) -> list[str]:
    """Return a maximum clique, using Bron-Kerbosch with pivoting.

    The search is exponential in the worst case, but fast for the graphs
    in this example (up to about 50 nodes).
    """
    adj = graph.neighbors()
    best: list[str] = []

    def expand(clique: list[str], candidates: set[str], excluded: set[str]) -> None:
        nonlocal best
        if not candidates and not excluded:
            if len(clique) > len(best):
                best = list(clique)
            return
        if len(clique) + len(candidates) <= len(best):
            return  # Cannot beat the best clique found so far.
        pivot = max(candidates | excluded, key=lambda n: len(adj[n] & candidates))
        for node in sorted(candidates - adj[pivot]):
            expand(clique + [node], candidates & adj[node], excluded & adj[node])
            candidates = candidates - {node}
            excluded = excluded | {node}

    expand([], set(graph.nodes), set())
    return best


def maximal_cliques(graph: Graph, min_size: int = 1) -> list[list[str]]:
    """Return every maximal clique with at least `min_size` nodes."""
    adj = graph.neighbors()
    found: list[list[str]] = []

    def expand(clique: list[str], candidates: set[str], excluded: set[str]) -> None:
        if not candidates and not excluded:
            if len(clique) >= min_size:
                found.append(clique)
            return
        pivot = max(candidates | excluded, key=lambda n: len(adj[n] & candidates))
        for node in sorted(candidates - adj[pivot]):
            expand(clique + [node], candidates & adj[node], excluded & adj[node])
            candidates = candidates - {node}
            excluded = excluded | {node}

    expand([], set(graph.nodes), set())
    return found
