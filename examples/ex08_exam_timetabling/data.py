"""Instance data for the exam timetabling example.

A school must place its final exams into time slots. A student cannot sit
two exams at the same time. So two exams that share at least one student
need different slots. Draw each exam as a node and join two nodes when
they share a student: this is the *conflict graph*. A timetable is then a
*coloring* of the graph, where each color is a time slot.

This module also builds classic graphs with a known chromatic number. The
tests use them to check the models.
"""

import itertools
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Graph:
    """An undirected graph. Each edge is a pair of node names."""

    name: str
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]

    def neighbors(self) -> dict[str, set[str]]:
        """Return the adjacency sets."""
        adj: dict[str, set[str]] = {n: set() for n in self.nodes}
        for a, b in self.edges:
            adj[a].add(b)
            adj[b].add(a)
        return adj


@dataclass(frozen=True)
class ExamData:
    """A full timetabling instance."""

    exams: tuple[str, ...]
    enrollments: tuple[tuple[str, ...], ...]  # One tuple of exams per student.
    days: int = 5  # Days in the exam week.
    slots_per_day: int = 2  # Morning and afternoon.

    @property
    def num_slots(self) -> int:
        """Return the number of slots in the exam week."""
        return self.days * self.slots_per_day

    def shared_students(self) -> dict[tuple[str, str], int]:
        """Return the number of shared students for every conflicting pair."""
        order = {e: k for k, e in enumerate(self.exams)}
        count: dict[tuple[str, str], int] = {}
        for exams in self.enrollments:
            for a, b in itertools.combinations(sorted(exams, key=order.get), 2):
                count[a, b] = count.get((a, b), 0) + 1
        return count

    def conflict_graph(self) -> Graph:
        """Return the graph whose edges join exams that share a student."""
        return Graph("school exams", self.exams, tuple(self.shared_students()))


# ------------------------------------------------------------- the school ---

# Each department lists its courses. The first two are open electives:
# students from other departments may take them too.
DEPARTMENTS = {
    "Sciences": (
        "Astronomy", "Ecology", "Physics", "Chemistry", "Biology", "Genetics", "Geology"
    ),
    "Mathematics": (
        "Statistics", "Logic", "Calculus", "Linear Algebra", "Number Theory",
        "Geometry", "Probability",
    ),
    "Languages": (
        "Spanish", "Creative Writing", "French", "German", "Latin", "Linguistics",
        "Literature",
    ),
    "Humanities": (
        "History", "Philosophy", "Geography", "Politics", "Sociology",
        "Anthropology", "Religion",
    ),
    "Arts": (
        "Music", "Photography", "Art History", "Painting", "Drama", "Design",
        "Film Studies",
    ),
    "Business": (
        "Economics", "Marketing", "Accounting", "Finance", "Management", "Law",
        "Entrepreneurship",
    ),
}  # fmt: skip
OPEN_ELECTIVES = 2


def school(num_students: int = 300, seed: int = 1) -> ExamData:
    """Return the default instance: 42 exams and 300 students.

    Each student belongs to one department and takes four of its seven
    courses. One student in five also takes an open elective from another
    department. So courses in the same department conflict a lot, and the
    electives link the departments.
    """
    rng = random.Random(seed)
    exams = tuple(c for courses in DEPARTMENTS.values() for c in courses)
    names = list(DEPARTMENTS)
    enrollments = []
    for _ in range(num_students):
        home = rng.choice(names)
        chosen = set(rng.sample(DEPARTMENTS[home], 4))
        if rng.random() < 0.2:
            other = rng.choice([d for d in names if d != home])
            chosen.add(rng.choice(DEPARTMENTS[other][:OPEN_ELECTIVES]))
        enrollments.append(tuple(sorted(chosen, key=exams.index)))
    return ExamData(exams, tuple(enrollments))


# --------------------------------------------------------- classic graphs ---


def complete_graph(n: int) -> Graph:
    """Return K_n. Every pair is joined, so it needs n colors."""
    nodes = tuple(str(k) for k in range(n))
    return Graph(f"K{n}", nodes, tuple(itertools.combinations(nodes, 2)))


def cycle_graph(n: int) -> Graph:
    """Return the cycle C_n. It needs 3 colors if n is odd, else 2."""
    nodes = tuple(str(k) for k in range(n))
    edges = tuple((nodes[k], nodes[(k + 1) % n]) for k in range(n))
    return Graph(f"C{n}", nodes, edges)


def petersen_graph() -> Graph:
    """Return the Petersen graph: 10 nodes, 15 edges, 3 colors."""
    outer = [(f"o{k}", f"o{(k + 1) % 5}") for k in range(5)]
    spokes = [(f"o{k}", f"i{k}") for k in range(5)]
    inner = [(f"i{k}", f"i{(k + 2) % 5}") for k in range(5)]
    nodes = tuple(f"o{k}" for k in range(5)) + tuple(f"i{k}" for k in range(5))
    return Graph("Petersen", nodes, tuple(outer + spokes + inner))


def mycielski_graph(k: int) -> Graph:
    """Return the Mycielski graph myciel<k> (DIMACS naming).

    Start from one edge (K2) and apply the Mycielski step k - 1 times.
    myciel3 has 11 nodes, myciel4 has 23, and myciel5 has 47. The graphs
    have no triangles, so the biggest clique has only 2 nodes. Yet myciel<k>
    needs k + 1 colors. This makes them hard for solvers: the clique bound
    does not help.
    """
    nodes = ["0", "1"]
    edges = [("0", "1")]
    for _ in range(k - 1):
        # For every node v add a twin u(v) joined to v's neighbors. Then add
        # one hub w joined to every twin.
        n = len(nodes)
        twin = {v: str(n + i) for i, v in enumerate(nodes)}
        hub = str(2 * n)
        new_edges = list(edges)
        for a, b in edges:
            new_edges += [(twin[a], b), (a, twin[b])]
        new_edges += [(twin[v], hub) for v in nodes]
        nodes = nodes + list(twin.values()) + [hub]
        edges = new_edges
    return Graph(f"myciel{k}", tuple(nodes), tuple(edges))


def queen_graph(n: int) -> Graph:
    """Return queen<n>_<n>: squares join if a queen can move between them.

    The queen5_5 graph needs 5 colors: each row is a clique of 5.
    """
    nodes = tuple(f"{r}{c}" for r in range(n) for c in range(n))
    edges = []
    for (r1, c1), (r2, c2) in itertools.combinations(
        [(r, c) for r in range(n) for c in range(n)], 2
    ):
        if r1 == r2 or c1 == c2 or abs(r1 - r2) == abs(c1 - c2):
            edges.append((f"{r1}{c1}", f"{r2}{c2}"))
    return Graph(f"queen{n}_{n}", nodes, tuple(edges))


def random_graph(n: int, p: float, seed: int = 0) -> Graph:
    """Return a random graph G(n, p) for tests."""
    rng = random.Random(seed)
    nodes = tuple(str(k) for k in range(n))
    edges = tuple(e for e in itertools.combinations(nodes, 2) if rng.random() < p)
    return Graph(f"G({n}, {p})", nodes, edges)
