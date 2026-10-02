"""Instance data for the project scheduling example: building a house.

A builder plans a family house from the first survey to the final
inspection. Each activity takes a fixed number of working days, may only
start after some other activities end, and needs crews (and sometimes the
crane) for its whole duration. The builder has only a few crews of each
trade. Durations and crew sizes are illustrative.
"""

import random
from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Resource:
    """A renewable resource: a crew type or a machine, available every day."""

    name: str
    capacity: int  # Units available on each day.


@dataclass(frozen=True)
class Activity:
    """One activity of the project."""

    name: str
    duration: int  # Working days.
    predecessors: tuple[str, ...] = ()  # Must end before this one starts.
    demand: dict[str, int] = field(default_factory=dict)  # Units per day.

    def needs(self, resource: str) -> int:
        """Return the units of a resource this activity uses per day."""
        return self.demand.get(resource, 0)


@dataclass(frozen=True)
class ProjectData:
    """A full problem instance."""

    name: str
    activities: tuple[Activity, ...]
    resources: tuple[Resource, ...]

    def activity(self, name: str) -> Activity:
        """Return the activity with this name."""
        return next(a for a in self.activities if a.name == name)

    def resource(self, name: str) -> Resource:
        """Return the resource with this name."""
        return next(r for r in self.resources if r.name == name)

    def with_capacity(self, resource: str, capacity: int) -> "ProjectData":
        """Return a copy with a changed capacity for one resource."""
        resources = tuple(
            replace(r, capacity=capacity) if r.name == resource else r
            for r in self.resources
        )
        return replace(self, resources=resources)


def _act(name: str, duration: int, predecessors=(), **demand: int) -> Activity:
    return Activity(name, duration, tuple(predecessors), demand)


def house() -> ProjectData:
    """Return the default instance: 28 activities, 6 resources."""
    resources = (
        Resource("laborers", 4),
        Resource("masons", 3),
        Resource("carpenters", 3),
        Resource("electricians", 2),
        Resource("plumbers", 2),
        Resource("crane", 1),
    )
    activities = (
        _act("survey and layout", 1, (), laborers=1),
        _act("excavation", 3, ["survey and layout"], laborers=3),
        _act("foundation formwork", 2, ["excavation"], carpenters=2, laborers=2),
        _act("underground plumbing", 2, ["excavation"], plumbers=2),
        _act(
            "foundation pour",
            2,
            ["foundation formwork"],
            masons=2,
            laborers=2,
            crane=1,
        ),
        _act(
            "basement walls",
            4,
            ["foundation pour", "underground plumbing"],
            masons=3,
            laborers=1,
        ),
        _act("ground floor slab", 2, ["basement walls"], masons=2, laborers=2, crane=1),
        _act("ground floor walls", 5, ["ground floor slab"], masons=3, laborers=2),
        _act("floor joists", 3, ["ground floor walls"], carpenters=3, crane=1),
        _act("upper floor walls", 5, ["floor joists"], masons=3, laborers=2),
        _act("roof trusses", 3, ["upper floor walls"], carpenters=3, crane=1),
        _act("roofing", 4, ["roof trusses"], carpenters=2, laborers=2),
        _act("windows and doors", 3, ["upper floor walls"], carpenters=2),
        _act(
            "exterior facade",
            6,
            ["roofing", "windows and doors"],
            masons=2,
            laborers=2,
        ),
        _act("rough electrical", 4, ["roofing", "windows and doors"], electricians=2),
        _act("rough plumbing", 4, ["roofing", "windows and doors"], plumbers=2),
        _act("heating system", 3, ["rough plumbing"], plumbers=1, electricians=1),
        _act(
            "insulation",
            3,
            ["rough electrical", "rough plumbing"],
            laborers=2,
            carpenters=1,
        ),
        _act("drywall", 5, ["insulation"], carpenters=3),
        _act("interior doors and trim", 3, ["drywall"], carpenters=2),
        _act("bathroom tiling", 4, ["drywall"], masons=2),
        _act("kitchen", 3, ["drywall"], carpenters=2, plumbers=1, electricians=1),
        _act("finish electrical", 2, ["drywall"], electricians=2),
        _act("finish plumbing", 2, ["bathroom tiling", "kitchen"], plumbers=2),
        _act(
            "painting",
            4,
            ["interior doors and trim", "finish electrical"],
            laborers=3,
        ),
        _act("flooring", 3, ["painting", "bathroom tiling"], carpenters=2),
        _act("landscaping", 4, ["exterior facade"], laborers=3),
        _act(
            "final inspection",
            1,
            ["flooring", "finish plumbing", "landscaping", "heating system"],
        ),
    )
    return ProjectData("family house", activities, resources)


def housing_row(n_houses: int) -> ProjectData:
    """Return n copies of the house project that share one set of crews.

    The houses do not depend on each other, so the critical path stays the
    same as for one house. But every house competes for the same crews and
    the same crane, so the resources now decide the project length.
    """
    one = house()
    activities = tuple(
        Activity(
            f"{a.name} [{h}]",
            a.duration,
            tuple(f"{p} [{h}]" for p in a.predecessors),
            a.demand,
        )
        for h in range(1, n_houses + 1)
        for a in one.activities
    )
    return ProjectData(f"row of {n_houses} houses", activities, one.resources)


def random_project(
    n_activities: int, n_resources: int = 2, seed: int = 0, max_duration: int = 4
) -> ProjectData:
    """Return a small random project for tests.

    Each activity gets up to two random predecessors among the earlier
    activities, so the precedence graph has no cycles.
    """
    rng = random.Random(seed)
    resources = tuple(Resource(f"r{k}", rng.randint(2, 4)) for k in range(n_resources))
    activities: list[Activity] = []
    for k in range(n_activities):
        earlier = [a.name for a in activities]
        preds = tuple(rng.sample(earlier, min(len(earlier), rng.randint(0, 2))))
        demand = {r.name: rng.randint(0, r.capacity) for r in resources}
        activities.append(
            Activity(f"a{k}", rng.randint(1, max_duration), preds, demand)
        )
    return ProjectData(f"random {n_activities}", tuple(activities), resources)
