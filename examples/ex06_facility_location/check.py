"""Independent checks for facility location plans. Uses no OR-Tools code.

* `feasibility_errors` checks that every customer has exactly one open
  site and that no site ships more than its capacity.
* `plan_cost` recomputes the cost from the data.
* `brute_force_optimum` tries every assignment of customers to sites.
  It only works for tiny instances, but it needs no solver at all.
"""

import numpy as np

from .data import LocationData

TOL = 1e-6


def feasibility_errors(
    data: LocationData, open_sites: list[str], assignment: dict[str, str]
) -> list[str]:
    """Return a list of broken rules. An empty list means feasible."""
    errors = []
    sites = {s.name: s for s in data.sites}
    opened = set(open_sites)
    for c in data.customers:
        site = assignment.get(c.name)
        if site is None:
            errors.append(f"{c.name} has no site")
        elif site not in opened:
            errors.append(f"{c.name} is served by closed {site}")
    unknown = set(assignment) - {c.name for c in data.customers}
    if unknown:
        errors.append(f"unknown customers: {sorted(unknown)}")
    for name in opened:
        load = sum(c.demand for c in data.customers if assignment.get(c.name) == name)
        if load > sites[name].capacity:
            errors.append(f"{name}: load {load} > capacity {sites[name].capacity}")
    return errors


def plan_cost(
    data: LocationData, open_sites: list[str], assignment: dict[str, str]
) -> tuple[float, float]:
    """Return (fixed cost, transport cost) per week."""
    sites = {s.name: s for s in data.sites}
    fixed = sum(sites[n].fixed_cost for n in open_sites)
    transport = sum(
        data.transport_cost(c, sites[assignment[c.name]]) for c in data.customers
    )
    return fixed, transport


def brute_force_optimum(data: LocationData) -> tuple[float, dict[str, str]]:
    """Return the optimal cost and assignment by full enumeration.

    An optimal plan never opens a site that serves nobody (that only adds
    cost). So it is enough to try every assignment of customers to sites
    and open exactly the sites in use. With m sites and n customers there
    are m**n assignments; numpy checks them all at once.
    """
    m, n = len(data.sites), len(data.customers)
    if m**n > 1_000_000:
        raise ValueError("instance too large for brute force")
    demand = np.array([c.demand for c in data.customers])
    capacity = np.array([s.capacity for s in data.sites])
    fixed = np.array([s.fixed_cost for s in data.sites])
    transport = np.array(
        [[data.transport_cost(c, s) for s in data.sites] for c in data.customers]
    )

    # Row k of `choice` is the k-th assignment in base m: choice[k, i] is
    # the site index for customer i.
    codes = np.arange(m**n)
    choice = (codes[:, None] // m ** np.arange(n)[None, :]) % m

    serves = choice[:, :, None] == np.arange(m)[None, None, :]  # (k, i, j)
    load = np.einsum("kij,i->kj", serves, demand)
    used = serves.any(axis=1)
    cost = used @ fixed + transport[np.arange(n)[None, :], choice].sum(axis=1)
    cost[(load > capacity).any(axis=1)] = np.inf

    best = int(np.argmin(cost))
    assignment = {
        c.name: data.sites[choice[best, i]].name for i, c in enumerate(data.customers)
    }
    return float(cost[best]), assignment
