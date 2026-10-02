"""Supply chain flows with the OR-Tools graph solvers.

Both parts use the same trick: add a virtual SOURCE node that feeds every
plant, and a virtual SINK node that every store feeds. Then

* the arc SOURCE -> plant carries what the plant makes (capacity = plant
  capacity, cost = production cost), and
* the arc store -> SINK carries what the store receives (capacity = store
  demand).

Part 1, max flow: how many pallets per week can the network deliver at
most? The solver also returns a minimum cut: the set of arcs that limits
the flow. These arcs are the bottlenecks worth upgrading.

Part 2, min-cost flow: deliver what the stores need at the lowest total
production and transport cost.
"""

from dataclasses import dataclass

from ortools.graph.python import max_flow, min_cost_flow

from .data import NetworkData

SOURCE, SINK = "SOURCE", "SINK"


class NoFeasibleFlowError(RuntimeError):
    """Raised when the network cannot meet all demand."""


@dataclass(frozen=True)
class Arc:
    """An arc of the extended network (lanes plus virtual arcs)."""

    tail: str
    head: str
    capacity: int
    unit_cost: int

    @property
    def kind(self) -> str:
        """Return "plant", "store", or "lane"."""
        if self.tail == SOURCE:
            return "plant"
        if self.head == SINK:
            return "store"
        return "lane"

    def __str__(self) -> str:
        if self.kind == "plant":
            return f"{self.head} capacity"
        if self.kind == "store":
            return f"{self.tail} demand"
        return f"{self.tail} -> {self.head}"


@dataclass(frozen=True)
class Flow:
    """A flow on the network: pallets per arc, plus summaries."""

    value: int  # Pallets delivered to stores per week.
    cost: int  # Production plus transport cost, dollars per week.
    arc_flow: dict[tuple[str, str], int]  # (tail, head) -> pallets.

    def lane_flow(self, data: NetworkData) -> dict[tuple[str, str], int]:
        """Return the flow on the real lanes only."""
        return {x.key: self.arc_flow[x.key] for x in data.lanes}


@dataclass(frozen=True)
class MaxFlowResult:
    """The maximum flow and a minimum cut that proves it."""

    flow: Flow
    source_side: frozenset[str]  # Nodes on the SOURCE side of the cut.
    cut: list[Arc]  # Arcs from the source side to the sink side.


def extended_arcs(data: NetworkData) -> list[Arc]:
    """Return all arcs: virtual plant and store arcs plus the real lanes."""
    arcs = [Arc(SOURCE, p.name, p.capacity, p.unit_cost) for p in data.plants]
    arcs += [Arc(x.origin, x.destination, x.capacity, x.unit_cost) for x in data.lanes]
    arcs += [Arc(s.name, SINK, s.demand, 0) for s in data.stores]
    return arcs


def _node_index(data: NetworkData) -> dict[str, int]:
    """Give each node an integer id. The graph solvers use ids, not names."""
    names = [SOURCE, *(p.name for p in data.plants), *data.warehouses]
    names += [*(s.name for s in data.stores), SINK]
    return {name: k for k, name in enumerate(names)}


def solve_max_flow(data: NetworkData) -> MaxFlowResult:
    """Find the most pallets the network can deliver, and the bottleneck."""
    index = _node_index(data)
    name = {k: n for n, k in index.items()}
    arcs = extended_arcs(data)

    solver = max_flow.SimpleMaxFlow()
    for a in arcs:
        # Each call returns an arc id; ids count up from 0 in call order.
        solver.add_arc_with_capacity(index[a.tail], index[a.head], a.capacity)

    status = solver.solve(index[SOURCE], index[SINK])
    if status != solver.OPTIMAL:
        raise RuntimeError(f"Max flow failed with status {status}")

    arc_flow = {(a.tail, a.head): solver.flow(k) for k, a in enumerate(arcs)}
    # The nodes reachable from SOURCE in the residual graph form one side
    # of a minimum cut. Arcs that leave this set are full: the bottleneck.
    source_side = frozenset(name[k] for k in solver.get_source_side_min_cut())
    cut = [a for a in arcs if a.tail in source_side and a.head not in source_side]
    cost = sum(a.unit_cost * arc_flow[a.tail, a.head] for a in arcs)
    return MaxFlowResult(Flow(solver.optimal_flow(), cost, arc_flow), source_side, cut)


def solve_min_cost_flow(data: NetworkData, deliver_all: bool = True) -> Flow:
    """Find the cheapest way to deliver pallets to the stores.

    With `deliver_all=True`, every store must get its full demand. If the
    network cannot do that, raise NoFeasibleFlowError.

    With `deliver_all=False`, deliver as much as the network can (the max
    flow), and among all such plans pick the cheapest one.
    """
    index = _node_index(data)
    arcs = extended_arcs(data)

    solver = min_cost_flow.SimpleMinCostFlow()
    for a in arcs:
        solver.add_arc_with_capacity_and_unit_cost(
            index[a.tail], index[a.head], a.capacity, a.unit_cost
        )
    # Supplies must balance: SOURCE sends what SINK takes.
    solver.set_node_supply(index[SOURCE], data.total_demand)
    solver.set_node_supply(index[SINK], -data.total_demand)

    if deliver_all:
        status = solver.solve()
        if status == solver.INFEASIBLE:
            raise NoFeasibleFlowError("The network cannot meet all demand.")
    else:
        # Treat the supplies as upper limits: send the max flow, at min cost.
        status = solver.solve_max_flow_with_min_cost()
    if status != solver.OPTIMAL:
        raise RuntimeError(f"Min-cost flow failed with status {status}")

    arc_flow = {(a.tail, a.head): solver.flow(k) for k, a in enumerate(arcs)}
    delivered = sum(arc_flow[s.name, SINK] for s in data.stores)
    return Flow(delivered, solver.optimal_cost(), arc_flow)


@dataclass(frozen=True)
class Upgrade:
    """The effect of adding capacity to one lane."""

    lane: tuple[str, str]
    extra: int  # Pallets per week added to the lane.
    delivered: int  # Max flow after the upgrade.
    cost: int  # Cost of the cheapest max flow after the upgrade.


def upgrade_options(data: NetworkData, extra: int) -> list[Upgrade]:
    """Try adding `extra` capacity to each lane in the minimum cut.

    A minimum cut is not always unique. So a lane in the cut does not
    always pay to upgrade: another cut with the same capacity may still
    hold the flow back. Solving again is the safe way to know.
    """
    options = []
    for arc in solve_max_flow(data).cut:
        if arc.kind != "lane":
            continue
        bigger = data.with_lane_capacity(arc.tail, arc.head, arc.capacity + extra)
        flow = solve_min_cost_flow(bigger, deliver_all=False)
        options.append(Upgrade((arc.tail, arc.head), extra, flow.value, flow.cost))
    return options
