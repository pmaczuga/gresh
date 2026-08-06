import numpy as np
import numpy.typing as npt

from src.gresh import Gresh
from src.types import Vector3

# Interior is always last added node
# So it will have id = `g.node_count() - 1`


def generate_graph(
    permutation: npt.NDArray[np.int32],
    edges: npt.NDArray[np.int32],
    interior: list[int],
    coords: npt.NDArray[np.float64],
    hanging: npt.NDArray[np.int32],
    boundaries: list[bool],
    refine=False,
):
    g = Gresh()
    v_map = np.zeros(len(permutation), dtype=np.int32)

    for v in permutation:
        v_map[v] = g.add_vertex(coords[:, v])

    if len(hanging) != 0:
        for i, h in enumerate(hanging[0, :]):
            v1 = v_map[hanging[1, i]]
            v2 = v_map[hanging[2, i]]
            g.set_hanging(v_map[h], v1, v2)

    for (v1, v2), boundary in zip(edges.T, boundaries):
        g.add_edge(v_map[v1], v_map[v2], boundary=boundary)

    vs = v_map[interior]
    g.add_pure_interior(*vs, refine=refine)

    return g


def shifted_arrays(a: npt.NDArray) -> list[npt.NDArray]:
    return [np.roll(a, n) for n in range(len(a))]


"Return graph on which production P1 should run."


def p1_graph_1(refine=True):
    "Return graph on which production P1 should run."
    g = Gresh()

    g.add_vertex(Vector3(0.0, 1.0, 0.0))
    g.add_vertex(Vector3(0.0, -1.0, 0.0))
    g.add_hanging(Vector3(0.0, 0.0, 0.0), 1, 2)
    g.add_vertex(Vector3(1.0, 0.0, 1.0))
    g.add_vertex(Vector3(0.5, 1.0, -1.0))

    g.add_pure_interior(2, 3, 4, refine=refine)

    g.add_edge(2, 3)
    g.add_edge(3, 4, boundary=True)
    g.add_edge(4, 2)

    return g


"Return graph on which production P1 should run."


def p1_graph_2(refine=True):
    g = Gresh()

    g.add_vertex(Vector3(0.0, 0.0, 0.0))
    g.add_vertex(Vector3(1.0, 0.0, 1.0))
    g.add_vertex(Vector3(0.5, 1.0, -1.0))

    g.add_pure_interior(0, 1, 2, refine=refine)

    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 0)

    return g


def graphs_by_production(refine=True) -> dict[str, list[Gresh]]:
    return {
        "p1": [p1_graph_1(refine), p1_graph_2(refine)],
        "p2": [],
        "p3": [],
        "p4": [],
        "p5": [],
        "p6": [],
    }


def get_graphs_for_production(p: str, refine=True) -> list[Gresh]:
    return graphs_by_production(refine)[p]


def get_graphs_except_production(p: str) -> list[Gresh]:
    graphs = graphs_by_production()
    return [g for other_p in graphs.keys() if other_p != p for g in graphs[other_p]]
