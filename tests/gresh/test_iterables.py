from src.gresh import AddVertexStrategy, Gresh, NodeType
from src.types import Vector3


def get_graph() -> Gresh:
    r"""
    ```
    5-----6-----7
    |\    |    /|
    | \   |   / |
    | 3h  |  4h |
    | / \ | / \ |
    |/   \|/   \|
    0-----1-----2
    ```
    """
    g = Gresh(AddVertexStrategy.USE_XYZ)
    v1 = g.add_vertex(Vector3(0, 0, 0))  # 0
    v2 = g.add_vertex(Vector3(2, 0, 0))  # 1
    v3 = g.add_vertex(Vector3(4, 0, 0))  # 2
    h4 = g.add_vertex(Vector3(1, 1, 0))  # 3
    h5 = g.add_vertex(Vector3(3, 1, 0))  # 4
    v6 = g.add_vertex(Vector3(0, 2, 0))  # 5
    v7 = g.add_vertex(Vector3(2, 2, 0))  # 6
    v8 = g.add_vertex(Vector3(4, 2, 0))  # 7
    g.set_hanging(h4, v2, v6)
    g.set_hanging(h5, v2, v8)

    g.add_pure_interior(v1, v2, h4)  # 8
    g.add_pure_interior(v1, h4, v6)  # 9
    g.add_pure_interior(v2, v7, v6)  # 10
    g.add_pure_interior(v3, h5, v8)  # 11
    g.add_pure_interior(v2, v3, h5)  # 12
    g.add_pure_interior(v2, v8, v7)  # 13

    g.add_edge(v2, v3)
    g.add_edge(v1, v2)
    g.add_edge(v3, v8)
    g.add_edge(v8, v7)
    g.add_edge(v7, v6)
    g.add_edge(v6, v1)
    g.add_edge(v1, h4)
    g.add_edge(v2, h4)
    g.add_edge(h4, v6)
    g.add_edge(v3, h5)
    g.add_edge(v2, h5)
    g.add_edge(h5, v8)

    return g


def test_vertices():
    g = get_graph()
    vs = [0, 1, 2, 5, 6, 7]
    assert set(g.vertices()) == set(vs)
    assert set(g.nodes_with_type(NodeType.VERTEX)) == set(vs)


def test_hanging_nodes():
    g = get_graph()
    vs = [3, 4]
    assert set(g.hanging_nodes()) == set(vs)
    assert set(g.nodes_with_type(NodeType.HANGING)) == set(vs)


def test_interiors():
    g = get_graph()
    vs = [8, 9, 10, 11, 12, 13]
    assert set(g.interiors()) == set(vs)
    assert set(g.nodes_with_type(NodeType.INTERIOR)) == set(vs)


def test_except_vertices():
    g = get_graph()
    vs = [3, 4, 8, 9, 10, 11, 12, 13]
    assert set(g.nodes_except_type(NodeType.VERTEX)) == set(vs)


def test_except_hanging():
    g = get_graph()
    vs = [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    assert set(g.nodes_except_type(NodeType.HANGING)) == set(vs)


def test_except_interior():
    g = get_graph()
    vs = [0, 1, 2, 3, 4, 5, 6, 7]
    assert set(g.nodes_except_type(NodeType.INTERIOR)) == set(vs)


def test_neighbors():
    g = get_graph()
    vs = [0, 3, 4, 2, 8, 10, 12, 13]
    assert set(g.neighbors(1)) == set(vs)


def test_vertex_neighbors():
    g = get_graph()
    vs = [0, 2]
    assert set(g.vertex_neighbors(1)) == set(vs)
    assert set(g.neighbors_with_type(1, NodeType.VERTEX)) == set(vs)


def test_interior_neighbors():
    g = get_graph()
    vs = [8, 10, 12, 13]
    assert set(g.interior_neighbors(1)) == set(vs)
    assert set(g.neighbors_with_type(1, NodeType.INTERIOR)) == set(vs)


def test_hanging_neighbors():
    g = get_graph()
    vs = [3, 4]
    assert set(g.hanging_neighbors(1)) == set(vs)
    assert set(g.neighbors_with_type(1, NodeType.HANGING)) == set(vs)


def test_edges():
    g = get_graph()
    es = set(
        [
            frozenset([0, 1]),
            frozenset([1, 2]),
            frozenset([2, 7]),
            frozenset([7, 6]),
            frozenset([6, 5]),
            frozenset([5, 0]),
            frozenset([0, 3]),
            frozenset([1, 3]),
            frozenset([3, 5]),
            frozenset([2, 4]),
            frozenset([1, 4]),
            frozenset([4, 7]),
        ]
    )
    assert frozenset(map(lambda x: frozenset(x), g.edges())) == es


def test_all_edges():
    g = Gresh()
    g.add_vertex(Vector3(0, 0, 0))
    g.add_vertex(Vector3(0, 2, 0))
    g.add_vertex(Vector3(1, 1, 0))
    g.add_interior(0, 1, 2)
    es = set(
        [
            frozenset([0, 1]),
            frozenset([1, 2]),
            frozenset([2, 0]),
            frozenset([0, 3]),
            frozenset([1, 3]),
            frozenset([2, 3]),
        ]
    )
    assert set(map(lambda x: frozenset(x), g.all_edges())) == es
