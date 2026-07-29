import numpy as np

from src.gresh import AddVertexStrategy, Gresh


def test_has_edge():
    g = Gresh(AddVertexStrategy.USE_XYZ)
    v1 = g.add_vertex(np.array([0, 0, 0]))
    v2 = g.add_vertex(np.array([2, 0, 0]))
    v3 = g.add_vertex(np.array([1, 1, 0]))
    g.add_edge(v1, v2)
    assert g.has_edge(v1, v2)
    assert not g.has_edge(v2, v3)
    assert not g.has_edge(v3, v1)


def test_boundary():
    g = Gresh(AddVertexStrategy.USE_XYZ)
    v1 = g.add_vertex(np.array([0, 0, 0]))
    v2 = g.add_vertex(np.array([2, 0, 0]))
    v3 = g.add_vertex(np.array([1, 1, 0]))
    g.add_edge(v1, v2)
    g.add_edge(v2, v3, boundary=True)
    g.add_edge(v3, v1)
    assert not g.is_on_boundary(v1, v2)
    assert g.is_on_boundary(v2, v3)
    assert not g.is_on_boundary(v3, v1)
    g.set_boundary(v1, v2)
    assert g.is_on_boundary(v1, v2)
    g.unset_boundary(v1, v2)
    assert not g.is_on_boundary(v1, v2)


def test_edge_length():
    g = Gresh(AddVertexStrategy.USE_XYZ)
    v1 = g.add_vertex(np.array([0, 0, 0]))
    v2 = g.add_vertex(np.array([1, 0, 0]))
    g.add_edge(v1, v2)
    assert g.distance(v1, v2) == 1.0
