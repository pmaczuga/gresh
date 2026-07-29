import numpy as np

from src.gresh import AddVertexStrategy, Gresh, NodeType


def test_node_count():
    g = Gresh(AddVertexStrategy.USE_XYZ)
    v1 = g.add_vertex(np.array([0, 0, 0]))
    v2 = g.add_vertex(np.array([2, 0, 0]))
    v3 = g.add_vertex(np.array([1, 1, 0]))
    g.add_interior(v1, v2, v3)
    assert g.node_count() == 4
    assert g.vertex_count() == 3
    assert g.interior_count() == 1
    assert g.hanging_count() == 0


def test_type():
    g = Gresh(AddVertexStrategy.USE_XYZ)
    v1 = g.add_vertex(np.array([0, 0, 0]))
    v2 = g.add_vertex(np.array([2, 0, 0]))
    v3 = g.add_vertex(np.array([1, 1, 0]))
    i1 = g.add_interior(v1, v2, v3)
    assert not g.is_vertex(i1)
    assert g.is_interior(i1)
    assert not g.is_hanging(i1)
    assert g.get_type(i1) == NodeType.INTERIOR


def test_refine():
    g = Gresh(AddVertexStrategy.USE_XYZ)
    v1 = g.add_vertex(np.array([0, 0, 0]))
    v2 = g.add_vertex(np.array([1, 0, 0]))
    v3 = g.add_vertex(np.array([0, 1, 0]))
    v4 = g.add_vertex(np.array([-1, 0, 0]))
    i1 = g.add_interior(v1, v2, v3)
    i2 = g.add_interior(v1, v3, v4, refine=True)
    assert not g.should_refine(i1)
    assert g.should_refine(i2)
    g.set_refine(i1)
    assert g.should_refine(i1)
    g.unset_refine(i1)
    assert not g.should_refine(i1)


def test_new_edges():
    g = Gresh(AddVertexStrategy.USE_XYZ)
    v1 = g.add_vertex(np.array([0, 0, 0]))
    v2 = g.add_vertex(np.array([2, 0, 0]))
    v3 = g.add_vertex(np.array([1, 1, 0]))
    g.add_interior(v1, v2, v3)
    assert g.has_edge(v1, v2)
    assert g.has_edge(v2, v3)
    assert g.has_edge(v3, v1)


def test_interior_connectivity():
    g = Gresh(AddVertexStrategy.USE_XYZ)
    v1 = g.add_vertex(np.array([0, 0, 0]))
    v2 = g.add_vertex(np.array([2, 0, 0]))
    v3 = g.add_vertex(np.array([1, 1, 0]))
    i1 = g.add_interior(v1, v2, v3)
    assert set(g.interior_connectivity(i1)) == set([v1, v2, v3])
