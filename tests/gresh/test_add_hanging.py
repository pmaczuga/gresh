import numpy as np
import pytest

from src.gresh import AddVertexStrategy, Gresh, NodeType


def test_graph_properties():
    g = Gresh(AddVertexStrategy.USE_XYZ)
    v1 = g.add_vertex(np.array([0, 0, 0]))
    v2 = g.add_vertex(np.array([2, 0, 0]))
    g.add_hanging(np.array([1, 0, 0]), v1, v2)
    assert g.node_count() == 3
    assert g.vertex_count() == 2
    assert g.interior_count() == 0
    assert g.hanging_count() == 1


def test_has_hanging_nodes():
    g = Gresh(AddVertexStrategy.USE_XYZ)
    v1 = g.add_vertex(np.array([0, 0, 0]))
    v2 = g.add_vertex(np.array([2, 0, 0]))
    assert not g.has_hanging_nodes()
    g.add_hanging(np.array([1, 0, 0]), v1, v2)
    assert g.has_hanging_nodes()


def test_unset_hanging():
    g = Gresh(AddVertexStrategy.USE_XYZ)
    v1 = g.add_vertex(np.array([0, 0, 0]))
    v2 = g.add_vertex(np.array([2, 0, 0]))
    h = g.add_hanging(np.array([1, 0, 0]), v1, v2)
    g.unset_hanging(h)
    assert not g.has_hanging_nodes()


def test_unset_non_hanging():
    g = Gresh(AddVertexStrategy.USE_XYZ)
    v1 = g.add_vertex(np.array([0, 0, 0]))
    v2 = g.add_vertex(np.array([2, 0, 0]))
    g.add_hanging(np.array([1, 0, 0]), v1, v2)
    g.unset_hanging(v1)
    assert not g.is_hanging(v1)
    assert g.has_hanging_nodes()


def test_coords():
    g = Gresh(AddVertexStrategy.USE_XYZ)
    v1 = g.add_vertex(np.array([0, 2, 6]))
    v2 = g.add_vertex(np.array([2, 2, 6]))
    idh = g.add_hanging(np.array([1, 2, 6]), v1, v2)
    assert (g.xyz(idh) == np.array([1, 2, 6])).all()
    assert (g.uv(idh) == np.array([1, 2])).all()
    assert (g.uve(idh) == np.array([1, 2, 6])).all()
    assert (g.get_elevation(idh) == 6.0).all()


def test_type():
    g = Gresh(AddVertexStrategy.USE_XYZ)
    v1 = g.add_vertex(np.array([0, 2, 6]))
    v2 = g.add_vertex(np.array([2, 2, 6]))
    idh = g.add_hanging(np.array([1, 2, 6]), v1, v2)
    assert not g.is_vertex(idh)
    assert not g.is_interior(idh)
    assert g.is_hanging(idh)
    assert g.get_type(idh) == NodeType.HANGING


def test_set_hanging():
    g = Gresh(AddVertexStrategy.USE_XYZ)
    v1 = g.add_vertex(np.array([0, 2, 6]))
    v2 = g.add_vertex(np.array([2, 2, 6]))
    idh = g.add_vertex(np.array([1, 2, 6]))
    g.set_hanging(idh, v1, v2)
    assert g.is_hanging(idh)


def test_get_hanging_node_between():
    g = Gresh(AddVertexStrategy.USE_XYZ)
    v1 = g.add_vertex(np.array([0, 0, 0]))
    v2 = g.add_vertex(np.array([2, 0, 0]))
    v3 = g.add_vertex(np.array([1, 1, 0]))
    idh = g.add_hanging(np.array([1, 0, 0]), v1, v2)
    g.add_pure_interior(v1, v2, v3)
    g.add_edge(v1, idh)
    g.add_edge(idh, v2)
    g.add_edge(v2, v3)
    g.add_edge(v3, v1)
    assert g.get_hanging_node_between(v1, v2) == idh


def test_get_hanging_node_not_exists():
    g = Gresh(AddVertexStrategy.USE_XYZ)
    v1 = g.add_vertex(np.array([0, 0, 0]))
    v2 = g.add_vertex(np.array([2, 0, 0]))
    v3 = g.add_vertex(np.array([1, 1, 0]))
    g.add_interior(v1, v2, v3)
    assert g.get_hanging_node_between(v1, v2) is None


def test_get_hanging_node_not_exists_no_common_edge():
    g = Gresh(AddVertexStrategy.USE_XYZ)
    v1 = g.add_vertex(np.array([0, 0, 0]))
    v2 = g.add_vertex(np.array([1, 0, 0]))
    v3 = g.add_vertex(np.array([2, 0, 0]))
    v4 = g.add_vertex(np.array([1, 1, 0]))
    g.add_interior(v1, v2, v4)
    g.add_interior(v2, v3, v4)
    assert g.get_hanging_node_between(v1, v3) is None


def test_get_hanging_node_with_two_of_them():
    g = Gresh(AddVertexStrategy.USE_XYZ)
    v1 = g.add_vertex(np.array([0, 0, 0]))
    v2 = g.add_vertex(np.array([2, 0, 0]))
    v3 = g.add_vertex(np.array([0, 1, 0]))
    v4 = g.add_vertex(np.array([2, 1, 0]))
    v5 = g.add_vertex(np.array([1, 2, 0]))
    h6 = g.add_hanging(np.array([1, 0, 0]), v1, v2)
    h7 = g.add_hanging(np.array([1, 1, 0]), v3, v4)
    g.add_pure_interior(v1, v2, h7)
    g.add_pure_interior(v3, v4, v5)
    g.add_edge(v1, h6)
    g.add_edge(h6, v2)
    g.add_edge(v2, h7)
    g.add_edge(h7, v1)
    g.add_edge(v3, h7)
    g.add_edge(h7, v4)
    g.add_edge(v4, v5)
    g.add_edge(v5, v3)
    assert g.get_hanging_node_between(v1, v2) == h6


def test_remove_hanging():
    g = Gresh(AddVertexStrategy.USE_XYZ)
    v1 = g.add_vertex(np.array([0, 0, 0]))
    v2 = g.add_vertex(np.array([2, 0, 0]))
    h = g.add_hanging(np.array([1, 0, 0]), v1, v2)
    assert g.hanging_count() == 1
    g.remove_node(h)
    assert g.hanging_count() == 0
    with pytest.raises(IndexError):
        g.get_type(h)
