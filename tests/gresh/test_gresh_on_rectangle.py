import numpy as np

from src.gresh import Gresh
from src.types import Vector3


def get_graph() -> Gresh:
    return Gresh.on_rectangle(0, 6, 0, 4, 2, 2)


def test_gresh_properties():
    g = get_graph()
    assert g.node_count() == 17
    assert g.vertex_count() == 9
    assert g.interior_count() == 8
    assert g.hanging_count() == 0


def test_coordinates():
    g = get_graph()
    assert np.allclose(g.uve(0), Vector3(0, 0, 0))
    assert np.allclose(g.uve(1), Vector3(3, 0, 0))
    assert np.allclose(g.uve(2), Vector3(6, 0, 0))
    assert np.allclose(g.uve(3), Vector3(0, 2, 0))
    assert np.allclose(g.uve(4), Vector3(3, 2, 0))
    assert np.allclose(g.uve(5), Vector3(6, 2, 0))
    assert np.allclose(g.uve(6), Vector3(0, 4, 0))
    assert np.allclose(g.uve(7), Vector3(3, 4, 0))
    assert np.allclose(g.uve(8), Vector3(6, 4, 0))


def test_connectivity():
    g = get_graph()
    assert set(g.interior_connectivity(9)) == set([0, 1, 3])
    assert set(g.interior_connectivity(10)) == set([1, 3, 4])
    assert set(g.interior_connectivity(11)) == set([1, 2, 4])
    assert set(g.interior_connectivity(12)) == set([4, 2, 5])
    assert set(g.interior_connectivity(13)) == set([3, 4, 6])
    assert set(g.interior_connectivity(14)) == set([6, 4, 7])
    assert set(g.interior_connectivity(15)) == set([4, 5, 7])
    assert set(g.interior_connectivity(16)) == set([7, 5, 8])


def test_edges():
    g = get_graph()
    assert g.has_edge(0, 1)
    assert g.has_edge(1, 3)
    assert g.has_edge(3, 0)
    assert g.has_edge(1, 4)
    assert g.has_edge(4, 3)
    assert g.has_edge(1, 2)
    assert g.has_edge(2, 4)
    assert g.has_edge(2, 5)
    assert g.has_edge(5, 4)
    assert g.has_edge(4, 6)
    assert g.has_edge(6, 3)
    assert g.has_edge(4, 7)
    assert g.has_edge(7, 6)
    assert g.has_edge(5, 7)
    assert g.has_edge(5, 8)
    assert g.has_edge(8, 7)


def test_boundary():
    g = get_graph()
    assert g.is_on_boundary(0, 1)
    assert not g.is_on_boundary(1, 3)
    assert g.is_on_boundary(3, 0)
    assert not g.is_on_boundary(1, 4)
    assert not g.is_on_boundary(4, 3)
    assert g.is_on_boundary(1, 2)
    assert not g.is_on_boundary(2, 4)
    assert g.is_on_boundary(2, 5)
    assert not g.is_on_boundary(5, 4)
    assert not g.is_on_boundary(4, 6)
    assert g.is_on_boundary(6, 3)
    assert not g.is_on_boundary(4, 7)
    assert g.is_on_boundary(7, 6)
    assert not g.is_on_boundary(5, 7)
    assert g.is_on_boundary(5, 8)
    assert g.is_on_boundary(8, 7)
