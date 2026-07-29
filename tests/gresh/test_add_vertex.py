from typing import override

import numpy as np

from src.gresh import AddVertexStrategy, Gresh, NodeType


def test_add_xyz():
    g = Gresh(AddVertexStrategy.USE_XYZ)
    id = g.add_vertex(np.array([1, 2, 3]))
    assert (g.xyz(id) == np.array([1, 2, 3])).all()
    assert (g.uv(id) == np.array([1, 2])).all()
    assert (g.uve(id) == np.array([1, 2, 3])).all()
    assert (g.get_elevation(id) == 3.0).all()


def test_add_uve():
    g = Gresh(AddVertexStrategy.USE_UVE)
    id = g.add_vertex(np.array([1, 2, 3]))
    assert (g.xyz(id) == np.array([1, 2, 3])).all()
    assert (g.uv(id) == np.array([1, 2])).all()
    assert (g.uve(id) == np.array([1, 2, 3])).all()
    assert (g.get_elevation(id) == 3.0).all()


def test_type():
    g = Gresh()
    id = g.add_vertex(np.array([1, 2, 3]))
    assert g.is_vertex(id)
    assert not g.is_interior(id)
    assert not g.is_hanging(id)
    assert g.get_type(id) == NodeType.VERTEX


def test_elevation():
    g = Gresh()
    id = g.add_vertex(np.array([1, 2, 3]))
    assert g.get_elevation(id) == 3
    g.set_elevation(id, 10)
    assert g.get_elevation(id) == 10


def test_custom_converter_xyz():
    class CustomGresh(Gresh):
        @override
        def convert(self, coords):
            return np.array([4, 5, 6])

    g = CustomGresh(AddVertexStrategy.USE_XYZ)
    id = g.add_vertex(np.array([1, 2, 3]))
    assert (g.xyz(id) == np.array([1, 2, 3])).all()
    assert (g.uv(id) == np.array([4, 5])).all()
    assert (g.uve(id) == np.array([4, 5, 6])).all()
    assert (g.get_elevation(id) == 6).all()


def test_custom_converter_uve():
    class CustomGresh(Gresh):
        @override
        def convert(self, coords):
            return np.array([1, 2, 3])

    g = CustomGresh(AddVertexStrategy.USE_UVE)
    id = g.add_vertex(np.array([4, 5, 6]))
    assert (g.xyz(id) == np.array([1, 2, 3])).all()
    assert (g.uv(id) == np.array([4, 5])).all()
    assert (g.uve(id) == np.array([4, 5, 6])).all()
    assert (g.get_elevation(id) == 6).all()
