import numpy as np

from src.gresh import AddVertexStrategy, Gresh
from src.types import Vector3


def test_convert_xyz():
    g = Gresh(AddVertexStrategy.USE_XYZ)
    coords = Vector3(1, 2, 3)
    g.add_vertex(Vector3(1, 2, 3))
    assert np.allclose(coords, g.uve(0))


def test_convert_uve():
    g = Gresh(AddVertexStrategy.USE_UVE)
    coords = Vector3(1, 2, 3)
    g.add_vertex(Vector3(1, 2, 3))
    assert np.allclose(coords, g.xyz(0))


def test_distance_xyz():
    g = Gresh(AddVertexStrategy.USE_XYZ)
    g.add_vertex(Vector3(0, 0, 0))
    g.add_vertex(Vector3(3, 4, 0))
    assert g.distance(0, 1) == 5.0


def test_distance_uve():
    g = Gresh(AddVertexStrategy.USE_UVE)
    g.add_vertex(Vector3(0, 0, 0))
    g.add_vertex(Vector3(3, 4, 0))
    assert g.distance(0, 1) == 5.0


def test_new_vertex_coords_xyz():
    g = Gresh(AddVertexStrategy.USE_XYZ)
    g.add_vertex(Vector3(0, 0, 0))
    g.add_vertex(Vector3(2, 4, 0))
    assert np.allclose(g.new_vertex_coords(0, 1), Vector3(1, 2, 0))


def test_new_vertex_coords_uve():
    g = Gresh(AddVertexStrategy.USE_UVE)
    g.add_vertex(Vector3(0, 0, 0))
    g.add_vertex(Vector3(2, 4, 0))
    assert np.allclose(g.new_vertex_coords(0, 1), Vector3(1, 2, 0))
