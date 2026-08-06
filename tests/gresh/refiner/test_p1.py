import numpy as np
import numpy.typing as npt
import pytest

from src.gresh import Gresh, NodeType
from src.refiner.p1 import P1
from src.types import Vector3
from tests.gresh.refiner.utils import (
    assert_counts,
    assert_vertex_is_type,
    has_vertex_with_coords,
)

from .production_graphs import (
    generate_graph,
    get_graphs_except_production,
    get_graphs_for_production,
    shifted_arrays,
)


def test_should_run():
    for g in get_graphs_for_production("p1"):
        assert P1().transform(g, g.node_count() - 1)


def test_should_not_run():
    for g in get_graphs_except_production("p1"):
        assert not P1().transform(g, g.node_count() - 1)


def test_should_not_run_no_refine():
    for g in get_graphs_for_production("p1", refine=False):
        assert not P1().transform(g, g.node_count() - 1)


def test_should_not_run_not_interior():
    for g in get_graphs_for_production("p1"):
        assert not P1().transform(g, 0)


def get_right_triangle(perm: npt.NDArray, boundaries: list[bool]) -> Gresh:
    r"""
    ```
    v
    |\
    | \ longest
    |  \
    v---v
    ```
    """
    edges = np.array([
        [0, 1, 2],
        [1, 2, 0]
    ])  # fmt: skip
    coords = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
    ])  # fmt: skip
    interior = [0, 1, 2]
    hanging = np.empty(0, dtype=np.int32)
    return generate_graph(perm, edges, interior, coords, hanging, boundaries, True)


@pytest.mark.parametrize("perm", shifted_arrays(np.array([0, 1, 2])))
def test_right_triangle_no_boundaries(perm):
    boundaries = [False, False, False]
    g = get_right_triangle(perm, boundaries)
    assert P1().transform(g, g.node_count() - 1)
    assert_counts(g, 3, 2, 1, 5)
    assert_vertex_is_type(g, Vector3(0.5, 0.5, 0.0), NodeType.HANGING)


@pytest.mark.parametrize("perm", shifted_arrays(np.array([0, 1, 2])))
def test_right_triangle_boundary_not_longest_edge(perm):
    boundaries = [True, False, False]
    g = get_right_triangle(perm, boundaries)
    assert P1().transform(g, g.node_count() - 1)
    assert_counts(g, 3, 2, 1, 5)
    assert_vertex_is_type(g, Vector3(0.5, 0.5, 0.0), NodeType.HANGING)


@pytest.mark.parametrize("perm", shifted_arrays(np.array([0, 1, 2])))
def test_right_triangle_boundary_on_longest_edge(perm):
    boundaries = [False, True, False]
    g = get_right_triangle(perm, boundaries)
    assert P1().transform(g, g.node_count() - 1)
    assert_counts(g, 4, 2, 0, 5)
    assert_vertex_is_type(g, Vector3(0.5, 0.5, 0.0), NodeType.VERTEX)


def get_isosceles_triangle(perm: npt.NDArray, boundaries: list[bool]) -> Gresh:
    r"""
    Isosceles triangle with two longest edges.
    ```
              v
             / \
    longest /   \ longest
           /     \
          v-------v
    ```
    """
    edges = np.array([
        [0, 1, 2],
        [1, 2, 0]
    ])  # fmt: skip
    coords = np.array([
        [0.0, 2.0, 1.0],
        [0.0, 0.0, 4.0],
        [0.0, 0.0, 0.0],
    ])  # fmt: skip
    interior = [0, 1, 2]
    hanging = np.empty(0, dtype=np.int32)
    return generate_graph(perm, edges, interior, coords, hanging, boundaries, True)


@pytest.mark.parametrize("perm", shifted_arrays(np.array([0, 1, 2])))
def test_isosceles_triangle_boundary_not_longest_edge(perm):
    boundaries = [True, False, False]
    g = get_isosceles_triangle(perm, boundaries)
    assert P1().transform(g, g.node_count() - 1)
    assert_counts(g, 3, 2, 1, 5)
    assert not has_vertex_with_coords(g, Vector3(1.0, 0.0, 0.0))
    # boundary is on lowest edge (short) - we don't want a new vertex there


@pytest.mark.parametrize("perm", shifted_arrays(np.array([0, 1, 2])))
def test_isosceles_triangle_boundary_longest_edge(perm):
    boundaries = [False, True, False]
    g = get_isosceles_triangle(perm, boundaries)
    assert P1().transform(g, g.node_count() - 1)
    assert_counts(g, 4, 2, 0, 5)
    assert_vertex_is_type(g, Vector3(1.5, 2.0, 0.0), NodeType.VERTEX)
