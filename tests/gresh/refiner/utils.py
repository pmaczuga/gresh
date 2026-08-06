import numpy as np
import numpy.typing as npt

from src.gresh import Gresh, NodeType


def assert_counts(g: Gresh, v_count: int, i_count: int, h_count: int, e_count: int):
    assert g.vertex_count() == v_count
    assert g.interior_count() == i_count
    assert g.hanging_count() == h_count
    assert len(list(g.edges())) == e_count


def assert_vertex_is_type(g: Gresh, coords: npt.NDArray[np.float64], type: NodeType):
    found_vertex = False
    for v in g.nodes_except_type(NodeType.INTERIOR):
        if np.allclose(g.xyz(v), coords):
            assert g.get_type(v) == type
            found_vertex = True
    assert found_vertex


def has_vertex_with_coords(g: Gresh, coords: npt.NDArray[np.float64]) -> bool:
    for v in g.nodes_except_type(NodeType.INTERIOR):
        if np.allclose(g.xyz(v), coords):
            return True
    return False
