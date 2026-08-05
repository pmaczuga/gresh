import numpy as np
import pytest

from src.gresh import Gresh, NodeType
from src.types import Vector3


def get_graph() -> Gresh:
    r"""
    ```
    3-----4
    |\    |
    | \   |
    | 2h  |
    | / \ |
    |/   \|
    0-----1
    ```
    """
    g = Gresh()
    g.add_vertex(Vector3(0, 0, 0))  # 0
    g.add_vertex(Vector3(2, 0, 0))  # 1
    g.add_vertex(Vector3(1, 1, 0))  # 2
    g.add_vertex(Vector3(0, 2, 0))  # 3
    g.add_vertex(Vector3(2, 2, 0))  # 4
    g.set_hanging(2, 1, 3)

    g.add_pure_interior(0, 1, 2)  # 5
    g.add_pure_interior(0, 2, 3)  # 6
    g.add_pure_interior(1, 3, 4)  # 7

    g.add_edge(0, 1)
    g.add_edge(1, 4)
    g.add_edge(3, 3)
    g.add_edge(3, 0)
    g.add_edge(2, 0)
    g.add_edge(2, 1)
    g.add_edge(2, 3)

    return g


def test_get_vertex_data():
    g = get_graph()
    data = g.get_vertex_data(0)
    assert data.type() == NodeType.VERTEX
    assert np.allclose(data.xyz(), Vector3(0, 0, 0))
    assert np.allclose(data.uve(), Vector3(0, 0, 0))


def test_get_hanging_data():
    g = get_graph()
    data = g.get_hanging_data(2)
    assert data.type() == NodeType.HANGING
    assert np.allclose(data.xyz(), Vector3(1, 1, 0))
    assert np.allclose(data.uve(), Vector3(1, 1, 0))
    assert data._v1 == 1 or data._v1 == 3
    assert data._v2 == 1 or data._v2 == 3


def test_get_interior_data():
    g = get_graph()
    data = g.get_interior_data(5)
    assert data.type() == NodeType.INTERIOR
    assert not data.refine()


def test_allow_getting_vertex_data_from_hanging():
    g = get_graph()
    data = g.get_vertex_data(2)
    assert data.type() == NodeType.HANGING


def test_exception_when_getting_vertex_data_from_interior():
    g = get_graph()
    expected_msg = "Trying to get VertexData from node of type INTERIOR"
    with pytest.raises(TypeError, match=expected_msg):
        g.get_vertex_data(5)


def test_exception_when_getting_hanging_data_from_interior():
    g = get_graph()
    expected_msg = "Trying to get HangingData from node of type INTERIOR"
    with pytest.raises(TypeError, match=expected_msg):
        g.get_hanging_data(5)


def test_exception_when_getting_hanging_data_from_vertex():
    g = get_graph()
    expected_msg = "Trying to get HangingData from node of type VERTEX"
    with pytest.raises(TypeError, match=expected_msg):
        g.get_hanging_data(0)


def test_exception_when_getting_interior_data_from_vertex():
    g = get_graph()
    expected_msg = "Trying to get InteriorData from node of type VERTEX"
    with pytest.raises(TypeError, match=expected_msg):
        g.get_interior_data(0)


def test_exception_when_getting_interior_data_from_hanging():
    g = get_graph()
    expected_msg = "Trying to get InteriorData from node of type HANGING"
    with pytest.raises(TypeError, match=expected_msg):
        g.get_interior_data(2)
