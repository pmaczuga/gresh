from src.gresh import Gresh
from src.types import Vector3


def get_graph():
    r"""
    ```
    4-----6
    |\   /|
    | \ / |
    |  2  |
    | / \ |
    |/   \|
    0-----1
    ```
    """
    g = Gresh()
    g.add_vertex(Vector3(0, 0, 0))  # 0
    g.add_vertex(Vector3(2, 0, 0))  # 1
    g.add_vertex(Vector3(1, 1, 0))  # 2
    g.add_interior(0, 1, 2)  # 3
    g.add_vertex(Vector3(0, 2, 0))  # 4
    g.add_interior(0, 2, 4)  # 5
    g.add_vertex(Vector3(2, 2, 0))  # 6
    g.add_interior(1, 6, 2)  # 7
    g.add_interior(2, 6, 4)  # 8
    return g


def test_vertex_map():
    g = get_graph()
    assert g.vertex_map() == dict([(0, 0), (1, 1), (2, 2), (4, 3), (6, 4)])
