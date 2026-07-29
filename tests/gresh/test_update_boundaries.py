from src.gresh import Gresh
from src.types import Vector3


def get_graph():
    r"""
    ```
    5-----6-----7
    |\   /|\   /|
    | \ / | \ / |
    |  3  |  4  |
    | / \ | / \ |
    |/   \|/   \|
    0-----1-----2
    ```
    """
    g = Gresh()

    g.add_vertex(Vector3(0, 0, 0))  # 0
    g.add_vertex(Vector3(2, 0, 0))  # 1
    g.add_vertex(Vector3(4, 0, 0))  # 2
    g.add_vertex(Vector3(1, 1, 0))  # 3
    g.add_vertex(Vector3(3, 1, 0))  # 4
    g.add_vertex(Vector3(0, 2, 0))  # 5
    g.add_vertex(Vector3(2, 2, 0))  # 6
    g.add_vertex(Vector3(4, 2, 0))  # 7

    g.add_interior(0, 1, 3)  # 8
    g.add_interior(0, 3, 5)  # 9
    g.add_interior(1, 6, 3)  # 10
    g.add_interior(3, 6, 5)  # 11
    g.add_interior(2, 4, 7)  # 12
    g.add_interior(1, 2, 4)  # 13
    g.add_interior(1, 4, 6)  # 14
    g.add_interior(4, 7, 6)  # 15

    return g


def test_update_boundaries():
    g = get_graph()
    g.update_boundaries()
    assert g.is_on_boundary(0, 1)
    assert not g.is_on_boundary(1, 3)
    assert not g.is_on_boundary(3, 0)
    assert not g.is_on_boundary(3, 5)
    assert g.is_on_boundary(5, 0)
    assert not g.is_on_boundary(6, 3)
    assert g.is_on_boundary(5, 6)
    assert not g.is_on_boundary(1, 6)
    assert g.is_on_boundary(1, 2)
    assert not g.is_on_boundary(2, 4)
    assert not g.is_on_boundary(4, 1)
    assert not g.is_on_boundary(6, 4)
    assert g.is_on_boundary(6, 7)
    assert not g.is_on_boundary(7, 4)
    assert g.is_on_boundary(2, 7)
