from src.gresh import Gresh
from src.types import Vector3


def get_graph_hanging() -> Gresh:
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
    v0 = g.add_vertex(Vector3(0, 0, 0))  # 0
    v1 = g.add_vertex(Vector3(2, 0, 0))  # 1
    h2 = g.add_vertex(Vector3(1, 1, 0))  # 2
    v3 = g.add_vertex(Vector3(0, 2, 0))  # 3
    v4 = g.add_vertex(Vector3(2, 2, 0))  # 4
    g.set_hanging(h2, v1, v3)

    g.add_pure_interior(v0, v1, h2)  # 5
    g.add_pure_interior(v0, h2, v3)  # 6
    g.add_pure_interior(v1, v3, v4)  # 7

    g.add_edge(0, 1)
    g.add_edge(1, 4)
    g.add_edge(3, 3)
    g.add_edge(3, 0)
    g.add_edge(2, 0)
    g.add_edge(2, 1)
    g.add_edge(2, 3)

    return g


def get_graph_no_hanging():
    r"""
    ```
    3-----2
    |\    |
    | \   |
    |  \  |
    |   \ |
    |    \|
    0-----1
    ```
    """
    g = Gresh()
    g.add_vertex(Vector3(0, 0, 0))  # 0
    g.add_vertex(Vector3(2, 0, 0))  # 1
    g.add_vertex(Vector3(1, 1, 0))  # 2
    g.add_vertex(Vector3(0, 2, 0))  # 3

    g.add_interior(0, 1, 3)  # 4
    g.add_interior(1, 2, 3)  # 5

    return g


def test_to_string_no_hanging():
    g = get_graph_no_hanging()
    assert g.__str__() == "Gresh with (4 vertices), (2 interiors) and (5 edges)"


def test_to_string_hanging():
    g = get_graph_hanging()
    assert (
        g.__str__()
        == "Gresh with (4 vertices), (3 interiors) and (7 edges) that has !1 hanging nodes!"
    )
