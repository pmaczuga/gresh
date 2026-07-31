import re

import py

from src.gresh import Gresh
from src.io import export_avsucd
from src.types import Vector3


def get_graph():
    r"""
    ```
    2-----3
    |\    |
    | \   |
    |  \  |
    |   \ |
    |    \|
    0-----1
    ```
    """

    g = Gresh()
    g.add_vertex(Vector3(0, 0, 0))
    g.add_vertex(Vector3(1, 0, 0))
    g.add_vertex(Vector3(0, 1, 0))
    g.add_vertex(Vector3(1, 1, 0))
    g.add_interior(0, 1, 2)
    g.add_interior(1, 3, 2)
    return g


def test_export_avsucd(tmpdir: py.path.local):
    g = get_graph()
    filename = str(tmpdir) + "mesh.inp"
    export_avsucd(g, filename)

    with open(filename, "r") as f:
        lines = f.readlines()
        string = "".join(lines)
        print(string)
        assert re.match(get_expected_result_regex(), string)


def get_expected_result_regex() -> str:
    return """4 2 4 1 0
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 0.0 1.0 0.0
4 1.0 1.0 0.0
1 0 tri \d \d \d
2 0 tri \d \d \d
2 1 3
vertex_id,nunits
uve,degree
1 1 0.0 0.0 0.0
2 2 1.0 0.0 0.0
3 3 0.0 1.0 0.0
4 4 1.0 1.0 0.0
1 1
interior_id,nunits
1 1
2 2"""
