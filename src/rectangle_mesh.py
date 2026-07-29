import numpy as np
import numpy.typing as npt


def rectangle_triangular_mesh(
    x_min: float, x_max: float, y_min: float, y_max: float, n_elem_x: int, n_elem_y: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]:
    """
    Return a tuple `(coords, conec)` representing the triangular mesh of a rectangle.

    - `coords` is a matrix, where each column is a vector of point coordinates.
    - `conec` is a connectivity in form of a matrix, where each column
      is a vector of vertex ID's in counter-clockwise order.

    Vertex ID's start from 0.
    """
    coords = generate_coords(x_min, x_max, y_min, y_max, n_elem_x, n_elem_y)
    conec = generate_conec(n_elem_x, n_elem_y)
    return coords, conec


def generate_coords(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    n_elem_x: int,
    n_elem_y: int,
) -> npt.NDArray[np.float64]:
    x = np.linspace(x_min, x_max, (n_elem_x) + 1)
    y = np.linspace(y_min, y_max, (n_elem_y) + 1)
    X, Y = np.meshgrid(x, y)
    coords = np.stack([X.reshape(-1), Y.reshape(-1)], 1).T
    return coords


def generate_conec(n_elem_x: int, n_elem_y: int) -> npt.NDArray[np.int32]:
    row_size = n_elem_x + 1
    h = np.array([0, 1, row_size], dtype=np.int32)
    conec = np.empty(tuple([3, 2 * n_elem_y * n_elem_x]), dtype=np.int32)
    for i in range(0, n_elem_y):
        for j in range(0, n_elem_x):
            n_base = i * row_size + j
            n_base2 = n_base + 1 + row_size
            conec[:, i * 2 * n_elem_x + 2 * j] = n_base + h
            conec[:, i * 2 * n_elem_x + 2 * j + 1] = n_base2 - h
    return conec
