import numpy as np

from src.rectangle_mesh import generate_conec, generate_coords


def test_get_coords():
    coords = generate_coords(0, 3, 4, 6, 3, 2)
    expected = np.array(
        [
            [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0],
            [4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0],
        ]
    )
    assert np.allclose(coords, expected)


def test_get_conec():
    conec = generate_conec(1, 1)
    conec_list = conec.T.tolist()
    assert (
        ([0, 1, 2] in conec_list)
        or ([1, 2, 0] in conec_list)
        or ([2, 0, 1] in conec_list)
    )
    assert (
        ([1, 3, 2] in conec_list)
        or ([3, 2, 1] in conec_list)
        or ([2, 1, 3] in conec_list)
    )
