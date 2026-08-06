import numpy as np
import numpy.typing as npt

type Vector3 = npt.NDArray[np.float64]


def Vector3(x: float, y: float, z: float) -> Vector3:
    return np.array([x, y, z], dtype=np.float64)
