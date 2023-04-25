import numpy as np
import numpy.typing as npt


def lowest_mode(matrix: npt.ArrayLike) -> npt.NDArray:
    """Computes the lowest eigevalue and mode

    Args:
        matrix (npt.ArrayLike): the matrix for which to compute the lowest mode

    Returns:
        float, npt.NDArray: Tupel of lowest (eval, evec)
    """
    evals, evecs = np.linalg.eigh(matrix)
    return evals[0], evecs[:, 0]


def eigenpairs(matrix: npt.ArrayLike) -> npt.NDArray:
    """Returns eigenvalues and vectors, sorted by eigenvalue

    Args:
        matrix (npt.ArrayLike): the matrix for which to compute the lowest mode

    Returns:
        float, npt.NDArray: Tupel of lowest (eval, evec)
    """
    evals, evecs = np.linalg.eigh(matrix)
    return evals, evecs
