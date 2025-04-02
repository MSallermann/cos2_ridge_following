from ridgefollowing.algorithms import modes
from scipy.stats import ortho_group
import numpy as np


def test_modes():
    ndim = 12
    basis = ortho_group.rvs(ndim)
    eigenvalues = np.linspace(-1.0, 10.0, ndim)
    test_matrix = np.transpose(basis) @ np.diag(eigenvalues) @ basis

    # test eigenpairs
    evals, evecs = modes.eigenpairs(test_matrix)
    evecs = basis @ evecs  # transform back into original coordiantes

    assert np.allclose(
        np.abs(evecs), np.diag(np.ones(ndim))
    )  # take abs value because direction of evec is undefined
    assert np.allclose(eigenvalues, evals)

    # test lowest mode
    eval, evec = modes.lowest_mode(test_matrix)
    assert np.isclose(eval, eigenvalues[0])
    evec_expected = np.zeros(ndim)
    evec_expected[0] = 1
    assert np.allclose(
        np.abs(basis @ evec), evec_expected
    )  # take abs value because direction of evec is undefined
