import ridgefollowing
from ridgefollowing.surfaces import (
    muller_brown,
    gaussians,
    peaks,
    leps,
    lepsho,
    lepshogauss,
    quadratic,
    cubic,
)
import numdifftools as nd
import numpy as np
import pytest

epsilon = 1e6


def test_against_fd():
    """Test the implementation of gradient and hessian functions against finite differences"""

    # Two dimensional test points
    test_points_2d = np.array(
        [[1.0, 0], [-1.0, 0.5], [2, 2], [-1, 4], [0, 0], [-1, 2], [-1.0, 4.742]]
    )

    # 1. The Muller brown surface
    esurf_mb = muller_brown.MullerBrownSurface()

    # 2. Peaks surface
    esurf_peaks = peaks.PeaksSurface()

    # 3. Leps surfaces
    esurf_leps = leps.LepsSurface()
    esurf_lepsho = lepsho.LepsHOSurface()
    esurf_lepshogauss = lepshogauss.LepsHOGaussSurface()

    # 4. Quadratic surface
    esurf_quadratic = quadratic.QuadraticSurface(hessian=np.diag([2.0, 4.0]))

    # 5. Cubic surface
    esurf_cubic = cubic.CubicSurface()

    # 6. A more general gaussian surface
    ndim = 12
    ngauss = 3
    magnitudes = np.linspace(-5, 5, ngauss)
    widths = np.linspace(1, 3, ngauss)

    centers = np.zeros(shape=(ngauss, ndim))
    for dim in range(ndim):
        centers[:, dim] = np.sin(np.linspace(-dim, 2 * dim, ngauss))

    matrices = np.zeros(shape=(ngauss, ndim, ndim))

    for i in range(ngauss):
        matrices[i] += np.diag(np.cos(np.linspace(1, 5, ndim)))
        for k in range(ngauss):
            matrices[i] += np.diag(np.linspace(0, 5, ndim - k), k=k)
            matrices[i] += np.diag(np.linspace(0, 5, ndim - k), k=-k)

    esurf_gauss = gaussians.GaussianSurface(
        magnitudes=magnitudes,
        matrices=matrices,
        centers=centers,
        widths=widths,
        ndim=ndim,
    )
    test_points_gauss = np.zeros(shape=(ngauss, ndim))
    test_points_gauss[:, 0] = -1.2 * centers[:, 1]
    test_points_gauss[:, 1] = 0.6 * centers[:, 0]

    # Perform the tests
    for esurf, test_points in [
        [esurf_mb, test_points_2d],
        [esurf_peaks, test_points_2d],
        [esurf_leps, test_points_2d],
        [esurf_lepsho, test_points_2d],
        [esurf_lepshogauss, test_points_2d],
        [esurf_quadratic, test_points_2d],
        [esurf_cubic, test_points_2d],
        [esurf_gauss, test_points_gauss],
    ]:
        for x in test_points:
            print(esurf)

            print(f"x = {x}")

            energy = esurf.energy(x)  # Can't really test, should at leat not throw
            print(f"energy = {energy}")

            # Compare gradient to FD
            grad_fd = esurf.fd_gradient(x)
            grad = esurf.gradient(x)
            print(f"grad_fd = {grad_fd}")
            print(f"grad = {grad}")

            assert np.allclose(grad, grad_fd)

            # Compare hessian to FD
            hessian_fd = esurf.fd_hessian(x)
            hessian = esurf.hessian(x)

            print("hessian_fd\n", hessian_fd)
            print("hessian\n", hessian)
            assert np.allclose(hessian_fd, hessian)

            # Test the directional gradient
            dir = test_points[0] / np.linalg.norm(test_points[0])
            directional_gradient = esurf.directional_gradient(x, dir)

            fun = lambda alpha: esurf.energy(x + alpha * dir)

            directional_gradient_fd = nd.Derivative(fun)(0)
            print("dir", dir)
            print("directional_gradient", directional_gradient)
            print("directional_gradient_fd", directional_gradient_fd)

            assert np.allclose(directional_gradient, directional_gradient_fd)

            # Curvature, just take the first test point as direction
            dir = test_points[0] / np.linalg.norm(test_points[0])

            curvature_fd = esurf.fd_curvature(x, dir)
            curvature = esurf.curvature(x, dir)
            print("dir", dir)
            print("curvature_fd", curvature_fd)
            print("curvature", curvature)
            assert np.allclose(curvature_fd, curvature)

            print("hessian * dir", hessian @ dir)
            assert np.allclose(
                curvature, hessian @ dir
            )  # The curvature should be equivalent to the action of the hessian
