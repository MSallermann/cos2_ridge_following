from ridgefollowing import energy_surface
import numpy as np
import numpy.typing as npt
from numba import jit
from numba import float64, int32


class GaussianSurface(energy_surface.EnergySurface):
    """Implements energy terms of the form
    E_gauss(r) = magnitude * exp( 0.5 / width**2  * (r-center)^T M (r-center) )
    """

    def __init__(
        self,
        magnitudes: npt.ArrayLike,
        matrices: npt.ArrayLike,
        centers: npt.ArrayLike,
        widths: npt.ArrayLike,
        ndim=2,
    ):
        super().__init__(ndim)

        self.matrices = matrices
        self.magnitudes = magnitudes
        self.centers = centers
        self.widths = widths
        self.n_gaussians = len(magnitudes)

        self.check_arrays()

    def check_arrays(self):
        assert self.magnitudes.shape == (self.n_gaussians,)
        assert self.matrices.shape == (self.n_gaussians, self.ndim, self.ndim)
        assert self.centers.shape == (self.n_gaussians, self.ndim)
        assert self.widths.shape == (self.n_gaussians,)

    @staticmethod
    @jit(nopython=True, cache=True)
    def energy_helper(x, n_gaussians, magnitudes, centers, widths, matrices):
        E = 0.0
        for igauss in range(n_gaussians):
            d = x - centers[igauss]
            w = 1.0 / (2.0 * widths[igauss] ** 2)
            M = np.dot(d, matrices[igauss] @ d)

            E += magnitudes[igauss] * np.exp(w * M)
        return E

    def energy(self, x: npt.ArrayLike) -> npt.NDArray:
        return GaussianSurface.energy_helper(
            x,
            n_gaussians=self.n_gaussians,
            magnitudes=self.magnitudes,
            centers=self.centers,
            widths=self.widths,
            matrices=self.matrices,
        )

    @staticmethod
    @jit(nopython=True, cache=True)
    def gradient_helper(x, ndim, n_gaussians, magnitudes, centers, widths, matrices):
        grad = np.zeros(ndim)

        for igauss in range(n_gaussians):
            d = x - centers[igauss]
            w = 1.0 / (2.0 * widths[igauss] ** 2)
            prefactor = magnitudes[igauss] * w

            exponential = np.exp(w * np.dot(d, matrices[igauss] @ d))

            grad += prefactor * 2 * matrices[igauss] @ d * exponential

        return grad

    def gradient(self, x: npt.ArrayLike) -> npt.NDArray:
        return GaussianSurface.gradient_helper(
            x,
            ndim=self.ndim,
            n_gaussians=self.n_gaussians,
            magnitudes=self.magnitudes,
            centers=self.centers,
            widths=self.widths,
            matrices=self.matrices,
        )

    @staticmethod
    @jit(nopython=True, cache=True)
    def hessian_helper(x, ndim, n_gaussians, magnitudes, centers, widths, matrices):
        hessian = np.zeros((ndim, ndim))

        for igauss in range(n_gaussians):
            d = x - centers[igauss]
            w = 1.0 / (2.0 * widths[igauss] ** 2)

            Md = matrices[igauss] @ d

            exponential = np.exp(w * np.dot(d, matrices[igauss] @ d))

            for i in range(ndim):
                for j in range(ndim):
                    hessian[i, j] += (
                        2.0
                        * magnitudes[igauss]
                        * w
                        * (
                            2.0 * w * exponential * Md[i] * Md[j]
                            + exponential * matrices[igauss, i, j]
                        )
                    )

        return hessian

    def hessian(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return GaussianSurface.hessian_helper(
            x,
            ndim=self.ndim,
            n_gaussians=self.n_gaussians,
            magnitudes=self.magnitudes,
            centers=self.centers,
            widths=self.widths,
            matrices=self.matrices,
        )

    def curvature(self, x: npt.ArrayLike, dir: npt.ArrayLike) -> npt.NDArray:
        dir_n = dir / np.linalg.norm(dir)

        curvature = np.zeros(self.ndim)
        for igauss in range(self.n_gaussians):
            d = x - self.centers[igauss]
            w = 1.0 / (2.0 * self.widths[igauss] ** 2)

            Md = np.matmul(self.matrices[igauss], d)
            Mdir = np.matmul(self.matrices[igauss], dir_n)
            exponential = np.exp(w * np.dot(d, np.matmul(self.matrices[igauss], d)))
            curvature += (
                2
                * self.magnitudes[igauss]
                * w
                * (2 * w * exponential * np.dot(Mdir, d) * Md + exponential * Mdir)
            )

        return curvature
