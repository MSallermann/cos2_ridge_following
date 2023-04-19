from ridgefollowing import energy_surface
import numpy as np
import numpy.typing as npt


class GaussianSurface(energy_surface.EnergySurface):
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

    def energy(self, x: npt.ArrayLike) -> npt.NDArray:
        E = 0

        for igauss in range(self.n_gaussians):
            d = x - self.centers[igauss]
            w = 1.0 / (2.0 * self.widths[igauss] ** 2)
            M = np.dot(d, np.matmul(self.matrices[igauss], d))

            E += self.magnitudes[igauss] * np.exp(w * M)

        return E

    def gradient(self, x: npt.ArrayLike) -> npt.NDArray:
        grad = np.zeros(self.ndim)

        for igauss in range(self.n_gaussians):
            d = x - self.centers[igauss]

            prefactor = self.magnitudes[igauss] * 1.0 / (2.0 * self.widths[igauss] ** 2)

            exponential = np.exp(
                1.0
                / (2.0 * self.widths[igauss] ** 2)
                * np.dot(d, np.matmul(self.matrices[igauss], d))
            )

            grad += prefactor * 2 * np.matmul(self.matrices[igauss], d) * exponential

        return grad
