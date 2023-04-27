from ridgefollowing.surfaces import lepsho, gaussians
import numpy as np
import numpy.typing as npt


class LepsHOGaussSurface(lepsho.LepsHOSurface):
    def __init__(self):
        super().__init__()

        magnitudes = np.array([1.5, 7.0])
        centers = np.array([[2.02083, -0.172881], [0.8, 2.0]])
        matrices = np.array(
            [
                [
                    [-1 / 0.1**2, 0],
                    [0, -1 / 0.35**2],
                ],
                [
                    [-1 / 0.447213**2, 0],
                    [0, -1 / 1.195229**2],
                ],
            ]
        )
        widths = np.array([1.0, 1.0])

        # Multiple inheritance sucks in python, so we use composition instead
        self.gaussian_surface = gaussians.GaussianSurface(
            magnitudes=magnitudes,
            matrices=matrices,
            centers=centers,
            widths=widths,
            ndim=2,
        )

    def energy(self, x: npt.ArrayLike) -> float:
        energy = super().energy(x)
        energy += self.gaussian_surface.energy(x)
        return energy

    def gradient(self, x: npt.ArrayLike) -> npt.NDArray:
        grad = super().gradient(x)
        grad += self.gaussian_surface.gradient(x)
        return grad

    def hessian(self, x: npt.ArrayLike) -> npt.NDArray:
        hessian = super().hessian(x)
        hessian += self.gaussian_surface.hessian(x)
        return hessian
