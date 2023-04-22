from ridgefollowing.surfaces import lepsho, gaussians
import numpy as np
import numpy.typing as npt


class LepsHOGaussSurface(lepsho.LepsHOSurface):
    def __init__(self):
        super().__init__()

        magnitudes = np.array([1.5])
        centers = np.array([[2.02083, -0.272881]])
        matrices = np.array(
            [
                [
                    [-1 / 0.1**2, 0],
                    [0, -1 / 0.35**2],
                ]
            ]
        )
        widths = np.array([1])

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
