from ridgefollowing.surfaces import gaussians
import numpy as np


class MullerBrownSurface(gaussians.GaussianSurface):
    def __init__(self):
        magnitudes = np.array([-200.0, -100.0, -170.0, 15.0], dtype=np.float64)
        centers = np.array(
            [[1.0, 0.0], [0.0, 0.5], [-0.5, 1.5], [-1.0, 1.0]], dtype=np.float64
        )
        widths = 1.0 / np.sqrt(2.0 * np.ones(4))
        matrices = np.array(
            [
                [[-1.0, 0.0], [0.0, -10.0]],
                [[-1.0, 0.0], [0.0, -10.0]],
                [[-6.5, 11.0 / 2.0], [11.0 / 2.0, -6.5]],
                [[0.7, 0.6 / 2.0], [0.6 / 2.0, 0.7]],
            ]
        )

        super().__init__(
            magnitudes=magnitudes,
            matrices=matrices,
            centers=centers,
            widths=widths,
            ndim=2,
        )

    def A(self):
        """Get A parameters"""
        return self.magnitudes

    def a(self):
        """Get a parameters"""
        return self.matrices[:, 0, 0]

    def b(self):
        """Get b parameters"""
        return self.matrices[:, 1, 1]

    def c(self):
        """Get c parameters"""
        return 2 * self.matrices[:, 0, 1]

    def xo1(self):
        """Get xo1 parameters"""
        return self.centers[:, 0]

    def xo2(self):
        """Get x02 parameters"""
        return self.centers[:, 1]
