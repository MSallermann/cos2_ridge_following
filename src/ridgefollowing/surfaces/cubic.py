from ridgefollowing.surfaces import quadratic
import numpy as np
import numpy.typing as npt


class CubicSurface(quadratic.QuadraticSurface):
    def __init__(self, matrix=np.diag([1.0, 2.0])):
        self.f1 = 0.1
        self.f2 = 0.1
        self.c1 = -0.3
        self.c2 = 0.2
        super().__init__(matrix, 2)

    def energy(self, x: npt.ArrayLike) -> float:
        return (
            super().energy(x)
            + self.f1 * x[0] ** 2 * x[1]
            + self.f2 * x[1] ** 2 * x[0]
            + self.c1 * x[0] ** 3
            + self.c2 * x[1] ** 3
        )

    def gradient(self, x: npt.ArrayLike) -> npt.NDArray:
        g = super().gradient(x)
        g += self.f1 * np.array([2.0 * x[0] * x[1], x[0] ** 2])

        g += self.f2 * np.array(
            [
                x[1] ** 2,
                2.0 * x[1] * x[0],
            ]
        )

        g[0] += 3 * self.c1 * x[0] ** 2
        g[1] += 3 * self.c2 * x[1] ** 2
        return g

    def hessian(self, x: npt.ArrayLike) -> npt.NDArray:
        h = super().hessian(x)

        h += self.f1 * np.array([[2.0 * x[1], 2.0 * x[0]], [2.0 * x[0], 0.0]])

        h += self.f2 * np.array([[2.0 * x[1], 0.0], [2.0 * x[0], 2.0 * x[1]]])

        h[0, :] += 6.0 * self.c1 * x[0]
        h[1, :] += 6.0 * self.c2 * x[1]
        return h
