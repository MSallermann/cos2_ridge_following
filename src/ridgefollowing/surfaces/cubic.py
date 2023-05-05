from ridgefollowing.surfaces import quadratic
import numpy as np
import numpy.typing as npt


class CubicSurface(quadratic.QuadraticSurface):
    def __init__(self, matrix=np.diag([1.0, 2.0])):
        self.f = 0.1
        super().__init__(matrix, 2)

    def energy(self, x: npt.ArrayLike) -> float:
        return super().energy(x) + self.f * x[0] ** 2 * x[1]
        # return 0.5 * np.transpose(x) @ self._hessian @ x

    def gradient(self, x: npt.ArrayLike) -> npt.NDArray:
        g = super().gradient(x)
        g += np.array([2 * self.f * x[0] * x[1], self.f * x[0] ** 2])
        return g

    def hessian(self, x: npt.ArrayLike) -> npt.NDArray:
        h = super().hessian(x)
        h += np.array(
            [[2.0 * self.f * x[1], 2.0 * self.f * x[0]], [2.0 * self.f * x[0], 0.0]]
        )
        return h
