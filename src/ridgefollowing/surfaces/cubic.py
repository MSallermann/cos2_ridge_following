from ridgefollowing.surfaces import quadratic
import numpy as np
import numpy.typing as npt


class CubicSurface(quadratic.QuadraticSurface):
    def __init__(self, hessian=np.diag([0.0, 0.0]), gradient=np.zeros(2)):
        self.x2y = -0.1  # x**2 y
        self.y2x = -0.3  # y**2 x
        self.x3 = 0.15  # x**3
        self.y3 = -0.5  # y**3
        self.factor = 1.0
        super().__init__(hessian=hessian, gradient=gradient, ndim=2)

    def setup_quapp_example(self, example_idx: int):
        """Sets up the parameters to reproduce the examples from the article
        'Gradient extremals and valley floor bifurcations on potential energy surfaces' by Quapp et al. (1988)
        """

        if example_idx == 2:
            # 1/2 * (xy + 2)*(y-x)
            self._hessian = np.zeros(shape=(2, 2))
            self._g0 = np.array([-1, 1])
            self.x2y = -0.5
            self.y2x = 0.5
            self.x3 = 0.0
            self.y3 = 0.0
        elif example_idx == 3:
            # 1/2 * (xy - 2)*(y-x)
            self._hessian = np.zeros(shape=(2, 2))
            self._g0 = np.array([1, -1])
            self.x2y = -0.5
            self.y2x = 0.5
            self.x3 = 0.0
            self.y3 = 0.0
        elif example_idx == 4:
            # 0.5 * (xy**2 - yx**2 + mu * x**2 + 2y - 3)
            self.mu = 1.0
            self._hessian = self.mu * np.array(
                [[1.0, 0.0], [0.0, 0.0]]
            )  # Notice: the quadratic energy contains a factor of 0.5 already
            self.y2x = 0.5
            self.x2y = -0.5
            self.x3 = 0.0
            self.y3 = 0.0
            self._g0 = np.array([0.0, 1.0])
        else:
            raise ValueError("Invalid example index")

    def scale_anharmonicity(self, factor):
        self.factor = factor

    def energy(self, x: npt.ArrayLike) -> float:
        return (
            super().energy(x)
            + self.factor * self.x2y * x[0] ** 2 * x[1]
            + self.factor * self.y2x * x[1] ** 2 * x[0]
            + self.factor * self.x3 * x[0] ** 3
            + self.factor * self.y3 * x[1] ** 3
        )

    def gradient(self, x: npt.ArrayLike) -> npt.NDArray:
        g = super().gradient(x)
        g += self.factor * self.x2y * np.array([2.0 * x[0] * x[1], x[0] ** 2])
        g += (
            self.factor
            * self.y2x
            * np.array(
                [
                    x[1] ** 2,
                    2.0 * x[1] * x[0],
                ]
            )
        )
        g[0] += 3.0 * self.factor * self.x3 * x[0] ** 2
        g[1] += 3.0 * self.factor * self.y3 * x[1] ** 2
        return g

    def hessian(self, x: npt.ArrayLike) -> npt.NDArray:
        h = quadratic.QuadraticSurface.hessian(self, x)
        h += (
            self.factor
            * self.x2y
            * np.array([[2.0 * x[1], 2.0 * x[0]], [2.0 * x[0], 0.0]])
        )
        h += (
            self.factor
            * self.y2x
            * np.array([[0.0, 2.0 * x[1]], [2.0 * x[1], 2.0 * x[0]]])
        )
        h[0, 0] += 6.0 * self.factor * self.x3 * x[0]
        h[1, 1] += 6.0 * self.factor * self.y3 * x[1]
        return h
