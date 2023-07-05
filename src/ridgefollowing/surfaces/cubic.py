from ridgefollowing.surfaces import quadratic
import numpy as np
import numpy.typing as npt
from numba.experimental import jitclass
from numba import float64, int32


@jitclass(
    [
        ("x2y", float64),
        ("y2x", float64),
        ("x3", float64),
        ("y3", float64),
        ("factor", float64),
        ("ndim", int32),
    ]
)
class CubicSurfaceHelper:
    def __init__(self, x2y, y2x, x3, y3, factor, ndim):
        self.x2y = -0.1  # x**2 y
        self.y2x = -0.3  # y**2 x
        self.x3 = 0.15  # x**3
        self.y3 = -0.5  # y**3
        self.factor = 1.0
        self.ndim = ndim

    def energy(self, x: npt.ArrayLike) -> float:
        return (
            self.factor * self.x2y * x[0] ** 2 * x[1]
            + self.factor * self.y2x * x[1] ** 2 * x[0]
            + self.factor * self.x3 * x[0] ** 3
            + self.factor * self.y3 * x[1] ** 3
        )

    def gradient(self, x: npt.ArrayLike) -> npt.NDArray:
        g = np.zeros(self.ndim)
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
        h = np.zeros((self.ndim, self.ndim))
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


class CubicSurface(quadratic.QuadraticSurface):
    """A twodimensional cubic surface"""

    def __init__(self, hessian=np.diag([0.0, 0.0]), gradient=np.zeros(2)):
        self.params = CubicSurfaceHelper(
            x2y=-0.1, y2x=-0.3, x3=0.1, y3=-0.5, factor=1.0, ndim=2
        )

        self.surf_quad = quadratic.QuadraticSurface(hessian=hessian, gradient=gradient)

        super().__init__(hessian=hessian, gradient=gradient, ndim=2)

    def setup_quapp_example(self, example_idx: int):
        """Sets up the parameters to reproduce the examples from the article
        'Gradient extremals and valley floor bifurcations on potential energy surfaces' by Quapp et al. (1988)
        """
        if example_idx == 2:
            # 1/2 * (xy + 2)*(y-x)
            self.surf_quad.params._hessian = np.zeros(shape=(2, 2))
            self.surf_quad.params.g0 = np.array([-1, 1])
            self.params.x2y = -0.5
            self.params.y2x = 0.5
            self.params.x3 = 0.0
            self.params.y3 = 0.0
        elif example_idx == 3:
            # 1/2 * (xy - 2)*(y-x)
            self.surf_quad.params._hessian = np.zeros(shape=(2, 2))
            self.surf_quad.params.g0 = np.array([1, -1])
            self.params.x2y = -0.5
            self.params.y2x = 0.5
            self.params.x3 = 0.0
            self.params.y3 = 0.0
        elif example_idx == 4:
            # 0.5 * (xy**2 - yx**2 + mu * x**2 + 2y - 3)
            self.surf_quad.params._hessian = self.params.mu * np.array(
                [[1.0, 0.0], [0.0, 0.0]]
            )  # Notice: the quadratic energy contains a factor of 0.5 already
            self.surf_quad.params.g0 = np.array([0.0, 1.0])
            self.params.mu = 1.0
            self.params.y2x = 0.5
            self.params.x2y = -0.5
            self.params.x3 = 0.0
            self.params.y3 = 0.0
        else:
            raise ValueError("Invalid example index")

    def energy(self, x: npt.NDArray) -> float:
        return self.surf_quad.energy(x) + self.params.energy(x)

    def gradient(self, x: npt.NDArray) -> npt.NDArray:
        return self.surf_quad.gradient(x) + self.params.gradient(x)

    def hessian(self, x: npt.NDArray) -> npt.NDArray:
        return self.surf_quad.hessian(x) + self.params.hessian(x)

    def scale_anharmonicity(self, factor):
        self.params.factor = factor
