from ridgefollowing import energy_surface
import numpy as np
import numpy.typing as npt


class CosineSSBENCH(energy_surface.EnergySurface):
    def __init__(
        self,
    ):
        super().__init__(ndim=2)
        self.A1 = 0.5
        self.Bxy1 = 0.2
        self.Bx1 = 0.6
        self.By1 = 0.5

        self.A2 = 1
        self.Bx2 = 1
        self.By2 = 1.5

        self.A3 = -1
        self.B3 = 0.008
        self.X3 = 17
        self.Y3 = 17

    def f1(self, x, y):
        return (
            self.A1
            * np.cos(self.Bxy1 * x * y)
            * np.cos(self.Bx1 * x)
            * np.cos(self.By1 * y)
        )

    def grad_f1(self, x, y):
        g_1 = (
            self.A1
            * np.cos(self.By1 * y)
            * (
                -self.Bxy1 * y * np.sin(self.Bxy1 * x * y) * np.cos(self.Bx1 * x)
                - self.Bx1 * np.cos(self.Bxy1 * x * y) * np.sin(self.Bx1 * x)
            )
        )
        g_2 = (
            self.A1
            * np.cos(self.Bx1 * x)
            * (
                -self.Bxy1 * x * np.sin(self.Bxy1 * x * y) * np.cos(self.By1 * y)
                - self.By1 * np.cos(self.Bxy1 * x * y) * np.sin(self.By1 * y)
            )
        )
        return np.array([g_1, g_2])

    def hess_f1(self, x, y):
        hxx = (
            self.A1
            * np.cos(self.By1 * y)
            * (
                -self.Bxy1
                * y
                * self.Bxy1
                * y
                * np.cos(self.Bxy1 * x * y)
                * np.cos(self.Bx1 * x)
                + 2.0
                * self.Bxy1
                * y
                * self.Bx1
                * np.sin(self.Bxy1 * x * y)
                * np.sin(self.Bx1 * x)
                - self.Bx1 * self.Bx1 * np.cos(self.Bxy1 * x * y) * np.cos(self.Bx1 * x)
            )
        )
        hyy = (
            self.A1
            * np.cos(self.Bx1 * x)  ### !Mistake was *y before
            * (
                -self.Bxy1
                * x
                * self.Bxy1
                * x
                * np.cos(self.Bxy1 * x * y)
                * np.cos(self.By1 * y)
                + 2.0
                * self.Bxy1
                * x
                * self.By1
                * np.sin(self.Bxy1 * x * y)
                * np.sin(self.By1 * y)
                - self.By1 * self.By1 * np.cos(self.Bxy1 * x * y) * np.cos(self.By1 * y)
            )
        )
        hxy = self.A1 * (
            self.By1
            * np.sin(self.By1 * y)
            * (
                self.Bxy1 * y * np.sin(self.Bxy1 * x * y) * np.cos(self.Bx1 * x)
                + self.Bx1 * np.cos(self.Bxy1 * x * y) * np.sin(self.Bx1 * x)
            )
            + np.cos(self.By1 * y)
            * (
                self.Bxy1
                * np.cos(self.Bx1 * x)
                * (
                    -np.sin(self.Bxy1 * x * y)
                    - x * y * self.Bxy1 * np.cos(self.Bxy1 * x * y)
                )
                + self.Bx1
                * self.Bxy1
                * x
                * np.sin(self.Bxy1 * x * y)
                * np.sin(self.Bx1 * x)
            )
        )
        return np.array([[hxx, hxy], [hxy, hyy]])

    def f2(self, x, y):
        return self.A2 * np.cos(self.Bx2 * x) * np.cos(self.By2 * y)

    def f3(self, x, y):
        return self.A3 * np.exp(
            -self.B3 * ((x - self.X3) ** 2.0 + (y - self.Y3) ** 2.0)
        )

    def grad_f2(self, x, y):
        g_1 = -self.A2 * self.Bx2 * np.sin(self.Bx2 * x) * np.cos(self.By2 * y)
        g_2 = -self.A2 * self.By2 * np.cos(self.Bx2 * x) * np.sin(self.By2 * y)
        return np.array([g_1, g_2])

    def hess_f2(self, x, y):
        hxx = (
            -self.A2 * self.Bx2 * self.Bx2 * np.cos(self.Bx2 * x) * np.cos(self.By2 * y)
        )
        hyy = (
            -self.A2 * self.By2 * self.By2 * np.cos(self.Bx2 * x) * np.cos(self.By2 * y)
        )
        hxy = (
            self.A2 * self.Bx2 * self.By2 * np.sin(self.Bx2 * x) * np.sin(self.By2 * y)
        )
        return np.array([[hxx, hxy], [hxy, hyy]])

    def grad_f3(self, x, y):
        g_1 = (
            self.A3
            * (-self.B3 * 2.0 * (x - self.X3))
            * np.exp(-self.B3 * ((x - self.X3) ** 2.0 + (y - self.Y3) ** 2.0))
        )
        g_2 = (
            self.A3
            * (-self.B3 * 2.0 * (y - self.Y3))
            * np.exp(-self.B3 * ((x - self.X3) ** 2.0 + (y - self.Y3) ** 2.0))
        )
        return np.array([g_1, g_2])

    def hess_f3(self, x, y):
        hxx = (
            self.A3
            * ((-self.B3 * 2.0 * (x - self.X3)) ** 2.0 - 2.0 * self.B3)
            * np.exp(-self.B3 * ((x - self.X3) ** 2.0 + (y - self.Y3) ** 2.0))
        )
        hyy = (
            self.A3
            * ((-self.B3 * 2.0 * (y - self.Y3)) ** 2.0 - 2.0 * self.B3)
            * np.exp(-self.B3 * ((x - self.X3) ** 2.0 + (y - self.Y3) ** 2.0))
        )
        hxy = (
            self.A3
            * (-self.B3 * 2.0 * (x - self.X3))
            * (-self.B3 * 2.0 * (y - self.Y3))
            * np.exp(-self.B3 * ((x - self.X3) ** 2.0 + (y - self.Y3) ** 2.0))
        )
        return np.array([[hxx, hxy], [hxy, hyy]])

    def energy(self, x: npt.NDArray) -> float:
        return self.f1(*x) + self.f2(*x) + self.f3(*x)

    def gradient(self, x: npt.NDArray) -> npt.NDArray:
        return self.grad_f1(*x) + self.grad_f2(*x) + self.grad_f3(*x)

    def hessian(self, x: npt.NDArray) -> npt.NDArray:
        return self.hess_f1(*x) + self.hess_f2(*x) + self.hess_f3(*x)
