from ridgefollowing import energy_surface
import numpy as np
import numpy.typing as npt
from typing import Optional
from numba.experimental import jitclass
from numba import float64, int32


@jitclass(
    [
        ("g0", float64[::1]),
        ("_hessian", float64[:, ::1]),
    ]
)
class QuadraticSurfaceHelper:
    def __init__(self, hessian: npt.NDArray, g0: npt.NDArray):
        self.g0 = g0
        self._hessian = hessian

    def energy(self, x: npt.NDArray) -> float:
        return self.g0 @ x + 0.5 * x @ self._hessian @ x

    def gradient(self, x: npt.NDArray) -> npt.NDArray:
        return self.g0 + self._hessian @ x

    def hessian(self, x: npt.NDArray) -> npt.NDArray:
        return self._hessian.copy()


class QuadraticSurface(energy_surface.EnergySurface):
    def __init__(
        self,
        hessian: npt.NDArray,
        gradient: Optional[npt.NDArray] = None,
        ndim: int = 2,
    ):
        super().__init__(ndim)

        self._hessian = hessian
        if gradient is None:
            g0 = np.zeros(ndim, dtype=float)
        else:
            g0 = gradient

        self.params = QuadraticSurfaceHelper(hessian=hessian, g0=g0)

    def energy(self, x: npt.ArrayLike) -> float:
        return self.params.energy(x)

    def gradient(self, x: npt.ArrayLike) -> npt.NDArray:
        return self.params.gradient(x)

    def hessian(self, x: npt.ArrayLike) -> npt.NDArray:
        return self.params.hessian(x)
