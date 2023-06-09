from ridgefollowing import energy_surface
import numpy as np
import numpy.typing as npt
from typing import Optional


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
            self._g0 = np.zeros(ndim)
        else:
            self._g0 = gradient

    def energy(self, x: npt.ArrayLike) -> float:
        return np.dot(self._g0, x) + 0.5 * np.transpose(x) @ self._hessian @ x

    def gradient(self, x: npt.ArrayLike) -> npt.NDArray:
        return self._g0 + self._hessian @ x

    def hessian(self, x: npt.ArrayLike) -> npt.NDArray:
        return self._hessian.copy()
