from ridgefollowing import energy_surface
import numpy as np
import numpy.typing as npt


class QuadraticSurface(energy_surface.EnergySurface):
    def __init__(
        self,
        hessian: npt.NDArray,
        ndim: int = 2,
    ):
        super().__init__(ndim)
        self._hessian = hessian

    def energy(self, x: npt.ArrayLike) -> float:
        return 0.5 * np.transpose(x) @ self._hessian @ x

    def gradient(self, x: npt.ArrayLike) -> npt.NDArray:
        return self._hessian @ x

    def hessian(self, x: npt.ArrayLike) -> npt.NDArray:
        return self._hessian
