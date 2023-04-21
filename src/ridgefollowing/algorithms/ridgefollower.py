from ridgefollowing import energy_surface
from ridgefollowing.algorithms import modes
import numpy.typing as npt
import numpy as np
import numdifftools as nd


class RidgeFollower:
    def __init__(self, energy_surface=energy_surface.EnergySurface) -> None:
        self.esurf = energy_surface

    def C(self, x: npt.ArrayLike) -> float:
        """Computes the cos function

        Args:
            x (npt.ArrayLike): the point at which to compute the cos function

        Returns:
            float: the value of the cos function
        """
        grad = self.esurf.gradient(x)
        grad /= np.linalg.norm(grad)
        mode = modes.lowest_mode(self.esurf.hessian(x))[1]
        return np.dot(grad, mode)

    def fd_grad_C(self, x: npt.ArrayLike) -> npt.NDArray:
        """Computes the gradient of the cos function, using finite differences

        Args:
            x (npt.ArrayLike): the point at which to compute the cos function

        Returns:
            npt.NDArray: the gradient of the cos function
        """
        return nd.Gradient(self.C)(x)

    def grad_C(self, x: npt.ArrayLike) -> npt.NDArray:
        """Computes the gradient of the cos function

        Args:
            x (npt.ArrayLike): the point at which to compute the cos function

        Returns:
            npt.NDArray: the gradient of the cos function
        """
        pass
