import abc
import numpy as np
import numpy.typing as npt
import numdifftools as nd


class EnergySurface(abc.ABC):
    """Abstract base class for energy surfaces. Any deriving class has to implement at least the energy function."""

    def __init__(self, ndim):
        self.ndim = ndim

    @abc.abstractmethod
    def energy(self, x: npt.NDArray) -> float:
        """Energy at point x

        Args:
            x (npt.NDArray): the point on the energy surface

        Returns:
            float: energy
        """
        # Overwrite in child classes
        ...

    def gradient(self, x: npt.NDArray) -> npt.NDArray:
        """Gradient of energy

        Args:
            x (npt.NDArray): the point on the energy surface

        Returns:
            npt.NDArray: the gradient
        """
        # Fallback uses FD
        return self.fd_gradient(x)

    def directional_gradient(self, x: npt.NDArray, dir: npt.NDArray) -> npt.NDArray:
        """Directional derivative

        Args:
            x (npt.NDArray): the point on the energy surface
            dir (npt.NDArray): the direction

        Returns:
            npt.NDArray: dot product of gradient and direction
        """
        dir_n = dir / np.linalg.norm(dir)
        return np.dot(self.gradient(x), dir_n)

    def curvature(self, x: npt.NDArray, dir: npt.NDArray) -> npt.NDArray:
        """The curvature at point x in direction dir

        Args:
            x (npt.NDArray): the point on the energy surface
            dir (npt.NDArray): the direction

        Attention!: dir must be normalized

        Returns:
            np.ArrayLike: the curvature, equivalent to Hessian * dir
        """
        # Fallback uses FD
        return self.fd_curvature(x, dir)

    def hessian(self, x: npt.NDArray) -> npt.NDArray:
        """Hessian matrix

        Args:
            x (npt.NDArray): point at which to compute the hessian

        Returns:
            npt.NDArray: the hessian
        """
        return self.fd_hessian(x)

    def fd_gradient(self, x: npt.NDArray) -> npt.NDArray:
        """Gradient of energy, computed with finite differences

        Args:
            x (npt.NDArray): the point on the energy surface

        Returns:
            npt.NDArray: the gradient
        """
        # Fallback uses FD
        return nd.Gradient(self.energy)(x)

    def fd_curvature(self, x: npt.NDArray, dir: npt.NDArray) -> npt.NDArray:
        """The curvature at point x in direction dir, computed with finite differences

        Args:
            x (npt.NDArray): the point on the energy surface
            dir (npt.NDArray): the direction

        Attention!: dir must be normalized

        Returns:
            np.ArrayLike: the curvature, equivalent to Hessian * dir
        """
        return nd.Gradient(self.directional_gradient)(x, dir)

    def fd_hessian(self, x: npt.NDArray) -> npt.NDArray:
        """Hessian matrix using finite differences

        Args:
            x (npt.NDArray): point at which to compute the hessian

        Returns:
            npt.NDArray: the hessian
        """
        return nd.Hessian(self.energy)(x)
