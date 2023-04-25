from ridgefollowing import energy_surface
from ridgefollowing.algorithms import modes
import numpy.typing as npt
from typing import Optional, List
import numpy as np
from scipy.optimize import minimize
import numdifftools as nd


class RidgeFollower:
    def __init__(
        self,
        energy_surface: energy_surface.EnergySurface,
        maxiter: int = 1000,
        tolerance: Optional[float] = None,
        n_iterations_folow: int = 100,
        radius: float = 0.5e-2,
    ) -> None:
        self.esurf: energy_surface.EnergySurface = energy_surface
        self.tolerance: float = tolerance
        self.maxiter: int = maxiter
        self.radius: float = radius
        self.n_iterations_follow: int = n_iterations_folow

        self.history: List[npt.NDArray] = []

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
        return self.fd_grad_C(x)

    def find_maximum_on_ring(self, x0: npt.NDArray, d0: npt.NDArray):
        """Finds the maximum value of C2 on a ring with radius around x0. d0 is the initial guess for the direction of the maximum

        Args:
            x0 (npt.NDArray): current position
            d0 (npt.NDArray): initial guess for direction
            radius (float, optional): radius of ring. Defaults to 1.
        """
        d0 /= np.linalg.norm(d0)

        def fun(d):
            """-C(x0 + radius*d)**2. Minus us sign because we use scipy.minimize"""
            return -self.C(x0 + self.radius * d) ** 2

        def grad(d):
            """gradient of C(x0 + radius * d) wrt to d. Minus us sign because we use scipy.minimize"""
            x = x0 + self.radius * d
            grad_c2_d = 2 * self.C(x) * self.grad_C(x) * self.radius
            # project out component along d0
            grad_c2_d -= np.dot(grad_c2_d, d) * d
            return -grad_c2_d

        def cb(d):
            """renormalize after every iteration"""
            d /= np.linalg.norm(d)

        res = minimize(
            fun,
            d0,
            method="CG",
            jac=grad,
            tol=self.tolerance,
            options=dict(maxiter=self.maxiter, disp=False),
            callback=cb,
        )

        return [res.x, -res.fun, -res.jac]

    def follow(self, x0: npt.NDArray, d0: npt.NDArray):
        self.history = []

        x_cur = np.array(x0)
        d_cur = np.array(d0)

        for i in range(self.n_iterations_follow):
            # Find maximum on ring
            d_cur, c2, grad_c2 = self.find_maximum_on_ring(x_cur, d_cur)
            x_cur += self.radius * d_cur

            E = self.esurf.energy(x_cur)
            self.history.append(np.array([*x_cur, *d_cur, c2, *grad_c2, E]))
