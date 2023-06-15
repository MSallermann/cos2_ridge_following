from ridgefollowing import energy_surface
from ridgefollowing.algorithms import ridgefollower, modes, spherical_optimizer
import numpy.typing as npt
from typing import Optional, List, Union
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import numdifftools as nd


class CosineFollower(ridgefollower.RidgeFollower):
    def __init__(
        self,
        energy_surface: energy_surface.EnergySurface,
        maxiter: int = 1000,
        tolerance: Optional[float] = 1e-4,
        n_iterations_follow: int = 100,
        radius: float = 0.5e-2,
    ) -> None:
        super().__init__(energy_surface, n_iterations_follow)

        self.tolerance: float = tolerance
        self.maxiter: int = maxiter
        self.radius: float = radius

        self.width_modified_gaussian: float = 1.0
        self.magnitude_modified_gaussian: float = 0.0

        self.n_modes: int = 2

        self.maximize = True

    def setup_history(self):
        super().setup_history()
        self.history.update(
            dict(
                c2=np.zeros(shape=(self.n_iterations_follow)),
                grad_c2=np.zeros(shape=(self.n_iterations_follow, self.esurf.ndim)),
                eval_diff=np.zeros(shape=(self.n_iterations_follow)),
            )
        )

    def C(self, x: npt.ArrayLike, output_evals: Optional[npt.NDArray] = None) -> float:
        """Computes the cos function

        Args:
            x (npt.ArrayLike): the point at which to compute the cos function

        Returns:
            float: the value of the cos function
        """
        grad = self.esurf.gradient(x)
        grad /= np.linalg.norm(grad)
        evals, evecs = modes.lowest_n_modes(self.esurf.hessian(x), self.n_modes)

        if not output_evals is None:
            output_evals[:] = evals

        return np.dot(grad, evecs[:, 0])

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

    def C2_mod(self, x: npt.ArrayLike) -> npt.NDArray:
        """Computes the modified C2 value"""
        evals = np.zeros(self.n_modes)
        C = self.C(x, evals)
        return C**2 - self.magnitude_modified_gaussian * np.exp(
            -0.5 / self.width_modified_gaussian**2 * (evals[1] - evals[0]) ** 2
        )

    def grad_C2_mod(self, x: npt.ArrayLike) -> npt.NDArray:
        """Computes the gradient of the modified C2 value"""
        return nd.Gradient(self.C2_mod)(x)

    def C2_anharmonic(
        self, x0: npt.NDArray, g0: npt.NDArray, h0: npt.NDArray, x: npt.NDArray
    ):
        # hessian = self.esurf.hessian(x) - h0
        # grad = self.esurf.gradient(x) - g0 - h0@(x-x0)

        hessian = h0
        # hessian = self.esurf.hessian(x)

        grad = self.esurf.gradient(x)  # g0 + h0@(x-x0)
        grad = g0 + h0 @ (x - x0)
        grad /= np.linalg.norm(grad)

        evals, evecs = modes.lowest_n_modes(hessian, self.n_modes)
        return np.dot(grad, evecs[:, 0]) ** 2

    def find_maximum_on_ring(
        self, x0: npt.NDArray, d0: npt.NDArray
    ) -> List[npt.NDArray]:
        """Finds the maximum value of C2 on a ring with radius around x0. d0 is the initial guess for the direction of the maximum

        Args:
            x0 (npt.NDArray): current position
            d0 (npt.NDArray): initial guess for direction
            radius (float, optional): radius of ring. Defaults to 1.
        """
        d0 /= np.linalg.norm(d0)

        prefactor = -1.0 if self.maximize else 1.0

        def fun(d):
            """-C(x0 + radius*d)**2. Minus sign because we use scipy.minimize"""
            return prefactor * self.C2_mod(x0 + self.radius * d)

        def grad(d):
            """gradient of C(x0 + radius * d) wrt to d. Minus sign because we use scipy.minimize"""
            x = x0 + self.radius * d
            grad_c2_d = self.grad_C2_mod(x) * self.radius
            # project out component along d0
            grad_c2_d -= np.dot(grad_c2_d, d) * d
            return prefactor * grad_c2_d

        opt = spherical_optimizer.SphericalOptimization(
            fun=fun,
            grad=grad,
            ndim=self.esurf.ndim,
            tolerance=self.tolerance*self.radius,
            assert_success=True,
            disp=False,
        )
        res = opt.minimize(d0)

        return [res.x_opt, prefactor * res.f_opt, prefactor * res.g_opt]

    def sample_on_ring(self, x0: npt.NDArray, npoints=27, anharmonic: bool = False):
        """Sample C2 and the projection of grad_C2 on a ring

        Args:
            x0 (npt.NDArray): _description_
            npoints (int, optional): _description_. Defaults to 27.
        """
        assert self.esurf.ndim == 2  # only works for two dimensional surfaces

        phi = np.linspace(0, 2 * np.pi, npoints + 1)[:-1]
        c2 = np.zeros(shape=(npoints))
        grad_c2 = np.zeros(shape=(npoints, self.esurf.ndim + 1))
        dirs = []

        if anharmonic:
            h0 = self.esurf.hessian(x0)
            g0 = self.esurf.gradient(x0)

        for ip, p in enumerate(phi):
            d = np.array([np.cos(p), np.sin(p)])
            dirs.append(d)
            d_orth = np.array([-d[1], d[0]])
            x_cur = x0 + self.radius * d

            if anharmonic:
                c2[ip] = self.C2_anharmonic(x0, g0, h0, x_cur)
            else:
                c2[ip] = self.C2_mod(x_cur)
                grad_c2[ip] = np.dot(self.grad_C2_mod(x_cur), d_orth)

        return [phi, c2, grad_c2, np.array(dirs)]

    def find_all_maxima_on_ring(self, x0: npt.NDArray, npoints=27) -> List[npt.NDArray]:
        """Tries to locate all maxima on a ring of radius

        Args:
            x0 (npt.NDArray): _description_
            d0 (npt.NDArray): _description_

        Returns:
            List[npt.NDArray]: _description_
        """
        assert self.esurf.ndim == 2  # only works for two dimensional surfaces

        maxima = np.zeros(shape=(npoints, self.esurf.ndim + 5))

        for iphi, phi in enumerate(np.linspace(0, 2 * np.pi, npoints + 1)[:-1]):
            d0 = np.array([np.cos(phi), np.sin(phi)])
            d_opt, c2, grad_c2 = self.find_maximum_on_ring(x0, d0)
            phi_opt = np.arctan2(d_opt[1], d_opt[0])
            maxima[iphi] = [*d_opt, phi, phi_opt, c2, *grad_c2]

        # Filter out duplicate maxima
        return maxima

    def make_ring_sample_plot(self):
        import matplotlib.pyplot as plt
        path = self.output_path / Path(f"ring_plots/ring_iteration_{self._iteration}.png")
        path.parent.mkdir(exist_ok=True, parents=True)
        phi, c2, _, _ = self.sample_on_ring(self._x_cur, 128)

        def get_phi_form_dir(d):
            return (np.arctan2(d[1], d[0]) + 2*np.pi) % (2*np.pi) 

        phi_prev = get_phi_form_dir(self._step_cur)

        phi_cur = get_phi_form_dir(self._d_cur)

        plt.plot(phi, c2)
        plt.axvline(phi_cur, color="red")
        plt.axvline(phi_prev, color="blue")
        plt.savefig(path)
        plt.close()


    def determine_step(self):
        # Find maximum on ring
        self._d_cur, self._c2, self._grad_c2 = self.find_maximum_on_ring(
            self._x_cur, self._step_cur
        )

        if np.dot(self._d_cur, self._step_cur) < 0:
            print(f"Return detected at iteration {self._iteration}")
            print(f"x_cur = [{self._x_cur}]")
            print(modes.eigenpairs(self.esurf.hessian(self._x_cur)))

        return self.radius * self._d_cur

    def log_history(self):
        super().log_history()
        self.history["c2"][self._iteration] = self._c2
        self.history["grad_c2"][self._iteration] = self._grad_c2
