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
        tolerance: Optional[float] = 1e-8,
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

        self._ridge_width: float = 0.0

        self.bifurcation_points = []

        self.maximize = True

    def setup_history(self):
        super().setup_history()
        self.history.update(
            dict(
                c2=np.zeros(shape=(self.n_iterations_follow)),
                grad_c2=np.zeros(shape=(self.n_iterations_follow, self.esurf.ndim)),
                eval_diff=np.zeros(shape=(self.n_iterations_follow)),
                ridge_width=np.zeros(shape=(self.n_iterations_follow)),
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

        order = 2
        grad, res = nd.Gradient(self.C2_mod, order=order, full_output=True)(x)

        while np.max(res.error_estimate) > self.tolerance and order <= 8:
            order *= 2
            grad, res = nd.Gradient(self.C2_mod, order=order, full_output=True)(x)

        return grad

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
            g = self.esurf.gradient(x)
            g /= np.linalg.norm(g)

            grad_c2_d -= np.dot(grad_c2_d, d) * d
            # grad_c2_d -= np.dot(grad_c2_d, g) * g

            return prefactor * grad_c2_d

        opt = spherical_optimizer.SphericalOptimization(
            fun=fun,
            grad=grad,
            ndim=self.esurf.ndim,
            tolerance=self.tolerance * self.radius,
            maxiter=10000,
            assert_success=False,
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

    def make_ring_sample_plot(self, show=False, N_phi=128):
        import matplotlib.pyplot as plt

        path = self.output_path / Path(
            f"ring_plots/ring_iteration_{self._iteration}.png"
        )
        path.parent.mkdir(exist_ok=True, parents=True)
        phi, c2, _, _ = self.sample_on_ring(self._x_cur, N_phi)

        def get_phi_from_dir(d):
            return (np.arctan2(d[1], d[0]) + 2 * np.pi) % (2 * np.pi)

        phi_prev = get_phi_from_dir(self._step_cur)

        phi_cur = get_phi_from_dir(self._d_cur)

        plt.plot(phi, c2)
        plt.axvline(phi_cur, color="red", label="opt")
        plt.axvline(phi_prev, color="blue", label="init")
        plt.legend()
        plt.savefig(path)
        if show:
            plt.show()
        plt.close()

    def make_stereographic_sample_plot(
        self, show=False, gradient=False, N_sample=128, x_range=[-3, 3]
    ):
        import matplotlib.pyplot as plt

        assert self.esurf.ndim == 2

        prefactor = -1.0 if self.maximize else 1.0
        x0 = self._x_cur

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
            tolerance=self.tolerance * self.radius,
            maxiter=10000,
            assert_success=False,
            disp=False,
        )

        x_stereo = np.linspace(x_range[0], x_range[1], N_sample)
        f_stereo = [opt.f_stereo(np.array([x])) for x in x_stereo]
        g_stereo = [opt.grad_stereo(np.array([x])) for x in x_stereo]

        x_stereo_prev = opt.embed_to_stereo(self._step_cur)
        x_stereo_cur = opt.embed_to_stereo(self._d_cur)

        path = self.output_path / Path(
            f"stereo_plots/ring_iteration_{self._iteration}.png"
        )
        path.parent.mkdir(exist_ok=True, parents=True)

        plt.plot(x_stereo, f_stereo, label="f")
        if gradient:
            plt.plot(x_stereo, g_stereo, label="g")
        plt.axvline(x_stereo_cur, color="red", label="opt")
        plt.axvline(x_stereo_prev, color="blue", label="init")
        plt.legend()
        plt.savefig(path)
        if show:
            plt.show()
        plt.close()

    def compute_ridge_width(self, step):
        assert (
            self.esurf.ndim == 2
        )  # This implementation only works for ndim=2, for higher dimensions the minimal width should be computed (??)

        # Direction orthogonal to current search direction
        orth_dir = np.array([-step[0], step[1]])
        orth_dir /= np.linalg.norm(orth_dir)

        def fun(a):
            return self.C2_mod(self._x_cur + a * orth_dir)

        order = 2
        d, res = nd.Derivative(fun, n=2, full_output=True, order=order)(0.0)

        while np.max(res.error_estimate) > self.tolerance and order <= 8:
            order *= 2
            d, res = nd.Derivative(fun, n=2, full_output=True, order=order)(0.0)

        # self._ridge_width = 1.0/(1.0 - d)
        self._ridge_width = d

    def determine_step(self):
        # Find maximum on ring
        self._d_cur, self._c2, self._grad_c2 = self.find_maximum_on_ring(
            self._x_cur, self._step_prev
        )

        if np.dot(self._d_cur, self._step_prev) < 0:
            print(f"Return detected at iteration {self._iteration}")
            print(f"x_cur = [{self._x_cur}]")
            print(modes.eigenpairs(self.esurf.hessian(self._x_cur)))

        # self.make_ring_sample_plot()

        step = self.radius * self._d_cur
        self.compute_ridge_width(step)

        if self._iteration > 3:
            W_3 = self._ridge_width
            W_2 = self.history["ridge_width"][self._iteration - 1]
            W_1 = self.history["ridge_width"][self._iteration - 2]

            E_3 = self._E
            E_2 = self.history["E"][self._iteration - 1]
            E_1 = self.history["E"][self._iteration - 2]

            width_criterion = W_2 > W_3 and W_2 > W_1
            energy_criterion = (E_3 > E_2 and E_2 > E_1) or (E_3 < E_2 and E_2 < E_1)

            if width_criterion and energy_criterion:
                print(f"Potential bifurcation at iteration {self._iteration-1}")
                self.bifurcation_points.append([self._x_cur, self._step_cur])

        return step

    def log_history(self):
        super().log_history()
        self.history["c2"][self._iteration] = self._c2
        self.history["grad_c2"][self._iteration] = self._grad_c2
        self.history["ridge_width"][self._iteration] = self._ridge_width
