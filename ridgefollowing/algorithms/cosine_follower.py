from energy_surfaces import energy_surface
from ridgefollowing.algorithms import ridgefollower, modes, spherical_optimizer
import numpy.typing as npt
from typing import Optional, List, Union
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import numdifftools as nd
from numba import njit


class CosineFollower(ridgefollower.RidgeFollower):
    def __init__(
        self,
        energy_surface: energy_surface.EnergySurface,
        maxiter: int = 1000,
        output_path: Optional[Path] = None,
        tolerance: Optional[float] = 1e-10,
        n_iterations_follow: int = 100,
        radius: float = 0.5e-2,
    ) -> None:
        super().__init__(energy_surface, n_iterations_follow, output_path)

        self.tolerance: float = tolerance
        self.tolerance_grad: float = tolerance

        self.maxiter: int = maxiter
        self.radius: float = radius

        self.width_modified_gaussian: float = 1.0
        self.magnitude_modified_gaussian: float = 0.0
        self.n_modes: int = 2

        self._d_cur = np.zeros(shape=(self.esurf.ndim))
        self._x_cur_temporary_save = np.zeros(shape=(self.esurf.ndim))

        self._ridge_width: float = 0.0
        self._ridge_width_fw: float = 0.0
        self._ridge_width_bw: float = 0.0

        self._ridge_width3: float = 0.0
        self._ridge_width4: float = 0.0

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
                ridge_width_fw=np.zeros(shape=(self.n_iterations_follow)),
                ridge_width_bw=np.zeros(shape=(self.n_iterations_follow)),
                ridge_width3=np.zeros(shape=(self.n_iterations_follow)),
                ridge_width4=np.zeros(shape=(self.n_iterations_follow)),
                c2_initial_guess_ratio=np.zeros(shape=(self.n_iterations_follow)),
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

    def adaptive_gradient(self, fun, x, n=1, method="central"):
        order = 2
        richardson_terms = 2

        grad_f, res = nd.Gradient(
            fun,
            order=order,
            method=method,
            full_output=True,
            richardson_terms=richardson_terms,
            n=n,
        )(x)
        err_f = np.max(res.error_estimate)

        while err_f > self.tolerance_grad and order <= 8 and richardson_terms <= 8:
            order *= 2
            grad, res = nd.Gradient(
                fun,
                order=order,
                method=method,
                full_output=True,
                richardson_terms=richardson_terms,
                n=n,
            )(x)
            err = np.max(res.error_estimate)
            if err < err_f:
                grad_f = grad.copy()
                err_f = err

        # if err_f > tolerance:
        # print(f"WARNING: current error is {err_f:.1e} > {tolerance}")

        return grad_f

    def grad_C2_mod(self, x: npt.ArrayLike) -> npt.NDArray:
        """Computes the gradient of the modified C2 value"""
        return self.adaptive_gradient(self.C2_mod, x)

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
            tolerance=self.tolerance * self.radius * 1e-2,
            maxiter=10000,
            assert_success=False,
            disp=False,
        )
        res = opt.minimize(d0)

        return [res.x_opt, prefactor * res.f_opt, prefactor * res.g_opt]

    def find_maximum_on_hyperplane(
        self, x0: npt.NDArray, normal: npt.NDArray, max_dist: float
    ):
        normal /= np.linalg.norm(normal)

        prefactor = -1.0 if self.maximize else 1.0

        def fun(delta_x):
            return prefactor * self.C2_mod(
                x0 + delta_x - np.dot(delta_x, normal) * normal
            )

        def grad(delta_x):
            grad_c2 = self.grad_C2_mod(x0 + delta_x - np.dot(delta_x, normal) * normal)
            # project out component along normal
            grad_c2 -= np.dot(grad_c2, normal) * normal
            return prefactor * grad_c2

        res = minimize(fun=fun, x0=np.zeros(self.esurf.ndim), jac=grad, method="CG")
        delta_x = res.x

        self._c2 = prefactor * res.fun
        self._grad_c2 = prefactor * res.jac

        n_delta_x = np.linalg.norm(delta_x)

        if n_delta_x > max_dist:
            delta_x = delta_x * max_dist / n_delta_x

        return delta_x

    def sample_on_ring(
        self,
        x0: npt.NDArray,
        npoints=27,
        anharmonic: bool = False,
        radial_factor: float = 1.0,
    ):
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
            x_cur = x0 + self.radius * radial_factor * d

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

    def make_ring_sample_plot(self, show=False, N_phi=128, radial_factor=1.0):
        import matplotlib.pyplot as plt

        if self.output_path is None:
            raise Exception("Output path needs to be set")

        folder = self.output_path / "ring_plots"
        folder.mkdir(exist_ok=True)

        phi, c2, _, _ = self.sample_on_ring(
            x0=self._x_cur, npoints=N_phi, radial_factor=radial_factor
        )

        def get_phi_from_dir(d):
            return (np.arctan2(d[1], d[0]) + 2 * np.pi) % (2 * np.pi)

        phi_prev = get_phi_from_dir(self._step_cur)

        phi_cur = get_phi_from_dir(self._d_cur)

        plt.plot(phi, c2)
        plt.axvline(phi_cur, color="red", label="opt")
        plt.axvline(phi_prev, color="blue", label="init")
        plt.legend()
        plt.savefig(
            folder / f"iteration_{self._iteration}_r_factor_{radial_factor:.1f}.png"
        )
        if show:
            plt.show()
        plt.close()

    def make_stereographic_sample_plot(
        self, show=False, gradient=False, N_sample=128, x_range=[-3, 3]
    ):
        import matplotlib.pyplot as plt

        if self.output_path is None:
            raise Exception("Output path needs to be set")

        folder = self.output_path / "stereo_plots"
        folder.mkdir(exist_ok=True)

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

        x_stereo_prev = opt.embed_to_stereo(
            self._step_cur / np.linalg.norm(self._step_cur)
        )
        x_stereo_cur = opt.embed_to_stereo(self._d_cur)

        plt.plot(x_stereo, f_stereo, label="f")
        if gradient:
            plt.plot(x_stereo, g_stereo, label="g")
        plt.axvline(x_stereo_cur, color="red", label="opt")
        plt.axvline(x_stereo_prev, color="blue", label="init")
        plt.legend()
        plt.savefig(folder / f"iteration_{self._iteration}.png")
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
            a = np.atleast_1d(a)
            return self.C2_mod(self._x_cur + a[0] * orth_dir)

        self._ridge_width = self.adaptive_gradient(fun, x=np.zeros(1), n=2)

        C2p = fun(self.radius)
        C20 = fun(0)
        C2m = fun(-self.radius)

        self._ridge_width_fw = (C2p - C20) / self.radius
        self._ridge_width_bw = (C2m - C20) / self.radius

        # self._ridge_width_fw = self.adaptive_gradient(
        #     fun, x=np.zeros(1), tolerance=self.tolerance, method="forward", n=2
        # )
        # self._ridge_width_bw = self.adaptive_gradient(
        #     fun, x=np.zeros(1), tolerance=self.tolerance, method="backward", n=2
        # )

    def feel_out_ridge(self, x0, search_direction, n_factors=1):
        """Try to 'feel' out a ridge in search_direction, by increasing the step size gradually"""

        radius_original = self.radius
        success = False

        factor_list = [2**i for i in range(n_factors)]

        for f in factor_list:
            print(f"f={f}")
            self.radius = radius_original * f
            # Find maximum on ring
            self._d_cur, self._c2, self._grad_c2 = self.find_maximum_on_ring(
                x0, search_direction
            )

            # Check the overlap
            if not np.dot(self._d_cur, search_direction) < 0.5:
                success = True
                break

        self.radius = radius_original
        return success

    def get_dir_with_max_verlap(self, v0, s):
        if np.dot(v0, s) < 0:
            return -s
        else:
            return s

    def locate_ridge(self, x0, search_direction, n_factors):
        """Try to locate a ridge starting from x0 in direction"""

        factor_list = [2**i for i in range(n_factors)]

        # First we try to locate a tangent
        success = self.feel_out_ridge(x0, search_direction, n_factors)

        if success:
            return self.radius * self._d_cur
        else:
            for f in factor_list:
                delta_x = self.find_maximum_on_hyperplane(
                    x0 + f * self.radius * search_direction,
                    normal=search_direction,
                    max_dist=10 * self.radius * f,
                )
                l = self.get_dir_with_max_verlap(
                    v0=search_direction, s=self.grad_C2_mod(x0 + delta_x)
                )
                success = self.feel_out_ridge(
                    x0 + delta_x, search_direction=l, n_factors=n_factors
                )
                if success:
                    break
            if not success:
                print(f"Cannot locate ridge at iteration {self._iteration}")
                print(f"x_cur = [{self._x_cur}]")
                print("Stopping")
                self.stop()
            return self.radius * self._d_cur + delta_x

    def determine_step(self):
        self._x_cur_temporary_save = self._x_cur.copy()
        dir_prev = self._step_prev / np.linalg.norm(self._step_prev)

        if self._iteration == 0:
            self.bifurcation_points.clear()

        step = self.locate_ridge(self._x_cur, dir_prev, n_factors=5)

        # success = self.feel_out_ridge(x0=self._x_cur, search_direction=dir_prev, n_factors=1)
        # step = self.radius * self._d_cur

        # normal = self._d_cur
        # normal = dir_prev
        # delta_x = self.find_maximum_on_hyperplane(self._x_cur + self.radius*normal, normal=normal)
        # step = self.radius*normal + delta_x

        # if not success:
        #     print(f"Return detected at iteration {self._iteration}")
        #     print(f"x_cur = [{self._x_cur}]")
        #     print(self._x_cur)
        #     print(modes.eigenpairs(self.esurf.hessian(self._x_cur)))
        #     print("Stopping")
        #     for f in [1, 2, 4, 8, 16, 32, 64, 128]:
        #         self.make_ring_sample_plot(radial_factor=float(f))

        #     self.make_stereographic_sample_plot()
        #     self.stop()

        self.compute_ridge_width(dir_prev)

        if self._iteration > 3:
            W_3 = self._ridge_width
            W_2 = self.history["ridge_width"][self._iteration - 1]
            W_1 = self.history["ridge_width"][self._iteration - 2]

            gradient_criterion = (
                self.history["G_norm"][self._iteration - 1] > 1e-2
            )  # Not a stationary point

            width_criterion = W_2 > W_3 and W_2 > W_1

            c2_critical = 1.0 - 1e-5

            dir_cur = step / np.linalg.norm(step)
            angle_criterion = np.dot(dir_prev, dir_cur) < 0.25

            main_ridge_to_side_ridge = (
                self.history["c2"][self._iteration - 1] > c2_critical
                and self._c2 <= c2_critical
            )
            side_ridge_to_main_ridge = (
                self.history["c2"][self._iteration - 1] < c2_critical
                and self._c2 >= c2_critical
            )

            if (
                (width_criterion and gradient_criterion)
                or angle_criterion
                or main_ridge_to_side_ridge
                or side_ridge_to_main_ridge
            ):
                print(f"Potential bifurcation at iteration {self._iteration-1}")
                x_bif = self.history["x_cur"][self._iteration - 1]

                if angle_criterion:
                    dir_bif = dir_cur
                else:
                    dir_bif = self.history["step_cur"][self._iteration - 1]

                self.bifurcation_points.append(
                    [x_bif, np.array([-dir_bif[1], dir_bif[0]])]
                )

                # self.make_ring_sample_plot()
                # self.make_stereographic_sample_plot()

        return step

    def log_history(self):
        super().log_history()
        self.history["c2"][self._iteration] = self._c2
        self.history["grad_c2"][self._iteration] = self._grad_c2
        self.history["ridge_width"][self._iteration] = self._ridge_width
        self.history["ridge_width_fw"][self._iteration] = self._ridge_width_fw
        self.history["ridge_width_bw"][self._iteration] = self._ridge_width_bw
        self.history["ridge_width3"][self._iteration] = self._ridge_width3
        self.history["ridge_width4"][self._iteration] = self._ridge_width4
        self.history["c2_initial_guess_ratio"][self._iteration] = (
            self.C2_mod(self._x_cur + self._step_cur) / self._c2
        )

        evals, evecs = modes.eigenpairs(self.esurf.hessian(self._x_cur))
        self.history["eval_diff"][self._iteration] = evals[1] - evals[0]
