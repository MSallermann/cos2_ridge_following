import numpy.typing as npt
import numpy as np
import numdifftools as nd
from pydantic import BaseModel
from typing import Optional


class NonLinearConjugateGradient:
    class Result(BaseModel):
        class Config:
            arbitrary_types_allowed = True

        x_opt: npt.NDArray
        g_opt: npt.NDArray
        f_opt: float
        iterations_total: int

    def __init__(
        self,
        fun_grad_cb,
        ndim: int,
        maxiter: Optional[int] = 1000,
        maxiter_ls: int = 10,
        disp: bool = False,
        tolerance: float = 1e-6,
    ) -> None:
        self.fun_grad_cb = fun_grad_cb
        self.max_iter: int = maxiter
        self.disp: bool = disp
        self.tolerance: float = tolerance
        self.ndim = ndim

        # some local quantities
        self._iter: int = 0
        self._xcur: npt.NDArray = np.zeros(ndim)
        self._fcur: float = 0
        self._gcur: npt.NDArray = np.zeros(ndim)
        self.sn: npt.NDArray = np.zeros(ndim)

        self.alpha_prev = 1.0

        self.max_iter_ls = maxiter_ls
        self.wolfe_c1 = 1e-4
        self.wolfe_c2 = 0.1

        self.delta_xn = np.zeros(ndim)
        self.delta_xn_prev = np.zeros(ndim)
        self.delta_sn = np.zeros(ndim)

    def message(self, msg):
        if self.disp:
            print(msg)

    def beta_pr(self, delta_xn: npt.NDArray, delta_xn_prev: npt.NDArray) -> float:
        """Polak Ribiere beta with automatic direction reset.

        Args:
            delta_xn (npt.NDArray): steepest descent direction
            delta_xn_prev (npt.NDArray): previous steepest descent direction

        Returns:
            float: beta
        """
        snorm_delta_xn_prev = np.linalg.norm(delta_xn_prev) ** 2
        if snorm_delta_xn_prev > np.finfo(float).eps:
            beta = np.dot(delta_xn, delta_xn - delta_xn_prev) / snorm_delta_xn_prev
            return max(beta, 0.0)
        else:
            return 0.0

    def linesearch_parabola(self, initial_alpha_scale=1.0):
        self.message("---------------------------------")
        self.message("    Begin parabola linesearch    ")
        self.message("---------------------------------")

        iter = 0

        # normalized search direction
        s = self.sn / np.linalg.norm(self.sn)

        # initial step
        alpha = initial_alpha_scale * np.linalg.norm(self.sn)
        g_dot_s_cur = np.dot(s, self._gcur)

        # For the first iteration, we initialize these values here
        x = self._x_cur + alpha * s
        f_alpha, g_alpha = self.fun_grad_cb(x)
        g_dot_s = np.dot(g_alpha, s)

        # self.message(
        #     f"                          f_cur = {self._fcur:>10.3e}, g_dot_s_cur = {g_dot_s_cur:>10.3e}, f_alpha = {f_alpha:>10.3e}, g_dot_s = {g_dot_s:>10.3e}\n"
        # )

        dont_use_parabola = False

        while True:
            # Check for convergence
            if np.abs(g_dot_s) < self.tolerance and f_alpha < self._fcur:
                break

            # Fit a parabola from f_cur, f_alpha and g_alpha
            # p(a) = f_cur + g_cur * a + c*alpha**2
            c = (g_dot_s - g_dot_s_cur) / (2.0 * alpha)
            alpha_opt = -g_dot_s_cur / (2.0 * c)

            if alpha_opt < 0 or f_alpha > self._fcur or dont_use_parabola:
                dont_use_parabola = True
                # If alpha_opt would lead us backwards, so we use the Wolfe conditions instead
                self.message("Negative alpha detected. Using Wolfe conditions")
                armijo_condition = (
                    f_alpha < self._fcur + self.wolfe_c1 * alpha * g_dot_s_cur
                )

                print(f"armijo lhs {f_alpha}")
                print(f"armijo rhs {self._fcur + self.wolfe_c1 * alpha * g_dot_s_cur}")

                # curvature_condition = -np.dot(
                #     self.sn, g_alpha
                # ) < -self.wolfe_c2 * np.dot(self.sn, self._gcur)

                if not armijo_condition:
                    alpha /= 2
                # elif not curvature_condition:
                #     alpha *= 2
                else:
                    break
            else:
                alpha = alpha_opt

            x = self._x_cur + alpha * s
            f_alpha, g_alpha = self.fun_grad_cb(x)
            g_dot_s = np.dot(g_alpha, s)

            iter += 1
            if iter >= self.max_iter_ls:
                break

            # self.message(
            #     f"          Iteration {iter}:    alpha = {alpha:.3e}, f = {f_alpha:>10.3e}, |g| = {np.linalg.norm(g_alpha):>10.3e}"
            # )
            # self.message(
            #     f"                          f_cur = {self._fcur:>10.3e}, g_dot_s_cur = {g_dot_s_cur:>10.3e}, f_alpha = {f_alpha:>10.3e}, g_dot_s = {g_dot_s:>10.3e}\n"
            # )

        if self._fcur < f_alpha:
            print(self._fcur)
            print(f_alpha)
            raise Exception()

        self._x_cur = x
        self._fcur, self._gcur = f_alpha, g_alpha

        return alpha

    def minimize(self, x0: npt.NDArray) -> Result:
        self.message("Running NCG")
        self._iter = 0  # reset local iteration count
        run = True

        self._x_cur = x0.copy()
        self._fcur, self._gcur = self.fun_grad_cb(self._x_cur)

        if np.linalg.norm(self._gcur) < self.tolerance:
            run = False
            self.message("Gradient is already close to zero. Stopping")

        while run:
            self.delta_xn_prev = self.delta_xn  # save delta_xn_previous
            self.delta_xn = -self._gcur  # delta_xn is the direction of steepest descent
            beta = self.beta_pr(self.delta_xn, self.delta_xn_prev)

            self.message(f"    beta = {beta}")

            # Update conjugate direction
            self.sn *= beta
            self.sn += self.delta_xn

            # Reset direction if not a descent direction
            if np.dot(self.sn, self._gcur) >= 0:
                self.sn = self.delta_xn

            self.alpha_prev = self.linesearch_parabola(1.0)

            self._iter += 1

            if (
                np.linalg.norm(self._gcur) < self.tolerance
                or self._iter >= self.max_iter
            ):
                run = False

            self.message(
                f"Iteration {self._iter} (max. {self.max_iter}), f = {self._fcur:.3e}, |g| = {np.linalg.norm(self._gcur):.3e}"
                + "\n"
                + 50 * "="
            )

        return NonLinearConjugateGradient.Result(
            x_opt=np.array(self._x_cur),
            g_opt=np.array(self._gcur),
            f_opt=self._fcur,
            iterations_total=self._iter,
        )
