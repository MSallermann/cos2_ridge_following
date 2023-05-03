import numpy.typing as npt
import numpy as np
import numdifftools as nd


class NonLinearConjugateGradient:
    def __init__(
        self,
        fun_grad_cb,
        ndim: int,
        maxiter: int = 1000,
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
        """Polate Ribiere beta. With automatic direction reset.

        Args:
            delta_xn (npt.NDArray): steepest descent direction
            delta_xn_prev (npt.NDArray): previous steepest descent direction

        Returns:
            float: beta
        """
        snorm_delta_xn_prev = np.linalg.norm(delta_xn_prev) ** 2
        if snorm_delta_xn_prev > np.finfo(float).eps:
            beta = np.dot(delta_xn, delta_xn - delta_xn_prev) / snorm_delta_xn_prev
            return np.max(beta, 0)
        else:
            return 0.0

    def linesearch(self, initial_alpha=1.0):
        alpha = initial_alpha
        run = True

        factor = 2
        iter = 0

        armijo_condition = False
        curvature_condition = False

        self.message("    Begin Linesearch")
        while run:
            x = self._x_cur + alpha * self.sn
            f_alpha, g_alpha = self.fun_grad_cb(x)

            armijo_condition = f_alpha < self._fcur + self.wolfe_c1 * alpha * np.dot(
                self.sn, self._gcur
            )
            curvature_condition = -np.dot(self.sn, g_alpha) < -self.wolfe_c2 * np.dot(
                self.sn, self._gcur
            )

            if not armijo_condition:
                alpha /= factor
            elif not curvature_condition:
                alpha *= factor
            else:
                run = False

            iter += 1
            if iter >= self.max_iter_ls:
                run = False

            self.message(
                f"          Iteration {iter}, alpha = {alpha}, f = {f_alpha}, |g| = {np.linalg.norm(g_alpha)}, armijo = {armijo_condition}, curvature = {curvature_condition}"
            )

        self._x_cur = x
        self._fcur, self._gcur = f_alpha, g_alpha
        return alpha

    def linesearch_parabola(self, initial_alpha=1.0):
        self.message("    Begin parabola linesearch")

        iter = 0

        s = self.sn / np.linalg.norm(self.sn)

        alpha = initial_alpha
        g_dot_s_cur = np.dot(s, self._gcur)

        # For the first iteration, we initialize these values here
        x = self._x_cur + alpha * s
        f_alpha, g_alpha = self.fun_grad_cb(x)
        g_dot_s = np.dot(g_alpha, s)

        while True:
            # Check for convergence
            if np.abs(g_dot_s) < self.tolerance:
                break

            # Fit a parabola from f_cur, f_alpha and g_alpha
            # p(a) = f_cur + g_cur * a + c*alpha**2
            c = (g_dot_s - g_dot_s_cur) / (2.0 * alpha)
            alpha_opt = -g_dot_s_cur / (2.0 * c)
            alpha = alpha_opt

            x = self._x_cur + alpha * s
            f_alpha, g_alpha = self.fun_grad_cb(x)
            g_dot_s = np.dot(g_alpha, s)

            iter += 1
            if iter >= self.max_iter_ls:
                break

            self.message(
                f"          Iteration {iter}, alpha = {alpha}, f = {f_alpha}, |g| = {np.linalg.norm(g_alpha)}"
            )
            self.message(
                f"          ... g_dot_s {g_dot_s}, g_dot_s_cur {g_dot_s_cur}, f_cur = {self._fcur}, f_alpha = {f_alpha}"
            )

        self._x_cur = x
        self._fcur, self._gcur = f_alpha, g_alpha
        return alpha

    def minimize(self, x0: npt.NDArray) -> npt.NDArray:
        self._iter = 0  # reset local iteration count
        run = True

        self._x_cur = x0.copy()
        self._fcur, self._gcur = self.fun_grad_cb(self._x_cur)

        while run:
            self.delta_xn_prev = self.delta_xn  # save delta_xn_previous
            self.delta_xn = -self._gcur  # delta_xn is the direction of steepest descent
            beta = self.beta_pr(self.delta_xn, self.delta_xn_prev)

            self.message(f"    beta = {beta}")

            # Update conjugate direction
            self.sn *= beta
            self.sn += self.delta_xn

            # self.alpha_prev = self.linesearch(self.alpha_prev)
            self.alpha_prev = self.linesearch_parabola(self.alpha_prev)

            self._iter += 1

            if (
                np.linalg.norm(self._gcur) < self.tolerance
                or self._iter >= self.max_iter
            ):
                run = False

            self.message(
                f"Iteration {self._iter} (max. {self.max_iter}), f = {self._fcur}, |g| = {np.linalg.norm(self._gcur)}"
            )
        return [self._x_cur, self._fcur, self._gcur]
