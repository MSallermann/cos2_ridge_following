from ridgefollowing.algorithms import ncg
from scipy.optimize import minimize
import numpy as np
import numpy.typing as npt
from typing import Optional
from pydantic import BaseModel
from numba import njit


class SphericalOptimization:
    class Result(BaseModel):
        class Config:
            arbitrary_types_allowed = True

        x_opt: npt.NDArray
        g_opt: npt.NDArray
        f_opt: float

        x_opt_stero: npt.NDArray
        g_opt_stero: npt.NDArray

        pole: int

        iterations_total: Optional[int] = None

    def __init__(
        self,
        fun,
        grad,
        ndim: int,
        tolerance: float = 1e-6,
        maxiter: Optional[int] = 1000,
        assert_success: bool = True,
        disp: bool = False,
    ) -> None:
        self.ndim: int = ndim
        self.fun = fun
        self.grad = grad
        self.tolerance: float = tolerance
        self.maxiter = maxiter
        self.pole: float = 1.0
        self.disp: bool = disp
        self.assert_success: bool = assert_success

        self.pole_switch_tolerance: float = 0.5

    def switch_pole_if_necessary(self, x_embed):
        if np.abs(1.0 - self.pole * x_embed[-1]) < self.pole_switch_tolerance:
            self.pole *= -1.0
            return True
        else:
            return False

    @staticmethod
    @njit(cache=True)
    def embed_to_stereo(x_embed, pole):
        """Convert embedding space coordinates to stereographic coordinates"""

        x_stereo = x_embed[:-1] / (1.0 - pole * x_embed[-1])
        return x_stereo

    @staticmethod
    @njit(cache=True)
    def stereo_to_embed(x_stereo, pole):
        """Convert stereographic coordinates to embedding space coordinates"""

        s2 = np.linalg.norm(x_stereo) ** 2
        x_embed = np.empty(len(x_stereo) + 1)
        x_embed[-1] = pole * (s2 - 1.0) / (s2 + 1.0)
        x_embed[:-1] = x_stereo * (1.0 - pole * x_embed[-1])
        return x_embed

    def f_stereo(self, x_stereo):
        """Compute the function value from stereographic coordinates"""
        assert len(x_stereo) == self.ndim - 1

        self.x_embed = SphericalOptimization.stereo_to_embed(x_stereo, self.pole)
        res = self.fun(self.x_embed)
        return res

    @staticmethod
    @njit(cache=True)
    def convert_embed_grad_to_stereo_grad(grad_embed, x_stereo, x_embed, pole):
        grad_stereo = np.zeros(len(x_stereo))
        s = np.linalg.norm(x_stereo)

        grad_stereo = (1.0 - pole * x_embed[-1]) * grad_embed[:-1]
        grad_stereo += (
            pole
            * 4.0
            / (s**2 + 1.0) ** 2
            * x_stereo
            * (grad_embed[-1] - pole * np.dot(grad_embed[:-1], x_stereo))
        )

        return grad_stereo

    def grad_stereo(self, x_stereo):
        """Compute the function gradient from stereographic coordinates"""
        assert len(x_stereo) == self.ndim - 1

        x_embed = SphericalOptimization.stereo_to_embed(x_stereo, self.pole)
        grad_embed = self.grad(x_embed)

        return SphericalOptimization.convert_embed_grad_to_stereo_grad(
            grad_embed=grad_embed, x_stereo=x_stereo, x_embed=x_embed, pole=self.pole
        )

    def switch_pole_cb(self, x_stereo):
        self.switch_pole_if_necessary(
            SphericalOptimization.stereo_to_embed(x_stereo, self.pole)
        )

    def minimize(self, x0) -> Result:
        self.switch_pole_if_necessary(x0)

        res = minimize(
            fun=self.f_stereo,
            method="L-BFGS-B",
            x0=SphericalOptimization.embed_to_stereo(x0, self.pole),
            jac=self.grad_stereo,
            options=dict(disp=self.disp, maxiter=self.maxiter),
            callback=self.switch_pole_cb,
            tol=self.tolerance,
        )

        if self.assert_success:
            if not res.success:
                print(f"gradient = {res.jac}")
                print(
                    f"max component. = {np.max( np.abs(res.jac) )} > {self.tolerance}"
                )
                raise Exception("Optimization unsuccessful")

        return SphericalOptimization.Result(
            x_opt=np.array(SphericalOptimization.stereo_to_embed(res.x, self.pole)),
            g_opt=np.array(SphericalOptimization.stereo_to_embed(res.jac, self.pole)),
            f_opt=res.fun,
            x_opt_stero=res.x,
            g_opt_stero=res.jac,
            pole=self.pole,
        )
