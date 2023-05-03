from ridgefollowing.algorithms import ncg
import numpy as np
import numpy.typing as npt
from numdifftools import Gradient
from typing import Optional
from pydantic import BaseModel


class SphericalOptimization:
    class Result(BaseModel):
        class Config:
            arbitrary_types_allowed = True

        x_opt: npt.NDArray
        g_opt: npt.NDArray
        f_opt: float

        x_opt_stero: npt.NDArray
        g_opt_stero: npt.NDArray

        iterations_total: Optional[int] = None

    def __init__(
        self,
        fun,
        grad,
        ndim: int,
        tolerance: float = 1e-6,
        assert_success: bool = True,
        disp: bool = False,
    ) -> None:
        self.ndim: int = ndim
        self.fun = fun
        self.grad = grad
        self.tolerance: float = tolerance
        self.x_embed = np.zeros(ndim)
        self.x_stereo = np.zeros(ndim - 1)
        self.pole: float = 1.0
        self.disp: bool = disp
        self.assert_success: bool = assert_success

    def embed_to_stereo(self, x_embed):
        """Convert embedding space coordinates to stereographic coordinates"""
        if 1.0 - self.pole * x_embed[-1] < 0.1:
            self.pole *= -1

        self.x_stereo[:] = x_embed[:-1] / (1.0 - self.pole * x_embed[-1])
        return self.x_stereo

    def stereo_to_embed(self, x_stereo):
        """Convert stereographic coordinates to embedding space coordinates"""
        s2 = np.linalg.norm(x_stereo) ** 2
        self.x_embed[-1] = self.pole * (s2 - 1.0) / (s2 + 1.0)
        self.x_embed[:-1] = x_stereo * (1.0 - self.pole * self.x_embed[-1])

        if 1.0 - self.pole * self.x_embed[-1] < 0.1:
            self.pole *= -1

        return self.x_embed

    def f_stereo(self, x_stereo):
        """Compute the function value from stereographic coordinates"""
        self.x_embed = self.stereo_to_embed(x_stereo)
        res = self.fun(self.x_embed)
        return res

    def grad_stero(self, x_stereo):
        """Compute the function gradient from stereographic coordinates"""
        x_embed = self.stereo_to_embed(x_stereo)
        grad_embed = self.grad(x_embed)
        grad_stereo = np.zeros(len(x_stereo))
        s = np.linalg.norm(x_stereo)

        temp = (s**2 + 1.0) ** 2
        grad_stereo[:] = (
            2.0 / (s**2 + 1.0) * grad_embed[:-1]
            - 4.0
            / temp
            * (np.dot(grad_embed[:-1], x_stereo) - self.pole * grad_embed[-1])
            * x_stereo
        )
        return grad_stereo

    def minimize(self, x0):
        res = minimize(
            fun=self.f_stereo,
            method="L-BFGS-B",
            x0=self.embed_to_stereo(x0),
            jac=self.grad_stero,
            options=dict(disp=self.disp),
        )
        if self.assert_success:
            assert res.success
        return self.stereo_to_embed(res.x)
