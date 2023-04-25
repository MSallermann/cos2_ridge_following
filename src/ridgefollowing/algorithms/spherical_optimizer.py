from scipy.optimize import minimize
import numpy as np
from numdifftools import Gradient


class SphericalOptimization:
    def __init__(self, fun, grad, ndim) -> None:
        self.ndim = ndim
        self.fun = fun
        self.grad = grad
        self.x_embed = np.zeros(ndim)
        self.x_stereo = np.zeros(ndim - 1)

    def embed_to_stereo(self, x_embed):
        """Convert embedding space coordinates to stereographic coordinates"""
        self.x_stereo[:] = x_embed[:-1] / (1.0 - x_embed[-1])
        return self.x_stereo

    def stereo_to_embed(self, x_stereo):
        """Convert stereographic coordinates to embedding space coordinates"""
        s2 = np.linalg.norm(x_stereo) ** 2
        self.x_embed[:-1] = 2.0 * x_stereo / (s2 + 1.0)
        self.x_embed[-1] = (s2 - 1.0) / (s2 + 1.0)
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
            * (np.dot(grad_embed[:-1], x_stereo) - grad_embed[-1])
            * x_stereo
        )
        return grad_stereo

    def minimize(self, x0):
        res = minimize(
            fun=self.f_stereo,
            x0=self.embed_to_stereo(x0),
            jac=self.grad_stero,
            options=dict(disp=False),
        )
        return self.stereo_to_embed(res.x)
