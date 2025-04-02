from ridgefollowing.algorithms import ncg
from scipy.optimize import minimize
import numpy as np


def test_ncg_quadratic():
    ndim = 12
    coeffs = np.linspace(2.0, 10.0, ndim)
    coeffs[5] = 1.0

    def fun(x):
        return np.dot(coeffs, x**2)

    def jac(x):
        return 2.0 * coeffs * x

    def fun_grad_cb(x):
        return fun(x), jac(x)

    ncg_opt = ncg.NonLinearConjugateGradient(
        fun_grad_cb=fun_grad_cb, ndim=ndim, maxiter=200, tolerance=1e-6, disp=False
    )

    x0 = np.ones(ndim)
    res = ncg_opt.minimize(x0)

    assert np.isclose(res.f_opt, 0.0, atol=ncg_opt.tolerance)
    assert np.allclose(res.x_opt, np.zeros(ndim), atol=ncg_opt.tolerance)
    assert np.allclose(res.g_opt, np.zeros(ndim), atol=ncg_opt.tolerance)
    assert res.iterations_total <= 12


def test_ncg():
    ndim = 28
    coeffs = np.linspace(2.0, 10.0, ndim)
    coeffs[5] = 1.0

    def fun(x):
        return np.dot(coeffs, x**2) + 1 * (x[0] - 1) ** 4 + 0.5 * x[0] ** 3

    def jac(x):
        return 2.0 * coeffs * x + 1 * 4 * (x[0] - 1) ** 3 + 3 * 0.5 * x[0] ** 2

    def fun_grad_cb(x):
        return fun(x), jac(x)

    ncg_opt = ncg.NonLinearConjugateGradient(
        fun_grad_cb=fun_grad_cb, ndim=ndim, maxiter=200, tolerance=1e-6, disp=False
    )

    x0 = np.ones(ndim)
    res = ncg_opt.minimize(x0)

    assert np.allclose(res.g_opt, np.zeros(ndim), atol=ncg_opt.tolerance)
