from ridgefollowing.algorithms import spherical_optimizer
import numpy as np
from numdifftools import Gradient


def test_spherical_optimizer():
    ndim = 12
    coeffs = np.linspace(10.0, 2.0, ndim)

    def fun(x):
        return np.dot(coeffs, x**2)

    def jac(x):
        return 2.0 * coeffs * x

    soptimizer = spherical_optimizer.SphericalOptimization(fun, jac, ndim=ndim)

    for i in range(10):
        # x_initial = np.array(np.random.random(ndim))
        x_initial = np.array(np.ones(ndim))

        x_initial /= np.linalg.norm(x_initial)

        x_stereo = soptimizer.embed_to_stereo(x_initial)
        f_embed = fun(x_initial)
        f_stereo = soptimizer.f_stereo(x_stereo)

        assert np.isclose(f_embed, f_stereo)

        grad_stereo = soptimizer.grad_stero(x_stereo)
        fd_grad_stereo = Gradient(soptimizer.f_stereo)(x_stereo)

        assert np.allclose(grad_stereo, fd_grad_stereo)

        x_stereo += np.array(np.random.random(ndim - 1))
        x_embed = soptimizer.stereo_to_embed(x_stereo)

        assert np.isclose(np.linalg.norm(x_embed), 1.0)

    x_initial = np.ones(ndim)
    x_initial /= np.linalg.norm(x_initial)

    x_opt = soptimizer.minimize(x_initial)
    x_opt_expected = np.zeros(ndim)
    x_opt_expected[0] = 1.0

    # assert np.allclose(x_opt, x_opt_expected, atol=1e-6)


test_spherical_optimizer()
