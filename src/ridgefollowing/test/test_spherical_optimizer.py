from ridgefollowing.algorithms import spherical_optimizer, cosine_follower
from ridgefollowing.surfaces import gaussians, lepshogauss

import numpy as np
from numdifftools import Gradient


def test_spherical_optimizer():
    ndim = 12
    coeffs = np.linspace(2.0, 10.0, ndim)
    coeffs[5] = 1.0

    def fun(x):
        return np.dot(coeffs, x**2)

    def jac(x):
        return 2.0 * coeffs * x

    soptimizer = spherical_optimizer.SphericalOptimization(
        fun, jac, ndim=ndim, assert_success=True, tolerance=1e-24
    )

    for i in range(10):
        x_initial = np.array(np.random.random(ndim))

        x_initial /= np.linalg.norm(x_initial)

        x_stereo = spherical_optimizer.SphericalOptimization.embed_to_stereo(
            x_initial, soptimizer.pole
        )
        f_embed = fun(x_initial)
        f_stereo = soptimizer.f_stereo(x_stereo)

        assert np.isclose(f_embed, f_stereo)

        grad_stereo = soptimizer.grad_stereo(x_stereo)
        fd_grad_stereo = Gradient(soptimizer.f_stereo)(x_stereo)
        assert np.allclose(grad_stereo, fd_grad_stereo)

        x_stereo += np.array(np.random.random(ndim - 1))
        x_embed = spherical_optimizer.SphericalOptimization.stereo_to_embed(
            x_stereo, soptimizer.pole
        )

        assert np.isclose(np.linalg.norm(x_embed), 1.0)

    x_initial = np.ones(ndim)
    x_initial /= np.linalg.norm(x_initial)

    res = soptimizer.minimize(x_initial)
    x_opt_expected = np.zeros(ndim)
    x_opt_expected[5] = 1.0

    print(x_opt_expected)
    print(res.x_opt)
    assert np.allclose(res.x_opt, x_opt_expected)


def test_pole():
    ndim = 2
    coeffs = np.array([1.0, -1.0])

    def fun(x):
        return np.dot(coeffs, x**2)

    def jac(x):
        return 2.0 * coeffs * x

    x_initial = np.array([1.0, 1.0])
    x_initial /= np.linalg.norm(x_initial)

    soptimizer = spherical_optimizer.SphericalOptimization(
        fun, jac, ndim=ndim, assert_success=True, disp=False, tolerance=1e-12
    )

    soptimizer.pole = -1
    x_stereo = spherical_optimizer.SphericalOptimization.embed_to_stereo(
        x_initial, soptimizer.pole
    )
    f_embed = fun(x_initial)
    f_stereo = soptimizer.f_stereo(x_stereo)
    assert np.isclose(f_embed, f_stereo)

    grad_stereo = soptimizer.grad_stereo(x_stereo)
    fd_grad_stereo = Gradient(soptimizer.f_stereo)(x_stereo)
    assert np.allclose(grad_stereo, fd_grad_stereo)

    x_stereo += np.array(np.random.random(ndim - 1))
    x_embed = spherical_optimizer.SphericalOptimization.stereo_to_embed(
        x_stereo, soptimizer.pole
    )
    assert np.isclose(np.linalg.norm(x_embed), 1.0)

    soptimizer.pole = 1
    res = soptimizer.minimize(x_initial)
    x_opt_expected = np.zeros(ndim)
    x_opt_expected[-1] = 1.0

    print(res.x_opt)
    print(x_opt_expected)
    assert np.allclose(np.abs(res.x_opt), x_opt_expected)
    assert soptimizer.pole == -1  # Pole should have automatically switched


def test_ring_search():
    centers = np.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])

    widths = np.ones(5) / 2

    magnitudes = np.ones(5)
    magnitudes[0] = -1.0

    matrices = np.array([np.diag([1.0, 1.0]) for i in range(5)])

    esurf = gaussians.GaussianSurface(
        magnitudes=magnitudes, matrices=matrices, widths=widths, centers=centers, ndim=2
    )

    def grad(x):
        g = esurf.gradient(x)
        return g - np.dot(g, x) * x

    sopt = spherical_optimizer.SphericalOptimization(
        esurf.energy,
        esurf.gradient,
        ndim=esurf.ndim,
        disp=False,
        maxiter=200,
        assert_success=True,
        tolerance=1e-12,
    )
    sopt.pole = -1

    directions_opt_expected = np.array(
        [[1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]]
    )

    Nphi = len(directions_opt_expected)

    energies = np.zeros(Nphi)
    energies_opt = np.zeros(Nphi)
    directions = np.zeros((Nphi, 2))
    directions_opt = np.zeros((Nphi, 2))

    x_stereo_ini = np.zeros(Nphi)
    x_stereo_opt = np.zeros(Nphi)
    pole_opt = np.zeros(Nphi)

    for ip, p in enumerate(directions_opt_expected):
        d = p + 0.01 * np.random.random(2)
        d /= np.linalg.norm(d)

        # sopt.switch_pole_if_necessary(d)
        x_stereo_ini[ip] = spherical_optimizer.SphericalOptimization.embed_to_stereo(
            d, sopt.pole
        )

        # Find expected direction with maximum overlap with initial direction
        overlaps = [
            np.dot(d, d_exp / np.linalg.norm(d_exp))
            for d_exp in directions_opt_expected
        ]
        idx_exp = np.argmax(overlaps)
        d_opt_expected = directions_opt_expected[idx_exp]

        directions[ip] = d

        res = sopt.minimize(d)

        directions_opt[ip] = res.x_opt
        energies[ip] = esurf.energy(d)
        energies_opt[ip] = esurf.energy(res.x_opt)
        x_stereo_opt[ip] = res.x_opt_stero
        pole_opt[ip] = res.pole

        print("d ", d)
        print("d_opt_expected ", d_opt_expected / np.linalg.norm(d_opt_expected))
        print("res ", res.x_opt)
        print("----")

        assert np.allclose(d_opt_expected / np.linalg.norm(d_opt_expected), res.x_opt)

    # import matplotlib.pyplot as plt
    # x_stereo = np.linspace(-10,10,1000)
    # f_stereo = [sopt.f_stereo(x) for x in x_stereo]
    # g_stereo = [sopt.grad_stereo(np.array([x])) for x in x_stereo]
    # plt.plot(x_stereo, f_stereo)
    # plt.plot(x_stereo, g_stereo)
    # plt.axhline(0,color="black")
    # plt.plot(x_stereo_ini, [sopt.f_stereo(x) for x in x_stereo_ini], marker="o", ls="None")
    # plt.plot(x_stereo_opt, [sopt.f_stereo(x) for x in x_stereo_opt], marker="x", ls="None")
    # plt.show()


def test_lepshogauss_regression():
    esurf = lepshogauss.LepsHOGaussSurface()
    follower = cosine_follower.CosineFollower(
        energy_surface=esurf, tolerance=1e-8, radius=1e-2
    )

    # x0 = [0.67595889, 4.2717716]
    x0 = [0.6759588922580516, 4.271771599520583]
    d0 = [-1.0, 0.0]

    x0 = [0.67550265, 4.13177255]
    d0 = [-0.00414767, -0.9999914]

    d0 /= np.linalg.norm(d0)

    prefactor = -1.0 if follower.maximize else 1.0

    def fun(d):
        """-C(x0 + radius*d)**2. Minus sign because we use scipy.minimize"""
        return prefactor * follower.C2_mod(x0 + follower.radius * d)

    # grad = Gradient(fun)

    def grad(d):
        """gradient of C(x0 + radius * d) wrt to d. Minus sign because we use scipy.minimize"""
        x = x0 + follower.radius * d
        grad_c2_d = follower.grad_C2_mod(x) * follower.radius
        # project out component along d0
        grad_c2_d -= np.dot(grad_c2_d, d) * d
        return prefactor * grad_c2_d

    opt = spherical_optimizer.SphericalOptimization(
        fun=fun,
        grad=grad,
        ndim=follower.esurf.ndim,
        tolerance=follower.tolerance * follower.radius,
        maxiter=10000,
        assert_success=True,
        disp=False,
    )

    res = opt.minimize(d0)

    follower._x_cur = x0
    follower._iteration = 0
    follower._step_cur = d0
    follower._d_cur = res.x_opt

    assert np.linalg.norm(res.g_opt_stero) < 1e-10


if __name__ == "__main__":
    # test_spherical_optimizer()
    # test_pole()
    # test_ring_search()
    test_lepshogauss_regression()
