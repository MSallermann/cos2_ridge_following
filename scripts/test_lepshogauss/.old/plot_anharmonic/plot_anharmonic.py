from energy_surfaces.surfaces import lepshogauss
from ridgefollowing.algorithms import cosine_follower
import numpy as np
import matplotlib.pyplot as plt

points = [
    np.array([1.006, -4.31109]),
    np.array([1.15154047, -0.99335039]),
    np.array([1.15154047, 0]),
    np.array([1, 1]),
    np.array([1.0779186977342878, -3.415465753253382]),
]

esurf = lepshogauss.LepsHOGaussSurface()
# esurf = quadratic.QuadraticSurface(np.diag([1, 2]))

follower = cosine_follower.CosineFollower(esurf, radius=5e-1, tolerance=1e-12)
follower.width_modified_gaussian = 0.1
follower.magnitude_modified_gaussian = 0.0


def taylor(phi, lambda_1, lambda_2, g0, r):
    G2 = g0[0] ** 2 + g0[1] ** 2
    C02 = g0[0] ** 2 / G2
    return C02 + 2 * r / G2 * (
        (1.0 - C02) * lambda_1 * g0[0] * np.cos(phi)
        - C02 * lambda_2 * np.sin(phi) * g0[1]
    )


for p in points:
    for ir, r in enumerate([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]):
        follower.radius = r
        phi, c2, gradc2, dirs = follower.sample_on_ring(p, npoints=64)
        color = f"C{ir}"

        plt.plot(phi, c2, color=color, ls="-", label=f"Full, r={r:.1e}")

        phi, c2, gradc2, dirs = follower.sample_on_ring(p, npoints=64, anharmonic=True)
        plt.plot(phi, c2, color=color, marker=".", ls="--", label=f"Taylor, r={r:.1e}")

        g0 = esurf.gradient(p)
        print(g0)
        # plt.plot(phi, taylor(phi, 1.0, 2.0, g0, r), color="C3", ls="--", label="Taylor_pred")

        # plt.plot(
        #     phi,
        #     c2 - taylor(phi, 1.0, 2.0, g0, r),
        #     color="C4",
        #     ls="--",
        #     label="2nd order",
        # )

        plt.legend()
        plt.ylabel("$C^2$")
        plt.xlabel("$\phi$")
        plt.savefig(f"plot_x_{p[0]:.2f}_y_{p[1]:.2f}_rad_{r:.2e}.png", dpi=300)
        # plt.savefig(f"plot_x_{p[0]:.2f}_y_{p[1]:.2f}.png", dpi=300)

        plt.close()
        # plt.show()
