from ridgefollowing.surfaces import lepshogauss, quadratic
from ridgefollowing.algorithms import minimizer, ridgefollower
import numpy as np
import matplotlib.pyplot as plt

points = [
            np.array([1.006, -4.31109]),
            # np.array([ 1.15154047, -0.99335039]),
            # np.array([ 1.15154047, 0])
            # np.array([1, 1]),
            # np.array([1.0779186977342878, -3.415465753253382])
          ]

esurf = lepshogauss.LepsHOGaussSurface()
# esurf = quadratic.QuadraticSurface( np.diag([1, 2]) )

follower = ridgefollower.RidgeFollower(esurf, radius=5e-1, tolerance=1e-12)
follower.width_modified_gaussian = 0.1
follower.magnitude_modified_gaussian = 0.0

for p in points:
    for r in [1e-1]:
        follower.radius = r
        phi, c2, gradc2, dirs = follower.sample_on_ring(p, npoints=64)
        plt.plot(phi, c2, color="C1", label="Full")

        phi, c2, gradc2, dirs = follower.sample_on_ring(p, npoints=64, anharmonic=True)
        plt.plot(phi, c2, color="C2", ls="--", label="Taylor")

        # gradc2_np = np.gradient(c2, phi[1] - phi[0])[0]
        # maxima = follower.find_all_maxima_on_ring(p, npoints=1)

        # # phi, c2_mod, gradc2_mod, dirs = follower.sample_on_ring( p, npoints=n_points, use_mod=True )

        # axins = plt.gca().twinx()
        # axins.plot(phi, gradc2, color="C1")

        # # print(gradc2_np/100)
        # # axins.plot(phi, gradc2_np*20, color="C2")

        # print(maxima)
        # phi_ini = maxima[:, 3]
        # phi_max = maxima[:, 4]
        # c2_max = maxima[:, 5]

        # plt.axhline(0)
        # plt.plot(phi_max, c2_max, ls="None", marker="o")
        # plt.plot(phi_ini, c2_max, ls="None", marker="o")

        # # plt.plot(phi, c2_mod, color="C1")
    # plt.yscale("log")
    plt.legend()
    plt.show()
