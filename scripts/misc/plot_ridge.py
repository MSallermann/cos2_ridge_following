from energy_surfaces.surfaces import muller_brown, quadratic
from ridgefollowing.plotting import plot_surface
from ridgefollowing.algorithms import minimizer, ridgefollower
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib

gui_env = ["Qt5Agg"]
for gui in gui_env:
    try:
        print("testing", gui)
        matplotlib.use(gui, force=True)
        from matplotlib import pyplot as plt

        break
    except Exception as e:
        print(e)
        continue
print("Using:", matplotlib.get_backend())

esurf = muller_brown.MullerBrownSurface()
lims = np.array([[-0.5, 1.0], [-0.5, 0.6]])
lims = np.array([[-1.5, 1.5], [-0.5, 2]])

# esurf = quadratic.QuadraticSurface( np.diag( [1,1] ) )
# lims = np.array([[-2,2],[-2,2]])

settings = plot_surface.PlotSettings(
    lims=lims,
    npoints=[64, 64],
    plot_energy=plot_surface.ScalarPlotSettings(
        contours_filled=False,
        contours=True,
        colors="white",
        colormap=None,
        contourlevels=15,
    ),
    # plot_c_grad_norm=plot_surface.ScalarPlotSettings(log_compression=True),
    plot_c2=plot_surface.ScalarPlotSettings(log_compression=False, colormap="cividis"),
    plot_gradient_c2=plot_surface.VectorPlotSettings(
        color="grey", streamplot=True, quiver=False
    ),
    output_data_folder="./data_ridge2",
    # input_data_folder="./data_ridge2",
    outfile="plot_ridge.png",
)

min = minimizer.Minimizer(energy_surface=esurf)
x_min_1 = min.minimize_energy(np.array([-0.5, 1.5]))
x_min_2 = min.minimize_energy(np.array([0.5, 0.0]))

follower = ridgefollower.RidgeFollower(esurf, radius=4e-2, n_iterations_follow=40)

follower.follow(x_min_2, [-1.0, 1.0])

settings.path_plots.append(
    plot_surface.PathPlotSettings(points=follower.history["x_cur"], color="C1")
)

# settings.path_plots.append(
#     plot_surface.PathPlotSettings(points=np.array(follower.history["bifurcation_points"]), color="grey", ls="None", marker="o")
# )

settings.show = True

ret = plot_surface.plot(esurf, settings=settings)

E = follower.history["E"]
plt.plot(E)
plt.ylabel("E")
plt.show()
plt.close()

c2 = follower.history["c2"]
plt.plot(c2)
plt.ylabel("c2")
plt.show()

plt.close()
d = follower.history["d_cur"]
grad_c2 = follower.history["grad_c2"]
plt.plot([np.dot(_d, _g) for _d, _g in zip(d, grad_c2)])
plt.ylabel("dot")
plt.show()
