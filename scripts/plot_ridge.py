from ridgefollowing.surfaces import muller_brown
from ridgefollowing.plotting import plot_surface
from ridgefollowing.algorithms import minimizer, ridgefollower
import numpy as np

settings = plot_surface.PlotSettings(
    lims=np.array([[-1.5, 1.5], [-0.5, 2]]),
    npoints=[256, 256],
    plot_energy=plot_surface.ScalarPlotSettings(
        contours_filled=False, contours=True, colormap=None, colors="yellow"
    ),
    plot_c2=plot_surface.ScalarPlotSettings(log_compression=False),
    output_data_folder="./data_ridge",
    # input_data_folder="./data_ridge",
    outfile="plot_ridge.png",
)

esurf = muller_brown.MullerBrownSurface()
ret = plot_surface.plot(esurf, settings=settings)

min = minimizer.Minimizer(energy_surface=esurf)
x_min_1 = min.minimize_energy(np.array([-0.5, 1.5]))
x_min_2 = min.minimize_energy(np.array([0.0, 0.5]))
x_min_3 = min.minimize_energy(np.array([0.5, 0.0]))

print(x_min_1)
print(x_min_2)
print(x_min_3)

follower = ridgefollower.RidgeFollower(esurf)
d = follower.find_maximum_on_ring(x_min_1, np.array([2.0, -2.0]), radius=1e-1)
print(d)
