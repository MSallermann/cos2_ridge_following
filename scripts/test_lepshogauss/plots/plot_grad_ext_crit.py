from energy_surfaces.surfaces import lepshogauss
from ridgefollowing.plotting import plot_surface
from pathlib import Path
import numpy as np
from ridgefollowing.algorithms import minimizer

esurf = lepshogauss.LepsHOGaussSurface()

lims = np.array([[0.25, 3.5], [-5, 5]])
folder = Path("./data400")

settings = plot_surface.PlotSettings(
    width=15 * plot_surface.cm,
    outfile="plot_grad_ext_crit.png",
    lims=lims,
    plot_energy=plot_surface.ScalarPlotSettings(
        contourlevels=40,
        contours_filled=False,
        contours=True,
        colors="grey",
        colormap=None,
        log_compression=False,
        zorder=9,
    ),
    plot_grad_ext_crit=plot_surface.ScalarPlotSettings(
        contourlevels=30,
        log_compression=False,
        colormap="coolwarm",
        contours_filled=True,
        flip_sign=False,
    ),
    output_data_folder=folder,
    input_data_folder=folder,
    npoints=np.array([400, 400]),
)

min = minimizer.Minimizer(energy_surface=esurf, tolerance=1e-7)
x_min_1 = min.minimize_energy(np.array([0.5, 4.0]))
x_min_2 = min.minimize_energy(np.array([1.0, -0.5]))
x_min_3 = min.minimize_energy(np.array([3, -2.0]))

for xm in [x_min_1, x_min_2, x_min_3]:
    settings.path_plots.append(
        plot_surface.PathPlotSettings(points=np.array([xm]), marker="o", color="black")
    )

np.set_printoptions(precision=16)
print(x_min_1, esurf.energy(x_min_1), esurf.gradient(x_min_1))
print(x_min_2, esurf.energy(x_min_2), esurf.gradient(x_min_2))
print(x_min_3, esurf.energy(x_min_3), esurf.gradient(x_min_3))

plot_surface.plot(esurf, settings=settings)
