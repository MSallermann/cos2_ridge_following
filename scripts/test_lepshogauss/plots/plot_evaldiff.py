from energy_surfaces.surfaces import lepshogauss
from ridgefollowing.plotting import plot_surface
from pathlib import Path
import numpy as np
from ridgefollowing.algorithms import minimizer

esurf = lepshogauss.LepsHOGaussSurface()

lims = np.array([[0.25, 3.5], [-5, 5]])
folder = Path("./data200")

settings = plot_surface.PlotSettings(
    width=15 * plot_surface.cm,
    outfile="plot_evaldiff.png",
    lims=lims,
    plot_energy=plot_surface.ScalarPlotSettings(
        contourlevels=40,
        contours_filled=False,
        contours=True,
        colors="grey",
        colormap=None,
        log_compression=False,
        zorder=999,
    ),
    plot_evaldiff=plot_surface.ScalarPlotSettings(
        contourlevels=30,
        log_compression=True,
        colormap="coolwarm",
        contours_filled=True,
    ),
    output_data_folder=folder,
    input_data_folder=folder,
    npoints=np.array([200, 200]),
)

evalues = np.load(folder / "eigenvalues.npy")
print(np.argmin(np.abs(evalues[:, :, 1] - evalues[:, :, 0])))

plot_surface.plot(esurf, settings=settings)
