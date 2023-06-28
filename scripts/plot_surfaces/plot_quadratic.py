from ridgefollowing.surfaces import quadratic
from ridgefollowing.plotting import plot_surface
import numpy as np

settings = plot_surface.PlotSettings(
    lims=np.array([[-2, 2], [-2, 2]]),
    plot_energy=plot_surface.ScalarPlotSettings(
        contours_filled=False, contours=True, colors="grey", colormap=None
    ),
    plot_c2=plot_surface.ScalarPlotSettings(),
    output_data_folder="./data_quadratic",
    outfile="plot_quadratic.png",
    npoints=[128, 128],
)

esurf = quadratic.QuadraticSurface(np.diag([1, 2]))
plot_surface.plot(esurf, settings=settings)
