from ridgefollowing.surfaces import leps, lepshogauss
from ridgefollowing.plotting import plot_surface
import numpy as np

esurf = lepshogauss.LepsHOGaussSurface()

settings = plot_surface.PlotSettings(
    lims=np.array([[0, 4], [-2, 2]]),
    plot_energy=plot_surface.ScalarPlotSettings(
        contourlevels=500,
        colormap="seismic",
        log_compression=False,
        contours_filled=False,
        contours=True,
    ),
    output_data_folder="./data_lepshogauss",
    outfile="plot_lepshogauss.png",
    npoints=[256, 256],
)

plot_surface.plot(esurf, settings=settings)
