from ridgefollowing.surfaces import leps, lepsho
from ridgefollowing.plotting import plot_surface
import numpy as np

esurf = lepsho.LepsHOSurface()

settings = plot_surface.PlotSettings(
    lims=np.array([[-1, 4], [-3, 3]]),
    plot_energy=plot_surface.ScalarPlotSettings(contourlevels=50, colormap="seismic"),
    output_data_folder="./data_lepsho",
    outfile="plot_lepsho.png",
)

plot_surface.plot(esurf, settings=settings)
