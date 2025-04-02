from energy_surfaces.surfaces import lepsho
from ridgefollowing.plotting import plot_surface
import numpy as np

esurf = lepsho.LepsHOSurface()

settings = plot_surface.PlotSettings(
    lims=np.array([[0.5, 3.5], [-4, 4]]),
    plot_energy=plot_surface.ScalarPlotSettings(contourlevels=50, colormap="seismic"),
    output_data_folder="./data_lepsho",
    outfile="plot_lepsho.png",
)

plot_surface.plot(esurf, settings=settings)
