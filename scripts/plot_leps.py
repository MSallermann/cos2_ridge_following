from ridgefollowing.surfaces import leps
from ridgefollowing.plotting import plot_surface
import numpy as np

esurf = leps.LepsSurface()

settings = plot_surface.PlotSettings(
    lims=np.array([[2, 6], [2, 6]]),
    plot_energy=plot_surface.ScalarPlotSettings(contourlevels=50, colormap="seismic"),
    output_data_folder="./data_leps",
    outfile="plot_leps.png",
)

plot_surface.plot(esurf, settings=settings)
