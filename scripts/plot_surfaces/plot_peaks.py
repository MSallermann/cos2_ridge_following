from energy_surfaces.surfaces import peaks
from ridgefollowing.plotting import plot_surface
import numpy as np

esurf = peaks.PeaksSurface()

settings = plot_surface.PlotSettings(
    lims=np.array([[-3, 3], [-3, 3]]),
    plot_energy=plot_surface.ScalarPlotSettings(contourlevels=50, colormap="seismic"),
    output_data_folder="./data_peaks",
    outfile="plot_peaks.png",
)

plot_surface.plot(esurf, settings=settings)
