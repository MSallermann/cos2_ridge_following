from ridgefollowing.surfaces import leps, lepshogauss
from ridgefollowing.plotting import plot_surface
import numpy as np

esurf = lepshogauss.LepsHOGaussSurface()

settings = plot_surface.PlotSettings(
    lims=np.array([[0.25, 3.5], [-5, 5]]),
    plot_energy=plot_surface.ScalarPlotSettings(),
    output_data_folder="./data_lepshogauss",
    outfile="plot_lepshogauss.png",
    npoints=[128, 128],
)

plot_surface.plot(esurf, settings=settings)
