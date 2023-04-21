from ridgefollowing.surfaces import muller_brown
from ridgefollowing.plotting import plot_surface
import numpy as np

esurf = muller_brown.MullerBrownSurface()

settings = plot_surface.PlotSettings(
    lims=np.array([[-1.5, 1.5], [-0.5, 2]]),
    plot_energy=plot_surface.ScalarPlotSettings(
        contours_filled=False,
        contours=True,
        contourlevels=30,
    ),
    plot_c2=plot_surface.ScalarPlotSettings(),
    plot_gradient_c=plot_surface.VectorPlotSettings(),
    output_data_folder="./data_brown_muller",
    input_data_folder="./data_brown_muller",
    outfile="plot_brown_muller.png",
)

plot_surface.plot(esurf, settings=settings)
