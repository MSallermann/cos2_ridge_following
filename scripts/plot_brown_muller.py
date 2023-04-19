from ridgefollowing.surfaces import muller_brown
from ridgefollowing.plotting import plot_surface
import numpy as np

esurf = muller_brown.MullerBrownSurface()

settings = plot_surface.PlotSettings()
settings.lims = np.array([ [-1.5,1.5], [-0.5,2] ])
# settings.plot_mode = True
# settings.plot_gradient = True
settings.plot_c2 = True

settings.outfile = "plot_brown_muller.png"

plot_surface.plot(esurf, settings=settings)