from ridgefollowing import energy_surface
from spirit_extras.plotting import Paper_Plot
from typing import Optional
from pydantic import BaseModel, Field
import numpy.typing as npt
import numpy as np


class PlotSettings(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    lims: npt.ArrayLike = np.array([[0, 1], [0, 1]])
    width: float = 9 * Paper_Plot.cm
    outfile: Optional[str] = None
    npoints: npt.ArrayLike = np.array([256, 256], dtype=np.int64)


def plot(surface: energy_surface.EnergySurface, ax=None, settings=PlotSettings()):
    if ax is None:
        pplot = Paper_Plot(width=settings.width)
        pplot.apply_absolute_margins(aspect_ratio=1)
        fig = pplot.fig()
        gs = pplot.gs()
        ax = fig.add_subplot(gs[0])

    assert surface.ndim == 2

    X = np.linspace(settings.lims[0, 0], settings.lims[0, 1], settings.npoints[0])
    Y = np.linspace(settings.lims[1, 0], settings.lims[1, 1], settings.npoints[1])

    Z = np.zeros(shape=settings.npoints)

    ax.contourf()
