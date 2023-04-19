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
    abs_horizontal_margins: npt.NDArray = np.array(
        [1.25 * Paper_Plot.cm, 0.25 * Paper_Plot.cm]
    )
    abs_vertical_margins: npt.NDArray = np.array(
        [1.25 * Paper_Plot.cm, 0.25 * Paper_Plot.cm]
    )
    vmax: Optional[float]
    vmin: Optional[float]
    contourlevels: float = 50
    colormap = "plasma"
    dpi: float = 300


def plot(surface: energy_surface.EnergySurface, ax=None, settings=PlotSettings()):
    if ax is None:
        pplot = Paper_Plot(width=settings.width)
        pplot.apply_absolute_margins(
            aspect_ratio=1,
            abs_horizontal_margins=settings.abs_horizontal_margins,
            abs_vertical_margins=settings.abs_vertical_margins,
        )
        fig = pplot.fig()
        gs = pplot.gs()
        ax = fig.add_subplot(gs[0])

    assert surface.ndim == 2

    X = np.linspace(settings.lims[0, 0], settings.lims[0, 1], settings.npoints[0])
    Y = np.linspace(settings.lims[1, 0], settings.lims[1, 1], settings.npoints[1])
    Z = np.zeros(shape=settings.npoints)

    for xi, x in enumerate(X):
        for yi, y in enumerate(Y):
            Z[yi, xi] = surface.energy([x, y])

    Zmin = np.min(Z)
    Zmax = np.max(Z)

    Z = 0.001 + (Z - Zmin) / (Zmax - Zmin)
    Z = np.log(Z)

    contours = ax.contourf(
        X,
        Y,
        Z,
        extend="neither",
        levels=settings.contourlevels,
        vmin=settings.vmin,
        vmax=settings.vmax,
        cmap=settings.colormap,
    )
    fig.colorbar(contours)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if not settings.outfile is None:
        fig.savefig(settings.outfile, dpi=settings.dpi)
