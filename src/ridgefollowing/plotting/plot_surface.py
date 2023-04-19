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
    npoints: npt.ArrayLike = np.array([100, 100], dtype=np.int64)
    abs_horizontal_margins: npt.NDArray = np.array(
        [1.25 * Paper_Plot.cm, 0.25 * Paper_Plot.cm]
    )
    abs_vertical_margins: npt.NDArray = np.array(
        [1.25 * Paper_Plot.cm, 0.25 * Paper_Plot.cm]
    )
    vmax: Optional[float]
    vmin: Optional[float]
    contourlevels: float = 20
    colormap = "plasma"
    plot_c2 : bool = False
    dpi: float = 300
    plot_gradient : bool = False
    plot_mode : bool = False

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
    energy = np.empty(shape=settings.npoints)
    c2 = np.empty(shape=settings.npoints)
    mode = np.empty(shape=(*settings.npoints, surface.ndim))
    gradient = np.empty(shape=(*settings.npoints, surface.ndim))

    for xi, x in enumerate(X):
        for yi, y in enumerate(Y):
            energy[yi, xi] = surface.energy([x, y])

            evals, evecs = np.linalg.eig( surface.hessian([x,y]) )
            eval_order = np.argsort(evals)
            mode[yi, xi] = evecs[eval_order[0]]

            gradient[yi, xi] = surface.gradient([x,y])
            gradient[yi, xi] = gradient[yi, xi] / np.linalg.norm(gradient[yi, xi])

            c2[yi, xi] = np.dot(mode[yi,xi], gradient[yi, xi])**2

    energymin = np.min(energy)
    energymax = np.max(energy)

    energy = 0.001 + (energy - energymin) / (energymax - energymin)
    energy = np.log(energy)

    if settings.plot_c2:
        contours = ax.contourf(
            X,
            Y,
            c2,
            extend="neither",
            levels=settings.contourlevels,
            vmin=settings.vmin,
            vmax=settings.vmax,
            cmap=settings.colormap,
        )

    contours = ax.contour(
        X,
        Y,
        energy,
        extend="neither",
        levels=settings.contourlevels,
        vmin=settings.vmin,
        vmax=settings.vmax,
        colors="white",
    )

    if settings.plot_mode:
        ax.streamplot(X, Y, mode[:,:,0], mode[:,:,1])

    if settings.plot_mode:
        ax.streamplot(X, Y, gradient[:,:,0], gradient[:,:,1])

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if not settings.outfile is None:
        fig.savefig(settings.outfile, dpi=settings.dpi)
