from ridgefollowing import energy_surface
from ridgefollowing.algorithms import modes, ridgefollower

# from ridgefollowing import ridgefollower
from spirit_extras.plotting import Paper_Plot
from typing import Optional
from pydantic import BaseModel
from pathlib import Path
import numpy.typing as npt
import numpy as np


class ScalarPlotSettings(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    contours: bool = False
    contours_filled: bool = True

    colormap: Optional[str] = "plasma"
    color: Optional[str] = None
    contourlevels: int = 20
    vmax: Optional[float] = None
    vmin: Optional[float] = None

    def plot(self, ax, X, Y, Z):
        plot_funcs = []
        if self.contours:
            plot_funcs.append(ax.contour)
        if self.contours_filled:
            plot_funcs.append(ax.contourf)

        return_vals = []
        for f in plot_funcs:
            return_vals.append(
                f(
                    X,
                    Y,
                    Z,
                    extend="neither",
                    levels=self.contourlevels,
                    vmin=self.vmin,
                    vmax=self.vmax,
                    cmap=self.colormap,
                )
            )
        return return_vals


class VectorPlotSettings(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    streamplot: bool = True
    quiver: bool = False
    color: Optional[str] = None

    def plot(self, ax, X, Y, U, V):
        plot_funcs = []

        if self.streamplot:
            plot_funcs.append(ax.streamplot)
        if self.quiver:
            plot_funcs.append(ax.quiver)

        return_vals = []
        for f in plot_funcs:
            return_vals.append(
                f(
                    X,
                    Y,
                    U,
                    V,
                    color=self.color,
                )
            )
        return return_vals


class PlotSettings(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    outfile: Optional[str] = None  # Figure .png file
    output_data_folder: Optional[Path] = None  # folder to save computed data in
    input_data_folder: Optional[Path] = None  # folder to read computed data from

    width: float = 9 * Paper_Plot.cm  # width of the figure
    abs_horizontal_margins: npt.NDArray = np.array(
        [1.25 * Paper_Plot.cm, 0.25 * Paper_Plot.cm]
    )
    abs_vertical_margins: npt.NDArray = np.array(
        [1.25 * Paper_Plot.cm, 0.25 * Paper_Plot.cm]
    )
    dpi: float = 300

    lims: npt.ArrayLike = np.array(
        [[0, 1], [0, 1]]
    )  # limits [[xmin, xmax], [ymin, ymax]
    npoints: npt.ArrayLike = np.array(
        [100, 100], dtype=np.int64
    )  # number of sample points [npointsx, npointsy]

    plot_energy: Optional[ScalarPlotSettings] = None
    plot_c2: Optional[ScalarPlotSettings] = None

    plot_gradient: Optional[VectorPlotSettings] = None
    plot_mode: Optional[VectorPlotSettings] = None
    plot_gradient_c: Optional[VectorPlotSettings] = None


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

    if settings.input_data_folder is None:
        X = np.linspace(settings.lims[0, 0], settings.lims[0, 1], settings.npoints[0])
        Y = np.linspace(settings.lims[1, 0], settings.lims[1, 1], settings.npoints[1])
        energy = np.empty(shape=settings.npoints)
        c2 = np.empty(shape=settings.npoints)
        mode = np.empty(shape=(*settings.npoints, surface.ndim))
        gradient = np.empty(shape=(*settings.npoints, surface.ndim))
        gradient_c = np.empty(shape=(*settings.npoints, surface.ndim))

        Rfollower = ridgefollower.RidgeFollower(surface)

        if not settings.output_data_folder is None:
            # assert settings.output_data_folder.is_dir()
            settings.output_data_folder.mkdir(parents=True, exist_ok=True)

        for xi, x in enumerate(X):
            for yi, y in enumerate(Y):
                i = xi * len(Y) + yi
                n = len(X) * len(Y)
                print(f"Point {i} / {n} ( {i/n*100:.2f} %)", end="\r")

                energy[yi, xi] = surface.energy([x, y])

                if settings.plot_mode:
                    mode[yi, xi] = modes.lowest_mode(surface.hessian([x, y]))[1]

                if settings.plot_gradient:
                    gradient[yi, xi] = surface.gradient([x, y])
                    gradient[yi, xi] = gradient[yi, xi] / np.linalg.norm(
                        gradient[yi, xi]
                    )

                if settings.plot_c2:
                    c2[yi, xi] = Rfollower.C([x, y]) ** 2

                if settings.plot_gradient_c:
                    gradient_c[yi, xi] = Rfollower.fd_grad_C([x, y])
    else:
        assert (
            settings.input_data_folder.is_dir() and settings.input_data_folder.exists()
        )
        X = np.load(settings.input_data_folder / "X.npy")
        Y = np.load(settings.input_data_folder / "Y.npy")
        energy = np.load(settings.input_data_folder / "energy.npy")
        c2 = np.load(settings.input_data_folder / "c2.npy")
        mode = np.load(settings.input_data_folder / "mode.npy")
        gradient = np.load(settings.input_data_folder / "gradient.npy")
        gradient_c = np.load(settings.input_data_folder / "gradient_c.npy")

    if not settings.output_data_folder is None:
        np.save(settings.output_data_folder / "X", X)
        np.save(settings.output_data_folder / "Y", Y)
        np.save(settings.output_data_folder / "energy", energy)
        np.save(settings.output_data_folder / "mode", mode)
        np.save(settings.output_data_folder / "gradient", gradient)
        np.save(settings.output_data_folder / "c2", c2)
        np.save(settings.output_data_folder / "gradient_c", gradient_c)

    energymin = np.min(energy)
    energymax = np.max(energy)

    energy = 0.001 + (energy - energymin) / (energymax - energymin)
    energy = np.log(energy)

    if settings.plot_c2:
        settings.plot_c2.plot(ax, X, Y, c2)

    if settings.plot_energy:
        settings.plot_energy.plot(ax, X, Y, energy)

    if settings.plot_mode:
        settings.plot_mode.plot(ax, X, Y, mode[:, :, 0], mode[:, :, 1])

    if not settings.plot_gradient is None:
        settings.plot_gradient.plot(ax, X, Y, gradient[:, :, 0], gradient[:, :, 1])

    if settings.plot_gradient_c:
        settings.plot_gradient_c.plot(
            ax, X, Y, gradient_c[:, :, 0], gradient_c[:, :, 1]
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if not settings.outfile is None:
        fig.savefig(settings.outfile, dpi=settings.dpi)
