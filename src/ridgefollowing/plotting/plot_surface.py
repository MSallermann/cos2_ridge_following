from ridgefollowing import energy_surface
from ridgefollowing.algorithms import modes, ridgefollower

# from ridgefollowing import ridgefollower
from spirit_extras.plotting import Paper_Plot
from typing import Optional, List
from pydantic import BaseModel
from pathlib import Path
from dataclasses import dataclass
import numpy.typing as npt
import numpy as np
import matplotlib.pyplot as plt


class PathPlotSettings(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    points: npt.NDArray
    ls: str = "-"
    color: str = "black"
    marker: str = "None"

    def plot(self, ax):
        ax.plot(
            self.points[:, 0],
            self.points[:, 1],
            ls=self.ls,
            color=self.color,
            marker=self.marker,
        )


class ScalarPlotSettings(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    contours: bool = False
    contours_filled: bool = True

    log_compression: bool = True

    colormap: Optional[str] = "coolwarm"
    linestyles: Optional[str] = None
    colors: Optional[str] = None
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
            if self.log_compression:
                Zmin = np.min(Z)
                Zmax = np.max(Z)
                Z = 0.001 + (Z - Zmin) / (Zmax - Zmin)
                Z = np.log(Z)

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
                    colors=self.colors,
                    linestyles=self.linestyles,
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


cm = Paper_Plot.cm


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

    xlim: Optional[npt.NDArray] = None
    ylim: Optional[npt.NDArray] = None

    path_plots: List[PathPlotSettings] = []

    plot_energy: Optional[ScalarPlotSettings] = None
    plot_c2: Optional[ScalarPlotSettings] = None
    plot_evaldiff: Optional[ScalarPlotSettings] = None
    plot_c_grad_norm: Optional[ScalarPlotSettings] = None

    plot_gradient: Optional[VectorPlotSettings] = None
    plot_mode: Optional[VectorPlotSettings] = None
    plot_gradient_c: Optional[VectorPlotSettings] = None
    plot_gradient_c2: Optional[VectorPlotSettings] = None

    show: bool = False


def plot(surface: energy_surface.EnergySurface, ax=None, settings=PlotSettings()):
    @dataclass
    class Return_Value:
        pplot = None
        ax = None

    return_value = Return_Value()
    return_value.ax = ax

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
        return_value.ax = ax
        return_value.pplot = pplot

    assert surface.ndim == 2

    if settings.input_data_folder is None:
        X = np.linspace(settings.lims[0, 0], settings.lims[0, 1], settings.npoints[0])
        Y = np.linspace(settings.lims[1, 0], settings.lims[1, 1], settings.npoints[1])
        energy = np.empty(shape=settings.npoints)
        c = np.empty(shape=settings.npoints)
        mode = np.empty(shape=(*settings.npoints, surface.ndim))
        gradient = np.empty(shape=(*settings.npoints, surface.ndim))
        gradient_c = np.empty(shape=(*settings.npoints, surface.ndim))
        eigenvalues = np.empty(shape=(*settings.npoints, surface.ndim))

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

                if settings.plot_mode or settings.plot_evaldiff:
                    hessian = surface.hessian([x, y])
                    eigenvals, eigenvecs = modes.eigenpairs(hessian)

                    mode[yi, xi] = eigenvecs[0]
                    eigenvalues[yi, xi] = eigenvals

                if settings.plot_gradient:
                    gradient[yi, xi] = surface.gradient([x, y])
                    gradient[yi, xi] = gradient[yi, xi] / np.linalg.norm(
                        gradient[yi, xi]
                    )

                if settings.plot_c2 or settings.plot_gradient_c2:
                    c[yi, xi] = Rfollower.C([x, y])

                if (
                    settings.plot_gradient_c
                    or settings.plot_c_grad_norm
                    or settings.plot_gradient_c2
                ):
                    gradient_c[yi, xi] = Rfollower.fd_grad_C([x, y])
    else:
        assert (
            settings.input_data_folder.is_dir() and settings.input_data_folder.exists()
        )
        X = np.load(settings.input_data_folder / "X.npy")
        Y = np.load(settings.input_data_folder / "Y.npy")
        energy = np.load(settings.input_data_folder / "energy.npy")
        c = np.load(settings.input_data_folder / "c.npy")
        mode = np.load(settings.input_data_folder / "mode.npy")
        gradient = np.load(settings.input_data_folder / "gradient.npy")
        gradient_c = np.load(settings.input_data_folder / "gradient_c.npy")
        eigenvalues = np.load(settings.input_data_folder / "eigenvalues.npy")

    if not settings.output_data_folder is None:
        np.save(settings.output_data_folder / "X", X)
        np.save(settings.output_data_folder / "Y", Y)
        np.save(settings.output_data_folder / "energy", energy)
        np.save(settings.output_data_folder / "mode", mode)
        np.save(settings.output_data_folder / "gradient", gradient)
        np.save(settings.output_data_folder / "c", c)
        np.save(settings.output_data_folder / "gradient_c", gradient_c)
        np.save(settings.output_data_folder / "eigenvalues", eigenvalues)

    if settings.plot_c2:
        settings.plot_c2.plot(ax, X, Y, c**2)

    if settings.plot_energy:
        settings.plot_energy.plot(ax, X, Y, energy)

    if settings.plot_evaldiff:
        settings.plot_evaldiff.plot(
            ax, X, Y, eigenvalues[:, :, 1] - eigenvalues[:, :, 0]
        )

    if settings.plot_c_grad_norm:
        settings.plot_c_grad_norm.plot(ax, X, Y, np.linalg.norm(gradient_c, axis=2))

    if settings.plot_mode:
        settings.plot_mode.plot(ax, X, Y, mode[:, :, 0], mode[:, :, 1])

    if not settings.plot_gradient is None:
        settings.plot_gradient.plot(ax, X, Y, gradient[:, :, 0], gradient[:, :, 1])

    if settings.plot_gradient_c:
        settings.plot_gradient_c.plot(
            ax, X, Y, gradient_c[:, :, 0], gradient_c[:, :, 1]
        )

    if settings.plot_gradient_c2:
        settings.plot_gradient_c2.plot(
            ax, X, Y, 2 * c * gradient_c[:, :, 0], 2 * c * gradient_c[:, :, 1]
        )

    for path_settings in settings.path_plots:
        path_settings.plot(ax)

    if not settings.xlim is None:
        ax.set_xlim(settings.xlim)

    if not settings.ylim is None:
        ax.set_ylim(settings.ylim)

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if not settings.outfile is None:
        fig.savefig(settings.outfile, dpi=settings.dpi)

    if settings.show:
        plt.show()

    return return_value
