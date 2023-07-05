from ridgefollowing import energy_surface
from ridgefollowing.algorithms import modes, cosine_follower, gradient_extremal_follower

# from ridgefollowing import ridgefollower
from spirit_extras.plotting import Paper_Plot
from typing import Optional, List, Union
from pydantic import BaseModel, ConfigDict
from pathlib import Path
from dataclasses import dataclass
import numpy.typing as npt
import numpy as np
import matplotlib.pyplot as plt


class PathPlotSettings(BaseModel):
    model_config = ConfigDict(
            arbitrary_types_allowed = True)

    points: npt.NDArray
    ls: str = "-"
    color: str = "black"
    marker: Optional[str] = None
    mec: Optional[str] = None
    zorder: Optional[int] = 1

    lw: Optional[float] = None
    kwargs: dict = dict()

    label_points: bool = False

    def plot(self, ax):
        ax.plot(
            self.points[:, 0],
            self.points[:, 1],
            ls=self.ls,
            lw=self.lw,
            color=self.color,
            mec=self.mec,
            marker=self.marker,
            zorder=self.zorder,
            **self.kwargs,
        )

        if self.label_points:
            for i, p in enumerate(self.points):
                ax.text(p[0], p[1], f"{i}")


class ScalarPlotSettings(BaseModel):
    model_config = ConfigDict(
            arbitrary_types_allowed = True)

    contours: bool = False
    contours_filled: bool = True

    flip_sign: bool = False

    log_compression: bool = True

    colormap: Optional[str] = "coolwarm"
    linestyles: Optional[str] = None
    colors: Optional[str] = None
    contourlevels: Union[int, List[float]] = 20
    vmax: Optional[float] = None
    vmin: Optional[float] = None
    extend: Optional[str] = None

    zorder: int = 0

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
            elif self.flip_sign:
                Z *= -1.0

            if self.vmax:
                Z = np.max(Z)

            return_vals.append(
                f(
                    X,
                    Y,
                    Z,
                    extend=self.extend,
                    levels=self.contourlevels,
                    vmin=self.vmin,
                    vmax=self.vmax,
                    cmap=self.colormap,
                    colors=self.colors,
                    linestyles=self.linestyles,
                    zorder=self.zorder,
                )
            )
        return return_vals


class VectorPlotSettings(BaseModel):
    model_config = ConfigDict(
            arbitrary_types_allowed = True)

    streamplot: bool = True
    quiver: bool = False
    color: Optional[str] = None
    kwargs: Optional[dict] = dict()
    sampling: int = 1

    zorder: int = 0

    label: Optional[str] = None

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
                    X[:: self.sampling],
                    Y[:: self.sampling],
                    U[:: self.sampling, :: self.sampling],
                    V[:: self.sampling, :: self.sampling],
                    color=self.color,
                    # label=self.label,
                    zorder=self.zorder,
                    **self.kwargs,
                )
            )
        return return_vals


cm = Paper_Plot.cm


class PlotSettings(BaseModel):
    model_config = ConfigDict(
            arbitrary_types_allowed = True,
            validate_assignment = True)

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

    lims: npt.NDArray = np.array([[0, 1], [0, 1]])  # limits [[xmin, xmax], [ymin, ymax]
    npoints: npt.NDArray = np.array(
        [100, 100], dtype=np.int64
    )  # number of sample points [npointsx, npointsy]

    xlim: Optional[npt.NDArray] = None
    ylim: Optional[npt.NDArray] = None

    path_plots: List[PathPlotSettings] = []

    plot_energy: Optional[ScalarPlotSettings] = None
    plot_c2: Optional[ScalarPlotSettings] = None
    plot_evaldiff: Optional[ScalarPlotSettings] = None
    plot_eval1: Optional[ScalarPlotSettings] = None
    plot_eval2: Optional[ScalarPlotSettings] = None

    # distance to gradient extremal
    plot_grad_ext_dist1: Optional[ScalarPlotSettings] = None
    plot_grad_ext_dist2: Optional[ScalarPlotSettings] = None
    plot_grad_ext_crit: Optional[ScalarPlotSettings] = None

    plot_c_grad_norm: Optional[ScalarPlotSettings] = None
    plot_grad_norm: Optional[ScalarPlotSettings] = None

    plot_c2_mod: Optional[ScalarPlotSettings] = None
    plot_sum_c2: Optional[ScalarPlotSettings] = None

    plot_gradient: Optional[VectorPlotSettings] = None
    plot_mode: Optional[VectorPlotSettings] = None
    plot_mode2: Optional[VectorPlotSettings] = None
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

        data_aspect_ratio = (settings.lims[0][1] - settings.lims[0][0]) / (settings.lims[1][1] - settings.lims[1][0])
        pplot.apply_absolute_margins(
            aspect_ratio=data_aspect_ratio,
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
        X = np.linspace(settings.lims[0, 0], settings.lims[0, 1], settings.npoints[0], dtype=float)
        Y = np.linspace(settings.lims[1, 0], settings.lims[1, 1], settings.npoints[1], dtype=float)
        energy = np.empty(shape=settings.npoints)
        c = np.empty(shape=settings.npoints)
        sum_c2 = np.empty(shape=settings.npoints)
        mode = np.empty(shape=(*settings.npoints, surface.ndim))
        mode2 = np.empty(shape=(*settings.npoints, surface.ndim))
        gradient = np.empty(shape=(*settings.npoints, surface.ndim))
        gradient_norm = np.empty(shape=settings.npoints)
        gradient_c = np.empty(shape=(*settings.npoints, surface.ndim))
        eigenvalues = np.empty(shape=(*settings.npoints, surface.ndim))

        grad_ext_dist1 = np.empty(shape=settings.npoints)
        grad_ext_dist2 = np.empty(shape=settings.npoints)
        grad_ext_crit = np.empty(shape=settings.npoints)

        Rfollower = cosine_follower.CosineFollower(surface)
        Gfollower = gradient_extremal_follower.GradientExtremalFollower(surface)

        if not settings.output_data_folder is None:
            # assert settings.output_data_folder.is_dir()
            settings.output_data_folder.mkdir(parents=True, exist_ok=True)

        for xi, x in enumerate(X):
            for yi, y in enumerate(Y):
                i = xi * len(Y) + yi
                n = len(X) * len(Y)
                print(f"Point {i} / {n} ( {i/n*100:.2f} %)", end="\r")

                xy = np.array([x,y])

                energy[yi, xi] = surface.energy(xy)

                if (
                    settings.plot_mode
                    or settings.plot_mode2
                    or settings.plot_evaldiff
                    or settings.plot_eval1
                    or settings.plot_eval2
                    or settings.plot_sum_c2
                    or settings.plot_grad_ext_dist1
                    or settings.plot_grad_ext_dist2
                    or settings.plot_grad_ext_crit
                ):
                    hessian = surface.hessian(xy)
                    eigenvals, eigenvecs = modes.eigenpairs(hessian)

                    mode[yi, xi] = eigenvecs[:, 0]
                    mode2[yi, xi] = eigenvecs[:, 1]
                    eigenvalues[yi, xi] = eigenvals

                    G = surface.gradient(xy)

                    mode[yi, xi] = eigenvecs[:, 0]
                    mode2[yi, xi] = eigenvecs[:, 1]

                    hg = hessian @ G
                    hg_norm = hg / np.linalg.norm(hg)

                    G_norm = G / np.linalg.norm(G)

                    grad_ext_crit[yi, xi] = np.dot(hg_norm, G_norm) ** 2

                    grad_ext_dist1[yi, xi] = 1.0 - np.dot(mode[yi, xi], G_norm) ** 2

                    grad_ext_dist2[yi, xi] = (
                        eigenvals[0] - np.dot(hg, G) / np.linalg.norm(G) ** 2
                    ) ** 2

                    # grad_ext_dist2[yi, xi] = 1.0 - np.dot(mode2[yi, xi], G_norm)**2

                    Gfollower.compute_v(hessian, v_prev=mode[yi, xi])
                    # (
                    #     _,
                    #     grad_ext_dist2[yi, xi],
                    #     _,
                    # ) = Gfollower.compute_approximate_ridge_location(
                    #     xcur, G, hessian, mode[yi, xi]
                    # )

                    # Gfollower.compute_v(hessian, v_prev=mode2[yi, xi])
                    # (
                    #     _,
                    #     grad_ext_dist2[yi, xi],
                    #     _,
                    # ) = Gfollower.compute_approximate_ridge_location(
                    #     xcur, G, hessian, mode2[yi, xi]
                    # )

                if (
                    settings.plot_gradient
                    or settings.plot_sum_c2
                    or settings.plot_grad_norm
                ):
                    gradient[yi, xi] = surface.gradient(xy)
                    gradient_norm[yi, xi] = np.linalg.norm(gradient[yi, xi])
                    gradient[yi, xi] = gradient[yi, xi] / np.linalg.norm(
                        gradient[yi, xi]
                    )
                    sum_c2[yi, xi] = (
                        np.dot(gradient[yi, xi], mode[yi, xi]) ** 2
                        + np.dot(gradient[yi, xi], mode2[yi, xi])
                    ) ** 2

                if settings.plot_c2 or settings.plot_gradient_c2:
                    c[yi, xi] = Rfollower.C(xy)

                if (
                    settings.plot_gradient_c
                    or settings.plot_c_grad_norm
                    or settings.plot_gradient_c2
                ):
                    gradient_c[yi, xi] = Rfollower.fd_grad_C(xy)
    else:
        assert (
            settings.input_data_folder.is_dir() and settings.input_data_folder.exists()
        )
        X = np.load(settings.input_data_folder / "X.npy")
        Y = np.load(settings.input_data_folder / "Y.npy")

        if (settings.input_data_folder / "energy.npy").exists():
            energy = np.load(settings.input_data_folder / "energy.npy")
        if (settings.input_data_folder / "c.npy").exists():
            c = np.load(settings.input_data_folder / "c.npy")
        if (settings.input_data_folder / "mode.npy").exists():
            mode = np.load(settings.input_data_folder / "mode.npy")
        if (settings.input_data_folder / "mode2.npy").exists():
            mode2 = np.load(settings.input_data_folder / "mode2.npy")
        if (settings.input_data_folder / "sum_c2.npy").exists():
            sum_c2 = np.load(settings.input_data_folder / "sum_c2.npy")
        if (settings.input_data_folder / "gradient.npy").exists():
            gradient = np.load(settings.input_data_folder / "gradient.npy")
        if (settings.input_data_folder / "gradient_norm.npy").exists():
            gradient_norm = np.load(settings.input_data_folder / "gradient_norm.npy")
        if (settings.input_data_folder / "gradient_c.npy").exists():
            gradient_c = np.load(settings.input_data_folder / "gradient_c.npy")
        if (settings.input_data_folder / "eigenvalues.npy").exists():
            eigenvalues = np.load(settings.input_data_folder / "eigenvalues.npy")
        if (settings.input_data_folder / "grad_ext_dist1.npy").exists():
            grad_ext_dist1 = np.load(settings.input_data_folder / "grad_ext_dist1.npy")
        if (settings.input_data_folder / "grad_ext_dist2.npy").exists():
            grad_ext_dist2 = np.load(settings.input_data_folder / "grad_ext_dist2.npy")
        if (settings.input_data_folder / "grad_ext_crit.npy").exists():
            grad_ext_crit = np.load(settings.input_data_folder / "grad_ext_crit.npy")

    if not settings.output_data_folder is None:
        np.save(settings.output_data_folder / "X", X)
        np.save(settings.output_data_folder / "Y", Y)

        if settings.plot_energy:
            np.save(settings.output_data_folder / "energy", energy)

        if (
            settings.plot_mode
            or settings.plot_mode2
            or settings.plot_evaldiff
            or settings.plot_sum_c2
            or settings.plot_eval1
            or settings.plot_eval2
            or settings.plot_grad_ext_dist1
            or settings.plot_grad_ext_dist2
            or settings.plot_grad_ext_crit
        ):
            np.save(settings.output_data_folder / "eigenvalues", eigenvalues)
            np.save(settings.output_data_folder / "mode", mode)
            np.save(settings.output_data_folder / "mode2", mode2)
            np.save(settings.output_data_folder / "sum_c2", sum_c2)
            np.save(settings.output_data_folder / "grad_ext_dist1", grad_ext_dist1)
            np.save(settings.output_data_folder / "grad_ext_dist2", grad_ext_dist2)
            np.save(settings.output_data_folder / "grad_ext_crit", grad_ext_crit)

        if settings.plot_gradient or settings.plot_grad_norm:
            np.save(settings.output_data_folder / "gradient", gradient)
            np.save(settings.output_data_folder / "gradient_norm", gradient_norm)

        if settings.plot_c2 or settings.plot_gradient_c2:
            np.save(settings.output_data_folder / "c", c)

        if (
            settings.plot_gradient_c
            or settings.plot_c_grad_norm
            or settings.plot_gradient_c2
        ):
            np.save(settings.output_data_folder / "gradient_c", gradient_c)

    if settings.plot_c2:
        settings.plot_c2.plot(ax, X, Y, c**2)

    if settings.plot_energy:
        settings.plot_energy.plot(ax, X, Y, energy)

    if settings.plot_evaldiff:
        settings.plot_evaldiff.plot(
            ax, X, Y, eigenvalues[:, :, 1] - eigenvalues[:, :, 0]
        )

    if settings.plot_eval1:
        settings.plot_eval1.plot(ax, X, Y, eigenvalues[:, :, 0])

    if settings.plot_eval2:
        settings.plot_eval2.plot(ax, X, Y, eigenvalues[:, :, 1])

    if settings.plot_c_grad_norm:
        settings.plot_c_grad_norm.plot(ax, X, Y, np.linalg.norm(gradient_c, axis=2))

    if settings.plot_grad_norm:
        settings.plot_grad_norm.plot(ax, X, Y, gradient_norm)

    if settings.plot_sum_c2:
        settings.plot_sum_c2.plot(ax, X, Y, sum_c2)

    if settings.plot_grad_ext_dist1:
        settings.plot_grad_ext_dist1.plot(ax, X, Y, grad_ext_dist1)

    if settings.plot_grad_ext_dist2:
        settings.plot_grad_ext_dist2.plot(ax, X, Y, grad_ext_dist2)

    if settings.plot_grad_ext_crit:
        settings.plot_grad_ext_crit.plot(ax, X, Y, grad_ext_crit)

    if settings.plot_mode:
        settings.plot_mode.plot(ax, X, Y, mode[:, :, 0], mode[:, :, 1])

    if settings.plot_mode2:
        settings.plot_mode2.plot(ax, X, Y, mode2[:, :, 0], mode2[:, :, 1])

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
    else:
        ax.set_xlim(settings.lims[0])

    if not settings.ylim is None:
        ax.set_ylim(settings.ylim)
    else:
        ax.set_ylim(settings.lims[1])


    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()

    if not settings.outfile is None:
        fig.savefig(settings.outfile, dpi=settings.dpi)

    if settings.show:
        plt.show()

    return return_value
