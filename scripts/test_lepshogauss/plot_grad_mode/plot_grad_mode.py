from ridgefollowing.surfaces import lepshogauss
from ridgefollowing.algorithms import minimizer
from ridgefollowing.plotting import plot_surface
from pathlib import Path
import numpy as np
import numpy.typing as npt


esurf = lepshogauss.LepsHOGaussSurface()

lims = np.array( [[1,1.15], [-1.5,-0.4]] )
folder = Path("../data500")
npoints = np.array([500,500])

settings = plot_surface.PlotSettings(
        width=15 * plot_surface.cm,
        outfile=f"plot_grad_mode_{lims}.png",
        lims=lims,
        plot_evaldiff=plot_surface.ScalarPlotSettings(
            contourlevels=20,
            contours_filled=False,
            contours=True,
            colors="grey",
            colormap=None,
            log_compression=True,
            linestyles="solid",
            zorder = -1
        ),
        plot_c2=plot_surface.ScalarPlotSettings(
            contourlevels=40, log_compression=False, colormap="coolwarm", contours_filled=True, zorder=-2
        ),
        plot_gradient=plot_surface.VectorPlotSettings(color="blue", streamplot=False, quiver=True, kwargs=dict(scale=20), sampling=1, label="Gradient"),
        plot_mode=plot_surface.VectorPlotSettings(color="red", streamplot=False, quiver=True, kwargs=dict(scale=20), sampling=1, label="Minimum Mode"),
        output_data_folder=folder,
        input_data_folder=folder,
        npoints=npoints,
        show=True,
    )

def make_plot(settings):

    def plot_walks(output_dir : Path, color):
        for walk_dir in output_dir.glob("*"):
            print(f"Loading from {walk_dir}")
            trajectory = np.load(walk_dir / "trajectory.npy")

            settings.path_plots.append(plot_surface.PathPlotSettings(points=trajectory, color=color))
            settings.path_plots.append(
                plot_surface.PathPlotSettings(points=np.array([trajectory[0]]), marker="o", color=color)
            )

    # plot_walks(Path("./walks_bifurcation_1"), color="red")
    # plot_walks(Path("./walks_bifurcation_2"), color="blue")
    # plot_walks(Path("./walks_bifurcation_3"), color="green")
    # plot_walks(Path("./walks_bifurcation_4"), color="yellow")
    # plot_walks(Path("./walks_return_1"), color="yellow")

    plot_surface.plot(esurf, settings=settings)

if __name__ == "__main__":

    folder = Path("../data100")
    L =     [
                np.array([[1.0497508613264181, 1.0640660677370612], [-1.0358282254699618, -0.9328841117058126]]),
                # np.array([[1.1, 1.4], [1.4, 2.3]]),
                # np.array([[1.5, 1.8], [-1, 0.11]]),
                # np.array([[1.8, 2.0], [0.8, 2.00]]),
                # np.array([[2.2, 2.5], [-0.9, 0.00]])
                # np.array([[2.338728843488924, 2.3431832930460974], [-0.6409472876746907, -0.6255815664564965]]),
            ]
    for lims in L:
        s = settings.copy()
        s.npoints = np.array([100,100])
        s.lims = lims
        s.input_data_folder = None
        plot_surface.plot(esurf, settings=s)


