from ridgefollowing.surfaces import lepshogauss
from ridgefollowing.algorithms import minimizer
from ridgefollowing.plotting import plot_surface
from pathlib import Path
import numpy as np

esurf = lepshogauss.LepsHOGaussSurface()

lims = np.array([[0.25, 3.5], [-5, 5]])
folder = Path("./data500")

settings = plot_surface.PlotSettings(
    width=15 * plot_surface.cm,
    outfile="plot.png",
    lims=lims,
    plot_energy=plot_surface.ScalarPlotSettings(
        contourlevels=40,
        contours_filled=False,
        contours=True,
        colors="grey",
        colormap=None,
        log_compression=False,
    ),
    plot_c2=plot_surface.ScalarPlotSettings(
        contourlevels=900, log_compression=False, colormap="coolwarm", contours_filled=True
    ),
    output_data_folder=folder,
    input_data_folder=folder,
    npoints=np.array([500, 500]),
    show=True,
)

def plot_walks(output_dir : Path, color):
    for walk_dir in output_dir.glob("*"):
        print(f"Loading from {walk_dir}")
        trajectory = np.load(walk_dir / "trajectory.npy")

        settings.path_plots.append(plot_surface.PathPlotSettings(points=trajectory, color=color))
        settings.path_plots.append(
            plot_surface.PathPlotSettings(points=np.array([trajectory[0]]), marker="o", color=color)
        )

if __name__ == "__main__":
    # plot_walks(Path("./walks_bifurcation_1"), color="red")
    # plot_walks(Path("./walks_bifurcation_2"), color="blue")
    # plot_walks(Path("./walks_bifurcation_3"), color="green")
    # plot_walks(Path("./walks_bifurcation_4"), color="yellow")

    plot_surface.plot(esurf, settings=settings)

