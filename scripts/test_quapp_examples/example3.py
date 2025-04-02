from energy_surfaces.surfaces import cubic
from ridgefollowing.algorithms import gradient_extremal_follower
from ridgefollowing.plotting import plot_surface
import numpy as np

settings = plot_surface.PlotSettings(
    width=10 * plot_surface.cm,
    lims=np.array([[-4, 4], [-4, 4]]),
    plot_energy=plot_surface.ScalarPlotSettings(
        contours_filled=False,
        contours=True,
        colormap=None,
        colors="grey",
        zorder=2,
        log_compression=False,
        contourlevels=50,
        vmin=-0.5,
        vmax=0.5,
        extend="neither",
    ),
    plot_c2=plot_surface.ScalarPlotSettings(colormap="coolwarm"),
    # plot_grad_norm=plot_surface.ScalarPlotSettings(),
    # input_data_folder="./data_quapp3",
    output_data_folder="./data_quapp3",
    outfile="plot_quapp3.png",
    npoints=np.array([200, 200]),
    # show=True,
)

esurf = cubic.CubicSurface()
esurf.setup_quapp_example(3)

follower_trajectories = []

follower = gradient_extremal_follower.GradientExtremalFollower(
    energy_surface=esurf, trust_radius=1e-2, n_iterations_follow=300
)
follower.follow(np.array([0.3, 0.3]), np.array([1.0, 0.0]))
follower_trajectories.append(follower.history["x_cur"].copy())

follower.follow(np.array([0.3, 0.3]), np.array([-1.0, 0.0]))
follower_trajectories.append(follower.history["x_cur"].copy())

follower.follow(np.array([0.3, 0.3]), np.array([0.0, 1.0]))
follower_trajectories.append(follower.history["x_cur"].copy())

follower.follow(np.array([0.3, 0.3]), np.array([0.0, -1.0]))
follower_trajectories.append(follower.history["x_cur"].copy())

follower.follow(np.array([-0.3, -0.3]), np.array([1.0, 0.0]))
follower_trajectories.append(follower.history["x_cur"].copy())

follower.follow(np.array([-0.3, -0.3]), np.array([-1.0, 0.0]))
follower_trajectories.append(follower.history["x_cur"].copy())

follower.follow(np.array([-0.3, -0.3]), np.array([0.0, 1.0]))
follower_trajectories.append(follower.history["x_cur"].copy())

follower.follow(np.array([-0.3, -0.3]), np.array([0.0, -1.0]))
follower_trajectories.append(follower.history["x_cur"].copy())

for i, t in enumerate(follower_trajectories):
    color = f"C{i}"
    settings.path_plots.append(
        plot_surface.PathPlotSettings(points=t, color=color, marker=".", zorder=11)
    )
    settings.path_plots.append(
        plot_surface.PathPlotSettings(
            points=np.array([t[0]]), marker="x", mec="black", color=color, zorder=11
        )
    )

plot_surface.plot(esurf, settings=settings)
