from energy_surfaces.surfaces import cubic
from ridgefollowing.algorithms import gradient_extremal_follower, cosine_follower, modes
from ridgefollowing.plotting import plot_surface
import numpy as np
from pathlib import Path

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
    # plot_grad_ext_dist2=plot_surface.ScalarPlotSettings(colormap="coolwarm"),
    plot_c2=plot_surface.ScalarPlotSettings(colormap="coolwarm"),
    # plot_mode=plot_surface.VectorPlotSettings(streamplot=False, quiver=True, kwargs=dict(scale=10)),
    # plot_mode2=plot_surface.VectorPlotSettings(),
    # plot_eval1=plot_surface.ScalarPlotSettings(colormap="coolwarm", contours_filled=False, contours=True, zorder=3, log_compression=True),
    # input_data_folder="./data_quapp4_200",
    output_data_folder="./data_quapp4_200",
    outfile="plot_quapp4.png",
    npoints=np.array([200, 200]),
    show=True,
)

esurf = cubic.CubicSurface()
esurf.setup_quapp_example(4)

follower_trajectories = []

follower = gradient_extremal_follower.GradientExtremalFollower(
    energy_surface=esurf, trust_radius=1e-2, n_iterations_follow=400
)

x_cur = np.array([1.333, 0.671632])
G = esurf.gradient(x_cur)
H = esurf.hessian(x_cur)
evals, evecs = modes.eigenpairs(H)
idx = 0
v = evecs[:, idx]
follower.cur_eval = evals[idx]
x0, dist, dir = follower.compute_approximate_ridge_location(x_cur, G, H, v)
# print(v)
# print(evals[idx])
# print(G)
# print(H)
# print(evals)

x0 = -np.linalg.inv(H) @ G
x0 = x0 - np.dot(x0, v) * v

print("evals", evals)
print("H^-1", -np.linalg.inv(H) @ G)
print("proj,", np.dot(x0, v) * v)

print(x0, dist, dir)
# exit(0)

follower_cos = cosine_follower.CosineFollower(
    energy_surface=esurf, radius=1, n_iterations_follow=5, tolerance=1e-6
)
follower_cos.maximize = False

for f in [follower_cos]:
    # f.follow(np.array([1.3, 0.3]), np.array([1.0, 0.0]))
    # follower_trajectories.append( f.history["x_cur"].copy() )

    # f.follow(np.array([1.3, 0.3]), np.array([-1.0, 0.0]))
    # follower_trajectories.append( f.history["x_cur"].copy() )

    # f.follow(np.array([3.1275729811771606, 1.2653225689228282]), np.array([0.0, 1.0]))
    # follower_trajectories.append(f.history["x_cur"].copy())

    f.output_path = "./1"
    f.follow(np.array([3.1275729811771606, 1.2653225689228282]), np.array([-1.0, 0.0]))
    follower_trajectories.append(f.history["x_cur"].copy())

    f.output_path = "./2"
    f.radius /= 2.0
    f.n_iterations_follow *= 2
    f.follow(np.array([3.1275729811771606, 1.2653225689228282]), np.array([0.0, 1.0]))
    follower_trajectories.append(f.history["x_cur"].copy())

    f.output_path = "./3"
    f.radius /= 2.0
    f.n_iterations_follow *= 2
    f.follow(np.array([3.1275729811771606, 1.2653225689228282]), np.array([0.0, 1.0]))
    follower_trajectories.append(f.history["x_cur"].copy())

    f.output_path = "./4"
    f.radius /= 2.0
    f.n_iterations_follow *= 2
    f.follow(np.array([3.1275729811771606, 1.2653225689228282]), np.array([0.0, 1.0]))
    follower_trajectories.append(f.history["x_cur"].copy())

    f.output_path = "./5"
    f.radius /= 2.0
    f.n_iterations_follow *= 2
    f.follow(np.array([3.1275729811771606, 1.2653225689228282]), np.array([0.0, 1.0]))
    follower_trajectories.append(f.history["x_cur"].copy())

    f.output_path = "./6"
    f.radius /= 2.0
    f.n_iterations_follow *= 2
    f.follow(np.array([3.1275729811771606, 1.2653225689228282]), np.array([0.0, 1.0]))
    follower_trajectories.append(f.history["x_cur"].copy())

    # follower_trajectories.append(f.history["x0"].copy())
    # f.dump_history(Path("./hist"))

    # f.follow(np.array([0.5833, 0.7034]), np.array([0.0, 1.0]))
    # follower_trajectories.append(f.history["x_cur"].copy())

    # f.follow(np.array([1.3, 0.3]), np.array([0.0, -1.0]))
    # follower_trajectories.append(f.history["x_cur"].copy())

    # f.follow(np.array([-1.3, 0.3]), np.array([1.0, 0.0]))
    # follower_trajectories.append(f.history["x_cur"].copy())

    # f.follow(np.array([-1.3, 0.3]), np.array([-1.0, 0.0]))
    # follower_trajectories.append(f.history["x_cur"].copy())

    # f.follow(np.array([-1.3, 0.3]), np.array([0.0, 1.0]))
    # follower_trajectories.append(f.history["x_cur"].copy())

    # f.follow(np.array([-1.3, 0.3]), np.array([0.0, -1.0]))
    # follower_trajectories.append(f.history["x_cur"].copy())

    # f.follow(np.array([0.3, 0.75]), np.array([0.3, -1.0]))
    # follower_trajectories.append(f.history["x_cur"].copy())


for i, t in enumerate(follower_trajectories):
    color = f"C{i+1}"
    settings.path_plots.append(
        plot_surface.PathPlotSettings(points=t, color=color, marker=".", zorder=11)
    )
    settings.path_plots.append(
        plot_surface.PathPlotSettings(
            points=np.array([t[0]]), marker="x", mec="black", color=color, zorder=11
        )
    )

plot_surface.plot(esurf, settings=settings)
