from energy_surfaces.surfaces import lepshogauss
from ridgefollowing.algorithms import minimizer, cosine_follower
from pathlib import Path
import numpy as np
import numpy.typing as npt


esurf = lepshogauss.LepsHOGaussSurface()
follower = cosine_follower.CosineFollower(
    energy_surface=esurf, radius=1e-1, n_iterations_follow=100
)

follower.magnitude_modified_gaussian = 0
follower.width_modified_gaussian = 0.7


def walk(x_start: npt.NDArray, dir_start: npt.NDArray, output_folder: Path):
    output_folder.mkdir(parents=True, exist_ok=True)
    follower.follow(x_start, dir_start)
    np.save(output_folder / "trajectory", follower.history["x_cur"])
    np.save(output_folder / "energy", follower.history["E"])
    np.save(output_folder / "gradient", follower.history["G"])


def phi_walks(x0, width, output_dir, phi):
    n_walks = len(phi)
    follower.width_modified_gaussian = width
    for ip, p in enumerate(phi):
        print(f"\nWalk {ip + 1} / {n_walks}")
        dir = np.array([np.cos(p), np.sin(p)])
        walk(x0, dir_start=dir, output_folder=output_dir / f"walk_phi_{p:.3f}")


# Find the minimum
min = minimizer.Minimizer(energy_surface=esurf)
x_min = min.minimize_energy(x0=np.array([1.0, 0.0]))

# """Run from the bifurcation point"""
x_bif = np.array([1.006, -4.31109])
phi_list = np.array([np.pi / 2])

phi_walks(x0=x_bif, width=0.5, output_dir=Path("./walks_bifurcation_1"), phi=phi_list)
# phi_walks( x0=x_bif, width=0.6, output_dir=Path("./walks_bifurcation_2"), phi=phi_list )
# phi_walks( x0=x_bif, width=0.7, output_dir=Path("./walks_bifurcation_3"), phi=phi_list )
# phi_walks( x0=x_bif, width=0.8, output_dir=Path("./walks_bifurcation_4"), phi=phi_list )


# """Run from the return point of the first trajectory"""
# x_return = np.array([ 1.15154047, -0.99335039])
# phi_walks( x0=x_return, width=0.5, output_dir=Path("./walks_return_1"), phi=np.linspace(0, 2*np.pi, 127)[:-1] )
