from ridgefollowing.surfaces import lepshogauss
from spirit_extras.calculation_folder import Calculation_Folder
from ridgefollowing.plotting import plot_surface
from pathlib import Path
from typing import List, Optional
import numpy as np

esurf = lepshogauss.LepsHOGaussSurface()

lims = np.array([[0.25, 3.5], [-5, 5]])
npoints = np.array([200, 200])

settings = plot_surface.PlotSettings(
    width=15 * plot_surface.cm,
    outfile="plot_ext.png",
    plot_energy=plot_surface.ScalarPlotSettings(
        contourlevels=40,
        contours_filled=False,
        contours=True,
        colors="grey",
        colormap=None,
        log_compression=False,
        zorder=9,
    ),
    plot_grad_ext_crit=plot_surface.ScalarPlotSettings(
        contourlevels=900,
        log_compression=True,
        colormap="coolwarm",
        contours_filled=True,
    ),
    plot_c2=plot_surface.ScalarPlotSettings(
        contourlevels=900,
        log_compression=False,
        colormap="coolwarm",
        contours_filled=True,
    ),
)


def plot_walks(output_dir: Path, color):
    trajectory = np.load(output_dir / "x_cur.npy")

    settings.path_plots.append(
        plot_surface.PathPlotSettings(
            points=trajectory, color=color, marker=".", zorder=10, label_points=True
        )
    )
    settings.path_plots.append(
        plot_surface.PathPlotSettings(
            points=np.array([trajectory[0]]), marker="x", color=color, zorder=10
        )
    )


def main(
    walk_dirs: List[Path],
    outfile: Path,
    show: bool,
    c2: bool,
    grad_norm: bool,
    data_folder: Optional[str],
    regenerate_data: bool,
):
    if not data_folder is None:
        f = Calculation_Folder(data_folder, descriptor_file="meta.toml")
        print("===============")
        print(f.info_string())
        print("===============")

        settings.lims = np.array(f["lims"])
        print(Path(f))
        settings.output_data_folder = Path(f)
        settings.input_data_folder = Path(f)
        if regenerate_data:
            settings.input_data_folder = None
        settings.npoints = np.array(f["npoints"])

    for ip, p in enumerate(walk_dirs):
        f = Calculation_Folder(p)
        plot_walks(Path(p) / f["outputfolder"], color=f"C{ip}")

    # settings.outfile = str(outfile)
    settings.show = show

    if not c2:
        settings.plot_c2 = None

    if not grad_norm:
        settings.plot_grad_ext_crit = None

    plot_surface.plot(esurf, settings=settings)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("walk_dirs", nargs="*")
    parser.add_argument("-o", nargs=1)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--c2", action="store_true")
    parser.add_argument("--norm", action="store_true")
    parser.add_argument("--datafolder", default="./data200")
    parser.add_argument("--regenerate_data", action="store_true")

    args = parser.parse_args()

    main(
        args.walk_dirs,
        args.o,
        args.show,
        args.c2,
        args.norm,
        args.datafolder,
        args.regenerate_data,
    )
