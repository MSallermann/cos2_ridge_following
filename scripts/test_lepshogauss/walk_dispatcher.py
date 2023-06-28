from pydantic import BaseModel, validator
from typing import List, Optional
from pathlib import Path
import numpy.typing as npt
from spirit_extras import calculation_folder
import argparse
import numpy as np
import enum
import matplotlib.pyplot as plt

from ridgefollowing.surfaces import lepshogauss
from ridgefollowing.algorithms import cosine_follower, gradient_extremal_follower


class FollowerTypes(enum.Enum):
    gradient_extremal = 0
    cosine = 1


class WalkSettings(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        allow_reuse = True

    x0: npt.NDArray
    d0: npt.NDArray

    radius: float
    type: FollowerTypes
    n_follow: int

    outputfolder: Path

    bifurcations: bool = False
    bifurcations_folder: Path = Path("bifurcations")

    @validator("x0", "d0", pre=True)
    def convert_to_ndarray(cls, v) -> npt.NDArray:
        return np.array(v, dtype=float)

    @validator("outputfolder", pre=True)
    def convert_to_Path(cls, v):
        return Path(v)

    @validator("type", pre=True)
    def check(cls, v):
        return FollowerTypes[v]


def plot_history(follower, output):
    output.mkdir(exist_ok=True)
    for k, v in follower.history.items():
        if len(v.shape) == 1:
            plt.plot(v, marker=".")
            plt.xlabel("iteration")
            plt.ylabel(k)
            # plt.xticks( [i for i in range(len(v))] )
            plt.savefig(str(output / f"{k}.png"))
            plt.tight_layout()
            plt.close()


def main(input_folder):
    f = calculation_folder.Calculation_Folder(input_folder)
    print(f.info_string())
    settings = WalkSettings(**f)

    esurf = lepshogauss.LepsHOGaussSurface()

    if settings.type == FollowerTypes.gradient_extremal:
        follower = gradient_extremal_follower.GradientExtremalFollower(
            energy_surface=esurf,
            trust_radius=settings.radius,
            n_iterations_follow=settings.n_follow,
            output_path=Path(f),
        )
    elif settings.type == FollowerTypes.cosine:
        follower = cosine_follower.CosineFollower(
            energy_surface=esurf,
            radius=settings.radius,
            n_iterations_follow=settings.n_follow,
            output_path=Path(f),
        )

    follower.follow(settings.x0, settings.d0)
    follower.dump_history()

    follower.plot_history()

    bif_points = follower.bifurcation_points.copy()

    f["bifurcation_points"] = bif_points
    f["x_final"] = follower.history["x_cur"][-1]
    f["d_final"] = follower.history["step_cur"][-1]

    f.to_desc()

    print("Running bifurcations ...")
    for i, (x, d) in enumerate(bif_points):
        bif_folder = Path(f) / settings.bifurcations_folder / f"bifurcation_{i}_plus"
        follower.output_path = bif_folder

        temp = calculation_folder.Calculation_Folder(
            bif_folder, create=True, descriptor_file="walk.toml"
        )
        temp.update(f)
        temp["d0"] = [d_ for d_ in d]
        temp["x0"] = [x_ for x_ in x]
        temp["bifurcations"] = False
        temp.to_desc()

        if settings.bifurcations:
            follower.follow(x, d)
            follower.dump_history()
            follower.plot_history()

        bif_folder = Path(f) / settings.bifurcations_folder / f"bifurcation_{i}_minus"
        follower.output_path = bif_folder

        temp = calculation_folder.Calculation_Folder(
            bif_folder, create=True, descriptor_file="walk.toml"
        )
        temp.update(f)
        temp["d0"] = [-d_ for d_ in d]
        temp["x0"] = [x_ for x_ in x]
        temp["bifurcations"] = False
        temp.to_desc()

        if settings.bifurcations:
            follower.follow(x, -d)
            follower.dump_history()
            follower.plot_history()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Dispatch walks")
    parser.add_argument("paths", nargs="*")

    args = parser.parse_args()

    # main(Path("/home/moritz/Coding/cos2_ridge_following/scripts/test_lepshogauss/cosine_bifurcation_walks/walk_min_2_-y_tr"))

    for p in args.paths:
        main(p)
