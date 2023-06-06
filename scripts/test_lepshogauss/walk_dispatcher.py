from pydantic import BaseModel, validator
from typing import List, Optional
from pathlib import Path
import numpy.typing as npt
from spirit_extras import calculation_folder
import argparse
import numpy as np
import enum

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

    mode_index: Optional[int]

    @validator("x0", "d0", pre=True)
    def convert_to_ndarray(cls, v) -> npt.NDArray:
        return np.array(v, dtype=float)

    @validator("outputfolder", pre=True)
    def convert_to_Path(cls, v):
        return Path(v)

    @validator("type", pre=True)
    def check(cls, v):
        return FollowerTypes[v]


def main(input_folder):
    f = calculation_folder.Calculation_Folder(input_folder)
    settings = WalkSettings(**f)

    esurf = lepshogauss.LepsHOGaussSurface()

    if settings.type == FollowerTypes.gradient_extremal:
        follower = gradient_extremal_follower.GradientExtremalFollower(
            energy_surface=esurf,
            trust_radius=settings.radius,
            n_iterations_follow=settings.n_follow,
            mode_index=settings.mode_index,
        )
    elif settings.type == FollowerTypes.cosine:
        follower = cosine_follower.CosineFollower(
            energy_surface=esurf,
            radius=settings.radius,
            n_iterations_follow=settings.n_follow,
        )

    follower.follow(settings.x0, settings.d0)

    if settings.outputfolder.is_absolute():
        follower.dump_history(settings.outputfolder)
    else:
        follower.dump_history(Path(f) / settings.outputfolder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Dispatch walks")
    parser.add_argument("paths", nargs="*")

    args = parser.parse_args()

    for p in args.paths:
        main(p)
