import abc
from ridgefollowing import energy_surface
from typing import Optional
import numpy.typing as npt
import numpy as np
from pathlib import Path


class RidgeFollower(abc.ABC):
    def __init__(
        self,
        energy_surface: energy_surface.EnergySurface,
        n_iterations_follow: int = 100,
    ) -> None:
        self.esurf: energy_surface.EnergySurface = energy_surface
        self.n_iterations_follow: int = n_iterations_follow
        self.print_progress: bool = True
        self.output_path: Path = Path(".")

        self._iteration: Optional[int] = None
        self._x_cur: npt.NDArray = np.zeros(self.esurf.ndim)
        self._d0: npt.NDArray = np.zeros(self.esurf.ndim)
        self._step_prev: npt.NDArray = np.zeros(self.esurf.ndim)
        self._step_cur: npt.NDArray = np.zeros(self.esurf.ndim)
        self._G: npt.NDArray = np.zeros(self.esurf.ndim)
        self._E: float = 0.0

    def setup_history(self):
        self.history: dict = dict(
            x_cur=np.zeros(shape=(self.n_iterations_follow, self.esurf.ndim)),
            step_cur=np.zeros(shape=(self.n_iterations_follow, self.esurf.ndim)),
            E=np.zeros(shape=(self.n_iterations_follow)),
            G=np.zeros(shape=(self.n_iterations_follow, self.esurf.ndim)),
        )

    def log_history(self):
        self.history["x_cur"][self._iteration] = np.array(self._x_cur)
        self.history["step_cur"][self._iteration] = np.array(self._step_cur)
        self.history["E"][self._iteration] = self._E
        self.history["G"][self._iteration] = self._G

    def dump_history(self, folder: Optional[Path] = None):
        if folder is None:
            folder = self.output_path

        folder.mkdir(exist_ok=True)
        for k, v in self.history.items():
            np.save(folder / k, v)

    @abc.abstractmethod
    def determine_step(self):
        ...

    def follow(self, x0: npt.NDArray, d0: npt.NDArray):
        self.setup_history()

        self._x_cur = np.array(x0)
        self._step_cur = d0
        self._step_prev = d0

        for i in range(self.n_iterations_follow):
            self._iteration = i
            if self.print_progress:
                prog = (i + 1) / self.n_iterations_follow * 100
                print(
                    f"Iteration {i} / {self.n_iterations_follow} ( {prog:.3f}% )",
                    end="\n",
                )

            # Save old direction
            self._step_prev = self._step_cur

            self._E = self.esurf.energy(self._x_cur)
            self._G = self.esurf.gradient(self._x_cur)
            self._step_cur = self.determine_step()
            self.log_history()

            self._x_cur += self._step_cur
