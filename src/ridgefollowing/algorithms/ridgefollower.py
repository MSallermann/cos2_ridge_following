import abc
from ridgefollowing import energy_surface
from ridgefollowing.algorithms import modes, spherical_optimizer
import numpy.typing as npt
from typing import Optional, List, Union
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import numdifftools as nd


class RidgeFollower(abc.ABC):
    def __init__(
        self,
        energy_surface: energy_surface.EnergySurface,
        n_iterations_follow: int = 100,
    ) -> None:
        self.esurf: energy_surface.EnergySurface = energy_surface
        self.n_iterations_follow: int = n_iterations_follow
        self.print_progress: bool = True

        self.history: dict = dict(
            x_cur=np.zeros(shape=(self.n_iterations_follow, self.esurf.ndim)),
            d_cur=np.zeros(shape=(self.n_iterations_follow, self.esurf.ndim)),
            E=np.zeros(shape=(self.n_iterations_follow)),
            G=np.zeros(shape=(self.n_iterations_follow, self.esurf.ndim)),
        )

    @abc.abstractmethod
    def determine_step(self):
        ...

    def log_history(self):
        self.history["x_cur"][self._iteration] = np.array(self._x_cur)
        self.history["d_cur"][self._iteration] = np.array(self._d_cur)
        self.history["E"][self._iteration] = self._E
        self.history["G"][self._iteration] = self._G

    def follow(self, x0: npt.NDArray, d0: npt.NDArray):
        self._x_cur = np.array(x0)
        self._d_cur = np.array(d0)

        for i in range(self.n_iterations_follow):
            self._iteration = i
            if self.print_progress:
                prog = i / self.n_iterations_follow * 100
                print(
                    f"Iteration {i} / {self.n_iterations_follow} ( {prog:.3f}% )",
                    end="\r",
                )

            # Save old direction
            self._d_prev = self._d_cur

            step = self.determine_step()

            self._x_cur += step

            self._E = self.esurf.energy(self._x_cur)
            self._G = self.esurf.gradient(self._x_cur)

            self.log_history()
