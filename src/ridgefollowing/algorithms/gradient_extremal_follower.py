from ridgefollowing import energy_surface
from ridgefollowing.algorithms import ridgefollower, modes
from typing import Optional, List, Union
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import numdifftools as nd


class GradientExtremalFollower(ridgefollower.RidgeFollower):
    def __init__(
        self,
        energy_surface: energy_surface.EnergySurface,
        n_iterations_follow: int = 100,
        trust_radius: float = 1e-4,
        mode_index: int = 0,
    ) -> None:
        super().__init__(energy_surface, n_iterations_follow)
        if not mode_index is None:
            self.mode_index = mode_index
        self.trust_radius = trust_radius
        self.v = np.zeros(self.esurf.ndim)

    def determine_step(self):
        self._H = self.esurf.hessian(self._x_cur)

        evals, evecs = modes.lowest_n_modes(self._H, n=2)

        self.v_prev = self.v
        self.v = evecs[:, self.mode_index]

        # Mode should point into one consistent direction
        if np.dot(self.v_prev, self.v) < 0:
            self.v *= -1

        lam = evals[self.mode_index]

        x0 = -np.linalg.inv(self._H) @ self._G
        x0 = (
            x0 - np.dot(x0, self.v) * self.v
        )  # solution of the projected newton equation

        # Compute the shortest distance between extremal and x_cur
        dir_extremal = x0 - self._x_cur
        dir_extremal = dir_extremal - np.dot(dir_extremal, self.v) * self.v
        dist_extremal = np.linalg.norm(dir_extremal)

        # print("--")
        # print("self._x_cur", self._x_cur)
        # print("x0", x0)
        # print("dist_extremal", dist_extremal )
        # print("v", self.v)
        # print("g", self._G)
        # print("--\n")

        if dist_extremal < self.trust_radius:
            # Determine alpha such that '|x0 - x_cur + alpha * v| = self.trust_radius'
            v2 = np.linalg.norm(self.v) ** 2
            delta = x0 - self._x_cur
            delta2 = np.linalg.norm(delta) ** 2
            delta_v = np.dot(self.v, delta)

            # Quadratic equation
            # 0 = v**2 * alpha*2 + 2*v*delta * alpha + delta**2 - r**2
            alpha_p = -delta_v / v2 + np.sqrt(
                (delta_v / v2) ** 2 - (delta2 - self.trust_radius**2) / v2
            )  # Take branch with larger alpha value

            step = x0 + alpha_p * self.v - self._x_cur

        else:
            # Move towards extremal
            move_dir = dir_extremal / dist_extremal
            step = self.trust_radius * move_dir

        # print("step", step)
        # print("step", np.linalg.norm(step) )

        return step
