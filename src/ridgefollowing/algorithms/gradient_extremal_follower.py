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
        trust_radius: float = 1e-5,
        mode_index: int = 0,
    ) -> None:
        super().__init__(energy_surface, n_iterations_follow)
        if not mode_index is None:
            self.mode_index = mode_index

        self.trust_radius_max = trust_radius
        self.trust_region_tolerance = 1e-5
        self.trust_region_factor = 0.5
        self.prediction_ratio = 0

        self._trust_radius = trust_radius

        self.trust_region_applicability_kappa = 1e1

        self.v = np.zeros(self.esurf.ndim)

        self.history.update(
            dict(
                trust_radius=np.zeros(shape=(self.n_iterations_follow)),
                prediction_ratio=np.zeros(shape=(self.n_iterations_follow)),
                step_type=np.zeros(shape=(self.n_iterations_follow)),
                x0=np.zeros(shape=(self.n_iterations_follow, self.esurf.ndim)),
                v=np.zeros(shape=(self.n_iterations_follow, self.esurf.ndim)),
                mode_index=np.zeros(shape=(self.n_iterations_follow)),
            )
        )

    def log_history(self):
        super().log_history()
        self.history["trust_radius"][self._iteration] = self._trust_radius
        self.history["prediction_ratio"][self._iteration] = self.prediction_ratio
        self.history["step_type"][self._iteration] = self.step_type_cur
        self.history["x0"][self._iteration] = self.x0
        self.history["v"][self._iteration] = self.v
        self.history["mode_index"][self._iteration] = self.mode_index

    def update_trust_radius(self):
        """Evaluates how close to a quadratic energy prediction the previous step was and then updates the trust radius"""

        if (
            self._iteration < 2
        ):  # If not enough data has been collected we dont't touch the trust_radius and return
            return

        E0 = self.history["E"][
            self._iteration - 1
        ]  # Energy at beginning of previous step
        G0 = self.history["G"][self._iteration - 1]  # Gradient at end of previous step
        E1 = self._E  # Energy at end of previous step
        G1 = self._G  # Gradient at end of previous step

        numerical_epsilon = 1e-16
        applicable = (
            np.abs(E1 - E0)
        ) > self.trust_region_applicability_kappa * numerical_epsilon

        if not applicable:
            # We have to increase the trust radius
            self._trust_radius /= self.trust_region_factor
            return

        # previous step
        delta = self._x_cur - self.history["x_cur"][self._iteration - 1]
        # Normalize to get direction
        step_length = np.linalg.norm(delta)
        delta /= step_length

        # linear term
        a = np.dot(G0, delta)
        # quadratic term
        b = 1 / (2.0 * step_length) * (np.dot(G1, delta) - a)

        # Energy predicted by quadratic approximation
        E_predicted = E0 + a * step_length + b * step_length**2

        self.prediction_ratio = np.abs((E_predicted - E1) / (E1 - E0))

        if (
            self.prediction_ratio > self.trust_region_tolerance
        ):  # If above tolerance, decrease trust radius
            self._trust_radius *= self.trust_region_factor
        elif (
            self.prediction_ratio < self.trust_region_factor / 10.0
            and self._trust_radius / self.trust_region_factor
            <= self.trust_radius_max  # only increase up to trust_radius_max
        ):  # If an order of magnitude below tolerance, increase trust radius
            self._trust_radius /= self.trust_region_factor

    def rolling_average_direction(self, n_steps=10):
        if self._iteration < 1:
            return np.zeros(self.esurf.ndim)

        n = min(self._iteration, n_steps)

        steps = self.history["step_cur"][self._iteration - n : self._iteration]
        avg = np.zeros(steps[0].shape)

        for s in steps:
            avg += s / np.linalg.norm(s) / n

        return avg

    # def compute_approximate_ridge_location(x_cure)

    def step_helper(self, x_cur, G, H, v):
        x0 = x_cur - np.linalg.inv(H) @ G
        x0 = x0 - np.dot(x0, v) * v  # solution of the projected newton equation

        self.x0 = x0
        dir_extremal = x0 - x_cur
        dir_extremal = dir_extremal - np.dot(dir_extremal, v) * v
        dist_extremal = np.linalg.norm(dir_extremal)

        if dist_extremal < self._trust_radius:
            # Determine alpha such that '|x0 - x_cur + alpha * v| = self._trust_radius'
            v2 = np.linalg.norm(v) ** 2
            delta = x0 - x_cur
            delta2 = np.linalg.norm(delta) ** 2
            delta_v = np.dot(v, delta)

            # Quadratic equation
            # 0 = v**2 * alpha*2 + 2*v*delta * alpha + delta**2 - r**2
            alpha_p = -delta_v / v2

            sqrt_part = np.sqrt(
                (delta_v / v2) ** 2 - (delta2 - self._trust_radius**2) / v2
            )  # Take branch with larger alpha value
            if np.isnan(sqrt_part):
                sqrt_part = 0.0
            alpha_p += sqrt_part

            step = x0 + alpha_p * v - x_cur
            self.step_type_cur = 0

        else:
            # Move towards extremal
            move_dir = dir_extremal / dist_extremal
            step = self._trust_radius * move_dir
            self.step_type_cur = 1

        return step

    def determine_step(self):
        self.update_trust_radius()

        if self._iteration == 0:
            dir_prev = (
                self._d0
            )  # The initial value of _step_cur is d0 from the function call of self.follow()
            self.v = dir_prev
        else:
            dir_prev = self.rolling_average_direction(200)

        self._H = self.esurf.hessian(self._x_cur)
        evals, evecs = modes.lowest_n_modes(self._H, n=2)

        # Update index of current mode, via maximum overlap
        overlaps = [np.abs(np.dot(self.v, evecs[:, i])) for i in range(len(evals))]
        self.mode_index = np.argmax(overlaps)
        self.v = evecs[:, self.mode_index]

        # Mode should point into one consistent direction
        if np.dot(dir_prev, self.v) < 0:
            self.v *= -1

        # Predictor step
        step_pr = self.step_helper(self._x_cur, self._G, self._H, self.v)

        G_pr = self.esurf.gradient(self._x_cur + step_pr)
        H_pr = self.esurf.hessian(self._x_cur + step_pr)
        evals_pr, evecs_pr = modes.lowest_n_modes(H_pr, n=2)

        overlaps_pr = [
            np.abs(np.dot(self.v, evecs_pr[:, i])) for i in range(len(evals_pr))
        ]
        mode_index_pr = np.argmax(overlaps_pr)
        v_pr = evecs_pr[:, mode_index_pr]

        # Mode should point into one consistent direction
        if np.dot(dir_prev, v_pr) < 0:
            v_pr *= -1

        step_corrector = self.step_helper(self._x_cur + step_pr, G_pr, H_pr, v_pr)

        return 0.5 * (step_pr + step_corrector)
