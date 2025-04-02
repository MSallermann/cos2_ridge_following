from energy_surfaces import energy_surface
from ridgefollowing.algorithms import ridgefollower, modes
import numpy as np
import numpy.typing as npt


class GradientExtremalFollower(ridgefollower.RidgeFollower):
    def __init__(
        self,
        energy_surface: energy_surface.EnergySurface,
        n_iterations_follow: int = 100,
        trust_radius: float = 1e-5,
    ) -> None:
        super().__init__(energy_surface, n_iterations_follow)

        self.mode_index: npt.NDArray = np.zeros(self.esurf.ndim)

        self.trust_radius_max: float = trust_radius
        self.trust_radius_min: float = 1e-1
        self.trust_region_tolerance: float = 1e-4
        self.trust_region_factor: float = 0.5
        self.prediction_ratio: float = 0.0

        self.step_type_cur: int = 0
        self._trust_radius: float = trust_radius

        self.trust_region_applicability_kappa: float = 1e1

        self.v: npt.NDArray = np.zeros(self.esurf.ndim)
        self._H: npt.NDArray = np.zeros((self.esurf.ndim, self.esurf.ndim))

    def setup_history(self):
        super().setup_history()
        self.history.update(
            dict(
                trust_radius=np.zeros(shape=(self.n_iterations_follow)),
                prediction_ratio=np.zeros(shape=(self.n_iterations_follow)),
                ridge_dist=np.zeros(shape=(self.n_iterations_follow)),
                step_type=np.zeros(shape=(self.n_iterations_follow)),
                x0=np.zeros(shape=(self.n_iterations_follow, self.esurf.ndim)),
                v=np.zeros(shape=(self.n_iterations_follow, self.esurf.ndim)),
                eval=np.zeros(shape=(self.n_iterations_follow)),
                eval2=np.zeros(shape=(self.n_iterations_follow)),
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
        self.history["eval"][self._iteration] = self.cur_eval
        self.history["eval2"][self._iteration] = self.cur_eval2
        self.history["mode_index"][self._iteration] = self.mode_index
        self.history["ridge_dist"][self._iteration] = self.ridge_dist

    def update_trust_radius(self, x0, x1, G0, G1, E0, E1):
        """Evaluates how close to a quadratic energy prediction the previous step was and then updates the trust radius"""

        numerical_epsilon = 1e-16
        applicable = (
            np.abs(E1 - E0)
        ) > self.trust_region_applicability_kappa * numerical_epsilon

        if not applicable:
            # We have to increase the trust radius
            if self._trust_radius / self.trust_region_factor <= self.trust_radius_max:
                self._trust_radius /= self.trust_region_factor
            return

        # previous step
        delta = x1 - x0
        # Normalize to get direction
        step_length = np.linalg.norm(delta)
        delta /= step_length

        # linear term
        a = np.dot(G0, delta)
        # quadratic term
        b = 1.0 / (2.0 * step_length) * (np.dot(G1, delta) - a)

        # Energy predicted by quadratic approximation
        E_predicted = E0 + a * step_length + b * step_length**2

        self.prediction_ratio = np.abs((E_predicted - E1) / (E1 - E0))

        if (
            self.prediction_ratio > self.trust_region_tolerance
            and self._trust_radius * self.trust_region_factor >= self.trust_radius_min
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

    def compute_approximate_ridge_location(self, x_cur, G, H, v):
        """Compute the location of the ridge from a local quadratic approximation and returns
        the location, the distance and the direction towards the ridge
        """

        x0 = -np.linalg.inv(H) @ G
        x0 = x0 - np.dot(x0, v) * v  # solution of the projected newton equation

        dir_extremal = x0
        dist_extremal = np.linalg.norm(dir_extremal)

        return x_cur + x0, dist_extremal, dir_extremal

    def compute_v(self, H, v_prev):
        evals, evecs = modes.lowest_n_modes(H, n=2)
        # Update index of current mode, via maximum overlap
        overlaps = [
            np.abs(np.dot(v_prev, evecs[:, i])) for i in range(len(evals))
        ]  # Overlap with previous evec
        mode_index = np.argmax(overlaps)
        v = evecs[:, mode_index]
        # Mode should point into one consistent direction
        if np.dot(v_prev, v) < 0:
            v *= -1
        self.mode_index = mode_index

        self.cur_eval = evals[mode_index]
        self.cur_eval2 = evals[(mode_index + 1) % self.esurf.ndim]

        return v

    def compute_quantities(self, x):
        self._E = self.esurf.energy(x)
        self._G = self.esurf.gradient(x)
        self._H = self.esurf.hessian(x)
        self.v = self.compute_v(self._H, self._step_prev)

    def move_towards_ridge(self):
        (
            self.x0,
            self.ridge_dist,
            self.ridge_dir,
        ) = self.compute_approximate_ridge_location(
            self._x_cur, self._G, self._H, self.v
        )

        if self.ridge_dist > self._trust_radius:
            self.step_type_cur = 1
            print(f"Lost ridge: ridge_dist = {self.ridge_dist}")
            print(f"Lost ridge: x0 = {self.x0}")
            print(f"Lost ridge: x = {self._x_cur}")
            print(f"Lost ridge: v = {self.v}")

        else:
            self.step_type_cur = 0
            return

        while self.ridge_dist > self._trust_radius:
            # Move towards ridge
            step = self._trust_radius * self.ridge_dir / self.ridge_dist
            self._x_cur += step

            G_prev = self._G
            E_prev = self._E

            self.compute_quantities(self._x_cur)

            (
                self.x0,
                self.ridge_dist,
                self.ridge_dir,
            ) = self.compute_approximate_ridge_location(
                self._x_cur, self._G, self._H, self.v
            )

            self.update_trust_radius(
                x0=self._x_cur - step,
                x1=self._x_cur,
                G0=G_prev,
                G1=self._G,
                E0=E_prev,
                E1=self._E,
            )

        return self._x_cur

    def step_along_ridge(self, x_cur, x0, v):
        # Determine alpha such that '|x0 - x_cur + alpha * v| = self._trust_radius'
        # Only works if the distance to the ridge is less than trust_radius

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

        return step

    def determine_step(self):
        x_cur_save = self._x_cur.copy()

        if (
            self._iteration >= 2
        ):  # If not enough data has been collected we dont't touch the trust_radius and return
            E0 = self.history["E"][
                self._iteration - 1
            ]  # Energy at beginning of previous step
            G0 = self.history["G"][
                self._iteration - 1
            ]  # Gradient at end of previous step
            E1 = self._E  # Energy at end of previous step
            G1 = self._G  # Gradient at end of previous step
            x0 = self.history["x_cur"][self._iteration - 1]
            x1 = self._x_cur
            self.update_trust_radius(x0, x1, G0, G1, E0, E1)

        self.compute_quantities(self._x_cur)
        self.move_towards_ridge()

        if self.step_type_cur == 1:
            step = self._x_cur - x_cur_save
            self._x_cur = x_cur_save
            return step

        # Predictor step
        step_pr = self.step_along_ridge(self._x_cur, self.x0, self.v)

        G_pr = self.esurf.gradient(self._x_cur + step_pr)
        H_pr = self.esurf.hessian(self._x_cur + step_pr)

        G_final = 0.5 * (self._G + G_pr)
        H_final = 0.5 * (self._H + H_pr)

        v_final = self.compute_v(H_final, self._step_prev)

        x0_corrector, _, _ = self.compute_approximate_ridge_location(
            self._x_cur, G_final, H_final, v_final
        )

        step_corrector = self.step_along_ridge(self._x_cur, x0_corrector, v_final)

        # Do some cleanup since the parent class assumes that determine_step does not mess with x_cur
        # step = self._x_cur + step_corrector - x_cur_save
        step = self._x_cur + step_corrector - x_cur_save
        self._x_cur = x_cur_save
        # step = step_pr
        # step = self._trust_radius * self.v
        return step
