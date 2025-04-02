from energy_surfaces import energy_surface
from ridgefollowing.algorithms import modes
import numpy.typing as npt
import numpy as np
import numdifftools as nd
from scipy.optimize import minimize
from typing import Optional, List


class Minimizer:
    def __init__(
        self,
        energy_surface: energy_surface.EnergySurface,
        tolerance: Optional[float] = None,
        maxiter: int = 1000,
        n_log: Optional[int] = 1,
    ) -> None:
        self.esurf: energy_surface.EnergySurface = energy_surface
        self.tolerance: float = tolerance
        self.maxiter: int = maxiter
        self.energy: Optional[float] = None

        self.n_log: Optional[int] = n_log

        self.trajectory: List[npt.NDArray] = []

    def minimize_energy(self, x0: npt.NDArray):
        assert x0.shape == (self.esurf.ndim,)

        cb = None
        if self.n_log:

            def cb(x):
                cb.counter += 1
                if cb.counter % self.n_log == 0:
                    self.trajectory.append(x)

            cb.counter = 0

        res = minimize(
            self.esurf.energy,
            x0,
            method="CG",
            jac=self.esurf.gradient,
            tol=self.tolerance,
            options=dict(maxiter=self.maxiter, disp=False),
            callback=cb,
        )
        assert res.success
        self.energy = res.fun
        return res.x
