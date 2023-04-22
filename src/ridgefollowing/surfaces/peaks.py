from ridgefollowing import energy_surface
import numpy as np
import numpy.typing as npt


class PeaksSurface(energy_surface.EnergySurface):
    def __init__(
        self,
    ):
        super().__init__(2)

    def energy(self, x: npt.ArrayLike) -> float:
        x_coord, y_coord = x
        energy = (
            3.0 * (1.0 - x_coord) ** 2 * np.exp(-(x_coord**2) - (y_coord + 1.00) ** 2)
            - 10.0
            * (x_coord / 5.00 - x_coord**3 - y_coord**5)
            * np.exp(-(x_coord**2) - y_coord**2)
            - 1.0 / 3.0 * np.exp(-((x_coord + 1.0) ** 2) - y_coord**2)
        )
        return energy
