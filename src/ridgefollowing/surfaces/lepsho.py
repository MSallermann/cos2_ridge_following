from ridgefollowing.surfaces import leps
import numpy as np
import numpy.typing as npt


class LepsHOSurface(leps.LepsSurface):
    """The LEPS surface with an additional contribution from an harmonic oscillator, modelling a condensed phase.
    See G. Henkelman, G. J ́ohannesson, H. Jonsson. Methods for Finding Saddle Points and Minimum Energy Paths, In ”Theoretical Methods in Condensed Phase Chemistry”, edited by S.D. Schwartz, pages 269-30
    """

    def __init__(self):
        super().__init__()
        self.kc = 0.2025
        self.c_ho = 1.154
        self.rAC = 3.742

    def energy_harm(self, x: npt.ArrayLike) -> float:
        return 2.0 * self.kc * (x[0] - (self.rAC / 2.0 - x[1] / self.c_ho)) ** 2

    def grad_harm(self, x: npt.ArrayLike) -> npt.NDArray:
        grad = np.zeros(2)
        grad[0] = 4.0 * self.kc * (x[0] - (self.rAC / 2.0 - x[1] / self.c_ho))
        grad[1] = (
            4.0 * self.kc * (x[0] - (self.rAC / 2.0 - x[1] / self.c_ho)) / self.c_ho
        )
        return grad

    def energy(self, x: npt.ArrayLike) -> float:
        rAB = x[0]
        chi = x[1]

        return self.V_LEPS(rAB, self.rAC - rAB, self.rAC) + self.energy_harm(x)

    def gradient(self, x: npt.ArrayLike) -> npt.NDArray:
        rAB = x[0]

        grad = np.zeros(2)
        grad_leps = self.gradient_V_LEPS(rAB, self.rAC - rAB, self.rAC)
        grad[0] = grad_leps[0] - grad_leps[1]

        grad += self.grad_harm(x)

        return grad
