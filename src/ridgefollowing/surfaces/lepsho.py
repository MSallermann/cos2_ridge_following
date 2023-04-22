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

    def energy(self, x: npt.ArrayLike) -> float:
        rAB = x[0]
        chi = x[1]

        return (
            self.V_LEPS(rAB, self.rAC - rAB, self.rAC)
            + 2.0 * self.kc * (rAB - (self.rAC / 2.0 - chi / self.c_ho)) ** 2
        )
