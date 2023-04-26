from ridgefollowing import energy_surface
import numpy as np
import numpy.typing as npt


class LepsSurface(energy_surface.EnergySurface):
    """The LEPS surface. See G. Henkelman, G. J ́ohannesson, H. J ́onsson. Methods for Finding Saddle Points and Mini- mum Energy Paths, In ”Theoretical Methods in Condensed Phase Chemistry”, edited by S.D. Schwartz, pages 269-30"""

    def __init__(
        self,
    ):
        super().__init__(ndim=2)
        self.a = 0.05
        self.b = 0.80
        self.c = 0.05
        self.dAB = 4.746
        self.dBC = 4.746
        self.dAC = 3.445
        self.r0 = 0.742
        self.alpha = 1.942

        self.rAC = 3.742  # the third parameter of the leps surface is rAC is fixed

        self.J_matrix = np.array(
            [[1.0, -0.5, -0.5], [-0.5, 1.0, -0.5], [-0.5, -0.5, 1.0]]
        )

        self.J_vec = np.zeros(3)

    def Q(self, r: float, d: float):
        """Q helper function

        Args:
            r (_type_): _description_
            d (_type_): _description_

        Returns:
            _type_: _description_
        """
        return (
            0.5
            * d
            * (
                1.5 * np.exp(-2.0 * self.alpha * (r - self.r0))
                - np.exp(-self.alpha * (r - self.r0))
            )
        )

    def J(self, r: float, d: float):
        """J helper function

        Args:
            r (_type_): _description_
            d (_type_): _description_

        Returns:
            _type_: _description_
        """
        return (
            0.25
            * d
            * (
                np.exp(-2.0 * self.alpha * (r - self.r0))
                - 6.0 * np.exp(-self.alpha * (r - self.r0))
            )
        )

    def V_LEPS(self, rAB, rBC, rAC):
        """The leps potential function

        Args:
            rAB (_type_): _description_
            rBC (_type_): _description_
            rAC (_type_): _description_

        Returns:
            _type_: _description_
        """
        Q_contribution = (
            self.Q(rAB, self.dAB) / (1.0 + self.a)
            + self.Q(rBC, self.dBC) / (1.0 + self.b)
            + self.Q(rAC, self.dAC) / (1.0 + self.c)
        )

        self.J_vec[0] = self.J(rAB, self.dAB) / (1.0 + self.a)
        self.J_vec[1] = self.J(rBC, self.dBC) / (1.0 + self.b)
        self.J_vec[2] = self.J(rAC, self.dAC) / (1.0 + self.c)

        J_contribution = np.sqrt(
            np.dot(self.J_vec, np.matmul(self.J_matrix, self.J_vec))
        )

        return Q_contribution - J_contribution

    def energy(self, x: npt.ArrayLike) -> float:
        """LEPS with fixed value of rAC"""
        rAB = x[0]
        rBC = x[1]
        return self.V_LEPS(rAB, rBC, self.rAC)
