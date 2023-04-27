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
        self.diff_J_vec = np.zeros(3)
        self.diff2_J_vec = np.zeros(3)

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

        J_contribution = np.sqrt(np.dot(self.J_vec, self.J_matrix @ self.J_vec))

        return Q_contribution - J_contribution

    def energy(self, x: npt.ArrayLike) -> float:
        """LEPS with fixed value of rAC"""
        rAB = x[0]
        rBC = x[1]
        return self.V_LEPS(rAB, rBC, self.rAC)

    def diff_Q(self, r: float, d: float):
        """derivative of Q helper function wrt r

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
                -3.0 * self.alpha * np.exp(-2.0 * self.alpha * (r - self.r0))
                + self.alpha * np.exp(-self.alpha * (r - self.r0))
            )
        )

    def diff_J(self, r: float, d: float):
        """derivative of J helper function wrt r

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
                -2.0 * self.alpha * np.exp(-2.0 * self.alpha * (r - self.r0))
                + 6.0 * self.alpha * np.exp(-self.alpha * (r - self.r0))
            )
        )

    def gradient_V_LEPS(self, rAB, rBC, rAC):
        """Grad of the leps potential function

        Args:
            rAB (_type_): _description_
            rBC (_type_): _description_
            rAC (_type_): _description_

        Returns:
            _type_: _description_
        """
        grad = np.zeros(3)
        grad[0] = self.diff_Q(rAB, self.dAB) / (1.0 + self.a)
        grad[1] = self.diff_Q(rBC, self.dBC) / (1.0 + self.b)
        grad[2] = self.diff_Q(rAC, self.dAC) / (1.0 + self.c)

        self.J_vec[0] = self.J(rAB, self.dAB) / (1.0 + self.a)
        self.J_vec[1] = self.J(rBC, self.dBC) / (1.0 + self.b)
        self.J_vec[2] = self.J(rAC, self.dAC) / (1.0 + self.c)

        self.diff_J_vec[0] = self.diff_J(rAB, self.dAB) / (1.0 + self.a)
        self.diff_J_vec[1] = self.diff_J(rBC, self.dBC) / (1.0 + self.b)
        self.diff_J_vec[2] = self.diff_J(rAC, self.dAC) / (1.0 + self.c)

        J_contribution = np.sqrt(np.dot(self.J_vec, self.J_matrix @ self.J_vec))

        J_grad = 1.0 / J_contribution * self.J_matrix @ self.J_vec * self.diff_J_vec

        grad -= J_grad
        return grad

    def gradient(self, x: npt.ArrayLike) -> npt.NDArray:
        """Gradient of LEPS potential

        Args:
            x (npt.ArrayLike): point in configuration space

        Returns:
            npt.NDArray: the gradient
        """

        rAB = x[0]
        rBC = x[1]
        return self.gradient_V_LEPS(rAB, rBC, self.rAC)[:2]

    def diff2_Q(self, r: float, d: float):
        """second derivative of Q helper function wrt r

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
                6.0 * self.alpha**2 * np.exp(-2.0 * self.alpha * (r - self.r0))
                - self.alpha**2 * np.exp(-self.alpha * (r - self.r0))
            )
        )

    def diff2_J(self, r: float, d: float):
        """second derivative of J helper function wrt r

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
                4.0 * self.alpha**2 * np.exp(-2.0 * self.alpha * (r - self.r0))
                - 6.0 * self.alpha**2 * np.exp(-self.alpha * (r - self.r0))
            )
        )

    def hessian_V_LEPS(self, rAB, rBC, rAC) -> npt.NDArray:
        """Hessian of the leps potential function

        Args:
            rAB (_type_): _description_
            rBC (_type_): _description_
            rAC (_type_): _description_

        Returns:
            _type_: _description_
        """

        hessian_q = np.diag(
            [
                self.diff2_Q(rAB, self.dAB) / (1.0 + self.a),
                self.diff2_Q(rBC, self.dBC) / (1.0 + self.b),
                self.diff2_Q(rAC, self.dAC) / (1.0 + self.c),
            ]
        )

        self.J_vec[0] = self.J(rAB, self.dAB) / (1.0 + self.a)
        self.J_vec[1] = self.J(rBC, self.dBC) / (1.0 + self.b)
        self.J_vec[2] = self.J(rAC, self.dAC) / (1.0 + self.c)

        self.diff_J_vec[0] = self.diff_J(rAB, self.dAB) / (1.0 + self.a)
        self.diff_J_vec[1] = self.diff_J(rBC, self.dBC) / (1.0 + self.b)
        self.diff_J_vec[2] = self.diff_J(rAC, self.dAC) / (1.0 + self.c)

        self.diff2_J_vec[0] = self.diff2_J(rAB, self.dAB) / (1.0 + self.a)
        self.diff2_J_vec[1] = self.diff2_J(rBC, self.dBC) / (1.0 + self.b)
        self.diff2_J_vec[2] = self.diff2_J(rAC, self.dAC) / (1.0 + self.c)

        J_contribution = np.sqrt(np.dot(self.J_vec, self.J_matrix @ self.J_vec))
        J_grad = 1.0 / J_contribution * self.J_matrix @ self.J_vec * self.diff_J_vec

        hessian_j = (
            1.0
            / J_contribution
            * self.J_matrix
            @ self.J_vec
            * np.diag(self.diff2_J_vec)
        )

        for k in range(3):
            for l in range(3):
                hessian_j[k, l] -= (
                    J_grad[l]
                    / J_contribution**2
                    * (self.J_matrix @ self.J_vec)[k]
                    * self.diff_J_vec[k]
                )
                hessian_j[k, l] += (
                    1.0
                    / J_contribution
                    * self.J_matrix[k, l]
                    * self.diff_J_vec[l]
                    * self.diff_J_vec[k]
                )

        return hessian_q - hessian_j

    def hessian(self, x: npt.ArrayLike) -> npt.NDArray:
        rAB = x[0]
        rBC = x[1]
        return self.hessian_V_LEPS(rAB, rBC, self.rAC)[:2, :2]
