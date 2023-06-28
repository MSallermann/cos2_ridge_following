from ridgefollowing import energy_surface
import numpy as np
import numpy.typing as npt
from numba import njit, int32, float64, typed
from numba.experimental import jitclass

@jitclass(
    [
        ("a",   float64),
        ("b",   float64),
        ("c",   float64),
        ("dAB", float64),
        ("dBC", float64),
        ("dAC", float64),
        ("r0", float64),
        ("alpha", float64),
        ("rAC", float64),
        ("J_matrix", float64[:,:]),
        ("J_vec", float64[:]),
        ("diff_J_vec", float64[:]),
        ("diff2_J_vec", float64[:]),
        ("ndim", int32)
    ]
)
class LepsSurfaceHelper:
    def __init__(self, a ,b ,c ,dAB ,dBC ,dAC ,r0 ,alpha ,rAC, J_matrix, J_vec, diff_J_vec, diff2_J_vec, ndim ):
        self.a = a
        self.b = b
        self.c = c
        self.dAB = dAB
        self.dBC = dBC
        self.dAC = dAC
        self.r0 = r0
        self.alpha = alpha
        self.rAC = rAC
        self.J_matrix = J_matrix
        self.J_vec = J_vec
        self.diff_J_vec = diff_J_vec
        self.diff2_J_vec = diff2_J_vec
        self.ndim = ndim

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

        self.helper = LepsSurfaceHelper(
            a = self.a,
            b = self.b,
            c = self.c,
            dAB = self.dAB,
            dBC = self.dBC,
            dAC = self.dAC,
            r0 = self.r0,
            alpha = self.alpha,
            rAC = self.rAC,
            J_matrix = self.J_matrix,
            J_vec=self.J_vec,
            diff_J_vec=self.diff_J_vec,
            diff2_J_vec=self.diff2_J_vec,
            ndim = self.ndim
        )


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
    
    @staticmethod
    @njit
    def V_LEPS(rAB, rBC, rAC, params):
        """The leps potential function

        Args:
            rAB (_type_): _description_
            rBC (_type_): _description_
            rAC (_type_): _description_

        Returns:
            _type_: _description_
        """
        Q_contribution = (
            params.Q(rAB, params.dAB) / (1.0 + params.a)
            + params.Q(rBC, params.dBC) / (1.0 + params.b)
            + params.Q(rAC, params.dAC) / (1.0 + params.c)
        )

        params.J_vec[0] = params.J(rAB, params.dAB) / (1.0 + params.a)
        params.J_vec[1] = params.J(rBC, params.dBC) / (1.0 + params.b)
        params.J_vec[2] = params.J(rAC, params.dAC) / (1.0 + params.c)

        J_contribution = np.sqrt(np.dot(params.J_vec, params.J_matrix @ params.J_vec))

        return Q_contribution - J_contribution

    def energy(self, x: npt.ArrayLike) -> float:
        """LEPS with fixed value of rAC"""
        rAB = x[0]
        rBC = x[1]
        return LepsSurface.V_LEPS(rAB, rBC, self.rAC, self.helper)

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

    @staticmethod
    @njit
    def gradient_V_LEPS(rAB, rBC, rAC, params):
        """Grad of the leps potential function

        Args:
            rAB (_type_): _description_
            rBC (_type_): _description_
            rAC (_type_): _description_

        Returns:
            _type_: _description_
        """
        grad = np.zeros(3)
        grad[0] = params.diff_Q(rAB, params.dAB) / (1.0 + params.a)
        grad[1] = params.diff_Q(rBC, params.dBC) / (1.0 + params.b)
        grad[2] = params.diff_Q(rAC, params.dAC) / (1.0 + params.c)

        params.J_vec[0] = params.J(rAB, params.dAB) / (1.0 + params.a)
        params.J_vec[1] = params.J(rBC, params.dBC) / (1.0 + params.b)
        params.J_vec[2] = params.J(rAC, params.dAC) / (1.0 + params.c)

        params.diff_J_vec[0] = params.diff_J(rAB, params.dAB) / (1.0 + params.a)
        params.diff_J_vec[1] = params.diff_J(rBC, params.dBC) / (1.0 + params.b)
        params.diff_J_vec[2] = params.diff_J(rAC, params.dAC) / (1.0 + params.c)

        J_contribution = np.sqrt(np.dot(params.J_vec, params.J_matrix @ params.J_vec))

        J_grad = 1.0 / J_contribution * params.J_matrix @ params.J_vec * params.diff_J_vec

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
        return LepsSurface.gradient_V_LEPS(rAB, rBC, self.rAC, self.helper)[:2]

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

    @staticmethod
    @njit
    def hessian_V_LEPS(rAB, rBC, rAC, params) -> npt.NDArray:
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
                params.diff2_Q(rAB, params.dAB) / (1.0 + params.a),
                params.diff2_Q(rBC, params.dBC) / (1.0 + params.b),
                params.diff2_Q(rAC, params.dAC) / (1.0 + params.c),
            ]
        )

        params.J_vec[0] = params.J(rAB, params.dAB) / (1.0 + params.a)
        params.J_vec[1] = params.J(rBC, params.dBC) / (1.0 + params.b)
        params.J_vec[2] = params.J(rAC, params.dAC) / (1.0 + params.c)

        params.diff_J_vec[0] = params.diff_J(rAB, params.dAB) / (1.0 + params.a)
        params.diff_J_vec[1] = params.diff_J(rBC, params.dBC) / (1.0 + params.b)
        params.diff_J_vec[2] = params.diff_J(rAC, params.dAC) / (1.0 + params.c)

        params.diff2_J_vec[0] = params.diff2_J(rAB, params.dAB) / (1.0 + params.a)
        params.diff2_J_vec[1] = params.diff2_J(rBC, params.dBC) / (1.0 + params.b)
        params.diff2_J_vec[2] = params.diff2_J(rAC, params.dAC) / (1.0 + params.c)

        J_contribution = np.sqrt(np.dot(params.J_vec, params.J_matrix @ params.J_vec))
        J_grad = 1.0 / J_contribution * params.J_matrix @ params.J_vec * params.diff_J_vec

        hessian_j = (
            1.0
            / J_contribution
            * params.J_matrix
            @ params.J_vec
            * np.diag(params.diff2_J_vec)
        )

        for k in range(3):
            for l in range(3):
                hessian_j[k, l] -= (
                    J_grad[l]
                    / J_contribution**2
                    * (params.J_matrix @ params.J_vec)[k]
                    * params.diff_J_vec[k]
                )
                hessian_j[k, l] += (
                    1.0
                    / J_contribution
                    * params.J_matrix[k, l]
                    * params.diff_J_vec[l]
                    * params.diff_J_vec[k]
                )

        return hessian_q - hessian_j

    def hessian(self, x: npt.ArrayLike) -> npt.NDArray:
        rAB = x[0]
        rBC = x[1]
        return LepsSurface.hessian_V_LEPS(rAB, rBC, self.rAC, self.helper)[:2, :2]
