import ridgefollowing
from ridgefollowing.surfaces import muller_brown
import numpy as np
import pytest

epsilon = 1e6


def test_brown_muller():
    esurf = muller_brown.MullerBrownSurface()

    test_points = np.array([[-1.0, 0.5], [-2, 2], [-1, 4], [0, 0], [-4, 5]])

    for x in test_points:
        energy = esurf.energy(x)  # Can't really test, should at leat not throw

        # Compare gradient to FD
        grad_fd = esurf.fd_gradient(x)
        grad = esurf.gradient(x)

        assert np.allclose(grad, grad_fd)
