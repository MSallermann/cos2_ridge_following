import numpy as np
from energy_surfaces.surfaces import muller_brown
from ridgefollowing.algorithms import cosine_follower

esurf = muller_brown.MullerBrownSurface()
follower = cosine_follower.CosineFollower(
    energy_surface=esurf, radius=5e-3, n_iterations_follow=200
)
follower.tolerance = 1e-4
follower.tolerance_grad = 1e-10
follower.follow(np.array([0.0, 0.0]), np.array([1.0, 0.0]))
