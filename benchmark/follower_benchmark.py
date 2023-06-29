import timeit
import numpy as np
from ridgefollowing.surfaces import muller_brown
from ridgefollowing.algorithms import cosine_follower


esurf = muller_brown.MullerBrownSurface()

follower = cosine_follower.CosineFollower(energy_surface=esurf, radius=5e-3, n_iterations_follow=200)
follower.follow( np.array([0.0,0.0]), np.array([1.0,0.0]) )
