import timeit
import numpy as np
from ridgefollowing.surfaces import muller_brown, leps, lepsho, lepshogauss, cosine_ssbench


def benchmark(fun, number=10000, x=np.array([1.0, 1.0]), prints=True):
    tmp = fun(x)
    time = timeit.timeit(lambda: fun(x), number=number) / number
    if prints:
        print(f"Time per function call: {time*1e6:.1f} us")
    return time


def benchmark_surface(esurf):
    print(f"Benchmarking {esurf}")
    print("Energy")
    benchmark(esurf.energy)
    print("Gradient")
    benchmark(esurf.gradient)
    print("Hessian")
    benchmark(esurf.hessian)


benchmark_surface(muller_brown.MullerBrownSurface())
benchmark_surface(leps.LepsSurface())
benchmark_surface(lepsho.LepsHOSurface())
benchmark_surface(lepshogauss.LepsHOGaussSurface())
benchmark_surface(cosine_ssbench.CosineSSBENCH())

