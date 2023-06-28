import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


folder = Path(
    "/home/moritz/Coding/cos2_ridge_following/scripts/test_lepshogauss/cosine_bifurcation_walks/bifurcation3/history/"
)


def plot_file(file):
    data = np.load(folder / file)
    plt.plot(data, label=file)


plot_file("ridge_width_bw.npy")
plot_file("ridge_width.npy")
plot_file("ridge_width_fw.npy")


plt.legend()
plt.show()
