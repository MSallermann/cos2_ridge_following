from rKNARR.potentials.lepsho import LEPSHOGAUSS
from rKNARR.atoms import Atoms
from ridgefollowing.plotting import plot_surface
from energy_surfaces import energy_surface
import numpy.typing as npt
import numpy as np


class rKNARRLepsho(energy_surface.EnergySurface):
    def __init__(self):
        super().__init__(2)
        self.calculator = LEPSHOGAUSS()
        self.atoms = Atoms(np.array([0, 0]), elements=None)

    def energy(self, x: npt.ArrayLike) -> float:
        self.atoms.setPosition(np.array(x))
        E, F = self.calculator.getEnergyAndForce(self.atoms)
        return E


settings = plot_surface.PlotSettings(
    lims=np.array([[0.5, 3.5], [-4, 4]]),
    plot_energy=plot_surface.ScalarPlotSettings(),
    output_data_folder="./data_rknarr_lepsho",
    outfile="plot_rknarr_lepsho.png",
)

esurf = rKNARRLepsho()
ret = plot_surface.plot(esurf, settings=settings)
