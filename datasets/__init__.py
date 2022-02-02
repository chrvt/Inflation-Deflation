import logging

from .base import IntractableLikelihoodError, DatasetNotAvailableError
from .sphere_simulator import SphereSimulator
from .torus_simulator import TorusSimulator
from .hyperboloid_simulator import HyperboloidSimulator
from .thin_spiral_simulator import ThinSpiralSimulator
from .swiss_roll_simulator import SwissRollSimulator
from .von_Mises_on_circle import VonMisesSimulator
from .spheroid_simulator import SpheRoidSimulator
from .stiefel_simulator import StiefelSimulator
from .utils import NumpyDataset

logger = logging.getLogger(__name__)


SIMULATORS = ["hyperboloid", "torus","sphere", "swiss_roll", "thin_spiral", "two_thin_spirals", "spheroid", "stiefel"]


def load_simulator(args):
    assert args.dataset in SIMULATORS
    if args.dataset == "torus":
        simulator = TorusSimulator(latent_distribution=args.latent_distribution,noise_type=args.noise_type)
    elif args.dataset == "hyperboloid":
        simulator = HyperboloidSimulator(latent_distribution=args.latent_distribution,noise_type=args.noise_type)
    elif args.dataset == "thin_spiral":    
        simulator = ThinSpiralSimulator(latent_distribution=args.latent_distribution,noise_type=args.noise_type)
    elif args.dataset == "swiss_roll":    
        simulator = SwissRollSimulator(epsilon=args.sig2, latent_distribution=args.latent_distribution,noise_type=args.noise_type)
    elif args.dataset == "von_Mises_circle":
        simulator = VonMisesSimulator(args.truelatentdim, args.datadim, epsilon=args.sig2)
    elif args.dataset == "sphere":
        simulator = SphereSimulator(kappa=6.0,epsilon=0.,latent_distribution=args.latent_distribution,noise_type=args.noise_type)
    elif args.dataset == "spheroid":
        simulator = SpheRoidSimulator(latent_distribution=args.latent_distribution,noise_type=args.noise_type)
    elif args.dataset == "stiefel":
        simulator = StiefelSimulator(latent_distribution=args.latent_distribution,noise_type=args.noise_type)
    else:
        raise ValueError("Unknown dataset {}".format(args.dataset))

    args.datadim = simulator.data_dim()
    return simulator
