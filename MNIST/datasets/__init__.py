import logging

from .base import IntractableLikelihoodError, DatasetNotAvailableError
from .spherical_simulator import SphericalGaussianSimulator
from .sphere_simulator import SphereSimulator
from .torus_simulator import TorusSimulator
from .hyperboloid_simulator import HyperboloidSimulator
from .conditional_spherical_simulator import ConditionalSphericalGaussianSimulator
from .images import ImageNetLoader, CelebALoader, FFHQStyleGAN2DLoader, IMDBLoader, FFHQStyleGAN64DLoader #, MNISTLoader
from .collider import WBFLoader, WBF2DLoader, WBF40DLoader
from .polynomial_surface_simulator import PolynomialSurfaceSimulator
from .lorenz import LorenzSimulator
from .thin_spiral import ThinSpiralSimulator
from .swiss_roll_simulator import SwissRollSimulator
from .thin_disk import ThinDiskSimulator
from .von_Mises_on_circle import VonMisesSimulator
from .mixture_on_sphere import MixtureSphereSimulator
from .utils import NumpyDataset
from .mnist_simulator import MNISTSimulator

logger = logging.getLogger(__name__)


SIMULATORS = ["mnist","power", "hyperboloid", "torus","sphere", "spherical_gaussian","von_Mises_circle", "thin_spiral", "swiss_roll", "sphere_mixture","sphere_bigcheckerboard","thin_disk", "conditional_spherical_gaussian", "lhc", "lhc40d", "lhc2d", "imagenet", "celeba", "gan2d", "gan64d", "lorenz", "imdb"]


def load_simulator(args):
    assert args.dataset in SIMULATORS
    if args.dataset == "power":
        simulator = PolynomialSurfaceSimulator(filename=args.dir + "/experiments/data/samples/power/manifold.npz")
    elif args.dataset == "torus":
        simulator = TorusSimulator(latent_distribution=args.latent_distribution,noise_type=args.noise_type)
    elif args.dataset == "hyperboloid":
        simulator = HyperboloidSimulator(latent_distribution=args.latent_distribution,noise_type=args.noise_type)
    elif args.dataset == "spherical_gaussian":
        simulator = SphericalGaussianSimulator(args.truelatentdim, args.datadim, epsilon=args.epsilon)
    elif args.dataset == "conditional_spherical_gaussian":
        simulator = ConditionalSphericalGaussianSimulator(args.truelatentdim, args.datadim, epsilon=args.epsilon)
    elif args.dataset == "thin_spiral":    
        simulator = ThinSpiralSimulator(args.truelatentdim, args.datadim, epsilon=args.epsilon)
    elif args.dataset == "swiss_roll":    
        simulator = SwissRollSimulator(args.truelatentdim, args.datadim, epsilon=args.epsilon, latent_distribution=args.latent_distribution,noise_type=args.noise_type)
    elif args.dataset == "thin_disk":    
        simulator = ThinDiskSimulator(args.truelatentdim, args.datadim, epsilon=args.epsilon)
    elif args.dataset == "von_Mises_circle":
        simulator = VonMisesSimulator(args.truelatentdim, args.datadim, epsilon=args.epsilon)
    elif args.dataset == "sphere":
        simulator = SphereSimulator(args.truelatentdim,args.datadim,kappa=6.0,epsilon=0.,latent_distribution=args.latent_distribution,noise_type=args.noise_type)
    elif args.dataset == "lhc":
        simulator = WBFLoader()
    elif args.dataset == "lhc2d":
        simulator = WBF2DLoader()
    elif args.dataset == "lhc40d":
        simulator = WBF40DLoader()
    elif args.dataset == "imagenet":
        simulator = ImageNetLoader()
    elif args.dataset == "celeba":
        simulator = CelebALoader()
    elif args.dataset == "gan2d":
        simulator = FFHQStyleGAN2DLoader()
    elif args.dataset == "gan64d":
        simulator = FFHQStyleGAN64DLoader()
    elif args.dataset == "lorenz":
        simulator = LorenzSimulator()
    elif args.dataset == "imdb":
        simulator = IMDBLoader()
    elif args.dataset == "mnist":
        simulator = MNISTSimulator(latent_dim = args.truelatentdim, data_dim = 784, noise_type = args.noise_type, sig2 = args.sig2)
    else:
        raise ValueError("Unknown dataset {}".format(args.dataset))

    args.datadim = simulator.data_dim()
    return simulator
