"""Model registries."""

from .fcnn import IsoformPredictor
from .residual_model import ResidualIsoformPredictor
from .transformer import TransformerIsoformPredictor
from .Vae import VAE
from .Vae import vae_loss   

__all__ = [
    "IsoformPredictor",
    "ResidualIsoformPredictor",
    "TransformerIsoformPredictor",
    "VAE",
    "vae_loss",
]
