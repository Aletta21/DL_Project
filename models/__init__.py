"""Model registries."""

from .fcnn import IsoformPredictor
from .residual_model import ResidualIsoformPredictor
from .transformer import TransformerIsoformPredictor

__all__ = [
    "IsoformPredictor",
    "ResidualIsoformPredictor",
    "TransformerIsoformPredictor",
]
