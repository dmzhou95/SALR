"""SALR: Structure-Aware Latent Representations for Incomplete Multi-view Clustering."""
from .solver import SALR, SALRConfig, VARIANTS
from .data import load_dataset, standardize_views, make_toy
from .masks import make_mask
from .metrics import clustering_metrics

__all__ = ["SALR", "SALRConfig", "VARIANTS", "load_dataset", "standardize_views", "make_toy",
           "make_mask", "clustering_metrics"]
__version__ = "1.0.0"
