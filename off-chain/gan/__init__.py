"""
GAN module for generating synthetic blockchain data.
Contains models, training, and generation utilities.
"""

from .models import Generator, Discriminator
from .training import WGANGPTrainer
from .generation import SyntheticDataGenerator

__all__ = [
    'Generator',
    'Discriminator',
    'WGANGPTrainer',
    'SyntheticDataGenerator'
] 