"""
frechet-audio-distance-exported: Lightweight FAD using exported PyTorch models.

This package provides Frechet Audio Distance (FAD) calculation with minimal
dependencies by using exported PyTorch models instead of torch.hub.
"""

from .fad import FrechetAudioDistance

__version__ = "0.1.0"
__all__ = ["FrechetAudioDistance"]
