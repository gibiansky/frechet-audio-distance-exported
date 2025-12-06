"""Audio embedding models for FAD calculation."""

from .vggish import VGGishCore, waveform_to_examples

__all__ = ["VGGishCore", "waveform_to_examples"]
