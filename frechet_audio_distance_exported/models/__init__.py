"""Audio embedding models for FAD calculation."""

from .vggish import VGGishCore, waveform_to_examples
from .encodec import (
    ENCODEC_CONFIGS,
    EMBEDDING_SIZE as ENCODEC_EMBEDDING_SIZE,
    MAX_AUDIO_SECONDS as ENCODEC_MAX_AUDIO_SECONDS,
    MAX_SAMPLES_24K as ENCODEC_MAX_SAMPLES_24K,
    MAX_SAMPLES_48K as ENCODEC_MAX_SAMPLES_48K,
    preprocess_for_encodec,
    pad_to_fixed_length as pad_to_fixed_encodec_length,
    pad_to_valid_encodec_length,
)
from .clap import (
    CLAP_SAMPLE_RATE,
    CLAP_EMBEDDING_SIZE,
    MAX_AUDIO_SECONDS as CLAP_MAX_AUDIO_SECONDS,
    MAX_SAMPLES as CLAP_MAX_SAMPLES,
    preprocess_for_clap,
    pad_audio_to_max_length as pad_clap_audio_to_max_length,
)

__all__ = [
    "VGGishCore",
    "waveform_to_examples",
    "ENCODEC_CONFIGS",
    "ENCODEC_EMBEDDING_SIZE",
    "ENCODEC_MAX_AUDIO_SECONDS",
    "ENCODEC_MAX_SAMPLES_24K",
    "ENCODEC_MAX_SAMPLES_48K",
    "preprocess_for_encodec",
    "pad_to_fixed_encodec_length",
    "pad_to_valid_encodec_length",
    "CLAP_SAMPLE_RATE",
    "CLAP_EMBEDDING_SIZE",
    "CLAP_MAX_AUDIO_SECONDS",
    "CLAP_MAX_SAMPLES",
    "preprocess_for_clap",
    "pad_clap_audio_to_max_length",
]
