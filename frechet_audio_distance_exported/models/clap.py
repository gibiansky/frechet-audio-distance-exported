"""CLAP audio encoder preprocessing and utilities for exported FAD.

CLAP (Contrastive Language-Audio Pretraining) 630k-audioset uses an HTSAT-tiny
(Hierarchical Token-Semantic Audio Transformer) audio encoder based on the
Swin Transformer architecture.

The exported model (clap_exported.pt2) contains the full HTSAT-tiny architecture
with the following key components:
- HTSAT-tiny encoder: embed_dim=96, depths=[2,2,6,2], window_size=8
- Projection head: Linear(768, 512) -> ReLU -> Linear(512, 512)
- L2 normalization on output

The model outputs 512-dimensional L2-normalized embeddings.

Input: Log-mel spectrogram [B, 1, 1001, 64] (48kHz, hop=480, 10 seconds)
Output: L2-normalized embeddings [B, 512]

Source package: laion_clap
Reference: https://github.com/LAION-AI/CLAP
"""

import numpy as np
import torch

from .pann import waveform_to_logmel

# ============================================================================
# CLAP Constants
# ============================================================================

CLAP_SAMPLE_RATE = 48000
CLAP_EMBEDDING_SIZE = 512
MAX_AUDIO_SECONDS = 10
MAX_SAMPLES = MAX_AUDIO_SECONDS * CLAP_SAMPLE_RATE  # 480000


# ============================================================================
# Preprocessing Functions
# ============================================================================

def preprocess_for_clap(
    audio: np.ndarray,
    sample_rate: int,
    return_tensor: bool = True,
    apply_quantization: bool = True
) -> np.ndarray:
    """Preprocess audio for CLAP (48kHz mel-spectrogram).

    This function converts audio waveforms to log-mel spectrograms using
    the CLAP-specific parameters (48kHz, hop=480).

    Args:
        audio: Audio waveform as numpy array, shape (samples,) or (samples, channels).
        sample_rate: Sample rate of input audio.
        return_tensor: If True, return as PyTorch tensor ready for model input.
        apply_quantization: If True, apply int16 quantization to match CLAP training.
            CLAP was trained with: audio = (audio * 32767).astype(int16).astype(float32) / 32767

    Returns:
        Log-mel spectrogram:
        - If return_tensor=True: torch.Tensor of shape [1, 1, time_steps, 64]
        - If return_tensor=False: np.array of shape [time_steps, 64]
    """
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # Apply int16 quantization to match CLAP training preprocessing
    # This is done during CLAP's training and helps match embeddings
    if apply_quantization:
        audio = audio.astype(np.float32)
        audio = (audio * 32767.0).astype(np.int16).astype(np.float32) / 32767.0

    # Use PANN's waveform_to_logmel with 48kHz config
    return waveform_to_logmel(
        audio,
        sample_rate,
        target_sample_rate=CLAP_SAMPLE_RATE,
        return_tensor=return_tensor
    )


def pad_audio_to_max_length(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Pad audio to maximum length (10 seconds).

    Args:
        audio: Audio waveform as numpy array
        sample_rate: Sample rate of audio

    Returns:
        Padded audio array

    Raises:
        ValueError: If audio is longer than 10 seconds
    """
    max_samples = MAX_AUDIO_SECONDS * sample_rate
    if len(audio) > max_samples:
        raise ValueError(
            f"Audio too long: {len(audio) / sample_rate:.2f}s > {MAX_AUDIO_SECONDS}s max"
        )

    if len(audio) < max_samples:
        audio = np.pad(audio, (0, max_samples - len(audio)), mode='constant')

    return audio
