"""Encodec preprocessing for FAD calculation.

This module provides preprocessing functions for Encodec models.
Unlike VGGish/PANN, Encodec works directly on raw waveforms - no mel-spectrogram needed.

The Encodec encoder outputs embeddings of shape [batch, 128, time_frames] where:
- 128 is the embedding dimension
- time_frames = samples / 320 (hop length)
- Frame rate is 75 fps at 24kHz, 150 fps at 48kHz
"""

import numpy as np
import resampy
import torch

# Maximum audio length for traced model (10 seconds)
# All inputs must be padded to this length before passing to the traced model.
# The output can then be trimmed based on the original input length.
MAX_AUDIO_SECONDS = 10
MAX_SAMPLES_24K = MAX_AUDIO_SECONDS * 24000  # 240000 samples
MAX_SAMPLES_48K = MAX_AUDIO_SECONDS * 48000  # 480000 samples

# Encodec configuration for different sample rates
ENCODEC_CONFIGS = {
    24000: {
        "sample_rate": 24000,
        "channels": 1,  # mono
        "embedding_dim": 128,
        "hop_length": 320,  # Total downsampling factor: 8*5*4*2 = 320
        "max_samples": MAX_SAMPLES_24K,
    },
    48000: {
        "sample_rate": 48000,
        "channels": 2,  # stereo
        "embedding_dim": 128,
        "hop_length": 320,
        "max_samples": MAX_SAMPLES_48K,
    },
}

# Embedding size (same for all Encodec models)
EMBEDDING_SIZE = 128


def preprocess_for_encodec(
    audio: np.ndarray,
    sample_rate: int,
    target_sample_rate: int = 24000,
    target_channels: int = 1,
    return_tensor: bool = True,
) -> torch.Tensor | np.ndarray:
    """Preprocess audio for Encodec encoder.

    This function:
    1. Converts stereo to mono or mono to stereo as needed
    2. Resamples to target sample rate
    3. Returns tensor in shape [1, channels, samples]

    Args:
        audio: Audio waveform as numpy array. Can be:
            - 1D array of shape (samples,) for mono
            - 2D array of shape (samples, channels) for stereo
        sample_rate: Sample rate of input audio
        target_sample_rate: Target sample rate (24000 or 48000)
        target_channels: Target number of channels (1 for 24kHz, 2 for 48kHz)
        return_tensor: If True, return PyTorch tensor; else numpy array

    Returns:
        Preprocessed audio as tensor [1, channels, samples] or numpy array [channels, samples]

    Raises:
        ValueError: If target_sample_rate is not 24000 or 48000
    """
    if target_sample_rate not in ENCODEC_CONFIGS:
        raise ValueError(
            f"Unsupported target sample rate: {target_sample_rate}. "
            f"Must be one of {list(ENCODEC_CONFIGS.keys())}"
        )

    # Handle input shape - ensure 1D or 2D
    if audio.ndim == 1:
        # Mono input
        num_channels = 1
    elif audio.ndim == 2:
        # Stereo input: (samples, channels)
        num_channels = audio.shape[1]
    else:
        raise ValueError(f"Audio must be 1D or 2D, got shape {audio.shape}")

    # Convert to mono or stereo as needed
    if target_channels == 1:
        # Need mono output
        if num_channels > 1:
            # Convert stereo to mono by averaging channels
            audio = np.mean(audio, axis=1)
    elif target_channels == 2:
        # Need stereo output
        if num_channels == 1:
            # Convert mono to stereo by duplicating
            if audio.ndim == 1:
                audio = np.column_stack([audio, audio])
            else:
                audio = np.concatenate([audio, audio], axis=1)

    # Ensure correct shape: (samples,) for mono or (samples, 2) for stereo
    if audio.ndim == 1:
        # Mono case
        pass
    elif audio.shape[1] != target_channels:
        raise ValueError(
            f"Channel conversion failed. Expected {target_channels} channels, got {audio.shape[1]}"
        )

    # Resample if needed
    if sample_rate != target_sample_rate:
        if audio.ndim == 1:
            audio = resampy.resample(audio, sample_rate, target_sample_rate)
        else:
            # Resample each channel separately
            audio = np.column_stack([
                resampy.resample(audio[:, c], sample_rate, target_sample_rate)
                for c in range(audio.shape[1])
            ])

    # Convert to float32
    audio = audio.astype(np.float32)

    # Reshape to [channels, samples]
    if audio.ndim == 1:
        audio = audio.reshape(1, -1)  # [1, samples]
    else:
        audio = audio.T  # [channels, samples]

    if return_tensor:
        # Add batch dimension: [1, channels, samples]
        return torch.from_numpy(audio).unsqueeze(0)
    else:
        return audio


def pad_to_fixed_length(x: torch.Tensor, target_sample_rate: int) -> torch.Tensor:
    """Pad audio to the fixed length required by the traced Encodec model.

    The traced Encodec model expects a fixed input length of 10 seconds
    (240000 samples at 24kHz, 480000 samples at 48kHz).

    Args:
        x: Audio tensor of shape [batch, channels, samples]
        target_sample_rate: 24000 or 48000

    Returns:
        Padded tensor with samples equal to max_samples for the sample rate
    """
    config = ENCODEC_CONFIGS[target_sample_rate]
    max_samples = config["max_samples"]
    samples = x.shape[-1]

    if samples > max_samples:
        raise ValueError(
            f"Audio too long: {samples} samples > {max_samples} max samples "
            f"({MAX_AUDIO_SECONDS} seconds at {target_sample_rate}Hz). "
            f"Please split audio into shorter segments."
        )

    if samples < max_samples:
        pad_amount = max_samples - samples
        x = torch.nn.functional.pad(x, (0, pad_amount))

    return x


def pad_to_valid_encodec_length(x: torch.Tensor) -> torch.Tensor:
    """Pad audio to be divisible by Encodec hop length (320).

    DEPRECATED: Use pad_to_fixed_length() instead for traced models.

    The Encodec encoder downsamples by factors [8, 5, 4, 2] = 320 total.
    Input samples must be divisible by 320 for proper dimension handling.

    Args:
        x: Audio tensor of shape [batch, channels, samples]

    Returns:
        Padded tensor with samples divisible by 320
    """
    hop_length = 320
    samples = x.shape[-1]
    remainder = samples % hop_length

    if remainder != 0:
        pad_amount = hop_length - remainder
        x = torch.nn.functional.pad(x, (0, pad_amount))

    return x
