"""PANN CNN14 model and preprocessing for exported FAD.

This module contains:
- PANNCore: The pure PyTorch CNN14 neural network (exportable with torch.export)
- Preprocessing functions: Convert audio waveforms to log-mel spectrograms using librosa

The PANN (Pretrained Audio Neural Networks) model is from:
Kong et al., "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition"
IEEE/ACM Transactions on Audio, Speech, and Language Processing 28 (2020)

Original implementation: https://github.com/qiuqiangkong/audioset_tagging_cnn
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import resampy

# ============================================================================
# PANN Parameters for each sample rate variant
# ============================================================================

PANN_CONFIGS = {
    8000: {
        "sample_rate": 8000,
        "window_size": 256,
        "hop_size": 80,
        "mel_bins": 64,
        "fmin": 50,
        "fmax": 4000,
    },
    16000: {
        "sample_rate": 16000,
        "window_size": 512,
        "hop_size": 160,
        "mel_bins": 64,
        "fmin": 50,
        "fmax": 8000,
    },
    32000: {
        "sample_rate": 32000,
        "window_size": 1024,
        "hop_size": 320,
        "mel_bins": 64,
        "fmin": 50,
        "fmax": 14000,
    },
    # 48kHz config used by CLAP (same architecture, different mel params)
    48000: {
        "sample_rate": 48000,
        "window_size": 1024,
        "hop_size": 480,
        "mel_bins": 64,
        "fmin": 50,
        "fmax": 14000,
    },
}

EMBEDDING_SIZE = 2048


# ============================================================================
# Preprocessing Functions (using librosa)
# ============================================================================

def waveform_to_logmel(
    audio: np.ndarray,
    sample_rate: int,
    target_sample_rate: int = 16000,
    return_tensor: bool = True
) -> np.ndarray:
    """Convert audio waveform to log-mel spectrogram for PANN.

    This function replicates the preprocessing done by torchlibrosa in the
    original PANN implementation.

    Args:
        audio: Audio waveform as numpy array, shape (samples,) or (samples, channels).
        sample_rate: Sample rate of input audio.
        target_sample_rate: Target sample rate (must be 8000, 16000, or 32000).
        return_tensor: If True, return as PyTorch tensor ready for model input.

    Returns:
        Log-mel spectrogram:
        - If return_tensor=True: torch.Tensor of shape [1, 1, time_steps, 64]
        - If return_tensor=False: np.array of shape [time_steps, 64]
    """
    if target_sample_rate not in PANN_CONFIGS:
        raise ValueError(f"target_sample_rate must be one of {list(PANN_CONFIGS.keys())}")

    config = PANN_CONFIGS[target_sample_rate]

    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # Resample if needed
    if sample_rate != target_sample_rate:
        audio = resampy.resample(audio, sample_rate, target_sample_rate)

    # Ensure float32
    audio = audio.astype(np.float32)

    # Compute STFT (matching torchlibrosa parameters)
    stft = librosa.stft(
        audio,
        n_fft=config["window_size"],
        hop_length=config["hop_size"],
        win_length=config["window_size"],
        window='hann',
        center=True,
        pad_mode='reflect'
    )

    # Power spectrogram
    power_spec = np.abs(stft) ** 2  # Shape: (n_fft//2+1, time_steps)

    # Mel filterbank
    mel_basis = librosa.filters.mel(
        sr=target_sample_rate,
        n_fft=config["window_size"],
        n_mels=config["mel_bins"],
        fmin=config["fmin"],
        fmax=config["fmax"]
    )  # Shape: (mel_bins, n_fft//2+1)

    # Apply mel filterbank
    mel_spec = np.dot(mel_basis, power_spec)  # Shape: (mel_bins, time_steps)

    # Log mel spectrogram (matching torchlibrosa power_to_db with ref=1.0, amin=1e-10, top_db=None)
    amin = 1e-10
    log_mel = 10.0 * np.log10(np.maximum(mel_spec, amin))
    # Note: ref=1.0 means we subtract 10*log10(1) = 0, so no change
    # top_db=None means no clamping

    # Transpose to (time_steps, mel_bins)
    log_mel = log_mel.T

    if return_tensor:
        # Shape: [1, 1, time_steps, mel_bins]
        log_mel = torch.from_numpy(log_mel).float().unsqueeze(0).unsqueeze(0)

    return log_mel


# ============================================================================
# ConvBlock (matches original PANN implementation)
# ============================================================================

class ConvBlock(nn.Module):
    """Convolutional block used in PANN CNN14.

    Two 3x3 convolutions with batch norm and ReLU, followed by average pooling.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor, pool_size: tuple = (2, 2)) -> torch.Tensor:
        """Forward pass with average pooling.

        Args:
            x: Input tensor of shape [batch, channels, time, freq]
            pool_size: Pooling kernel size

        Returns:
            Output tensor after convolutions and pooling
        """
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.avg_pool2d(x, kernel_size=pool_size)
        return x


# ============================================================================
# PANNCore Model (Exportable)
# ============================================================================

class PANNCore(nn.Module):
    """Pure PyTorch PANN CNN14 model without preprocessing.

    This model takes preprocessed log-mel spectrogram as input
    and outputs 2048-dimensional embeddings.

    Input shape: [batch, 1, time_steps, 64] (log-mel spectrogram)
    Output shape: [batch, 2048] (embeddings)

    The CNN architecture is identical for all sample rate variants (8k, 16k, 32k).
    Only the preprocessing parameters differ.

    Architecture:
    - bn0: BatchNorm2d(64) applied across mel bins
    - 6 ConvBlocks: 1→64→128→256→512→1024→2048
    - Global pooling: mean + max over time dimension
    - fc1: Linear(2048, 2048) with ReLU
    """

    def __init__(self):
        super(PANNCore, self).__init__()

        # Batch norm applied to mel bins (after transpose)
        self.bn0 = nn.BatchNorm2d(64)

        # Convolutional blocks
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        # Fully connected layer for embedding
        self.fc1 = nn.Linear(2048, 2048, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [batch, 1, time_steps, 64]
               (log-mel spectrogram from waveform_to_logmel)

        Returns:
            Embeddings tensor of shape [batch, 2048]
        """
        # Apply batch norm across mel bins
        # Input: [batch, 1, time, 64]
        # Transpose to [batch, 64, time, 1] for bn, then back
        x = x.transpose(1, 3)  # [batch, 64, time, 1]
        x = self.bn0(x)
        x = x.transpose(1, 3)  # [batch, 1, time, 64]

        # Convolutional blocks with pooling
        # Each block halves spatial dimensions (except block6 which uses (1,1))
        x = self.conv_block1(x, pool_size=(2, 2))  # [batch, 64, time/2, 32]
        x = self.conv_block2(x, pool_size=(2, 2))  # [batch, 128, time/4, 16]
        x = self.conv_block3(x, pool_size=(2, 2))  # [batch, 256, time/8, 8]
        x = self.conv_block4(x, pool_size=(2, 2))  # [batch, 512, time/16, 4]
        x = self.conv_block5(x, pool_size=(2, 2))  # [batch, 1024, time/32, 2]
        x = self.conv_block6(x, pool_size=(1, 1))  # [batch, 2048, time/32, 2]

        # Global pooling over frequency dimension
        x = torch.mean(x, dim=3)  # [batch, 2048, time/32]

        # Global pooling over time dimension (max + mean)
        x1, _ = torch.max(x, dim=2)  # [batch, 2048]
        x2 = torch.mean(x, dim=2)    # [batch, 2048]
        x = x1 + x2                  # [batch, 2048]

        # Fully connected layer with ReLU
        x = F.relu_(self.fc1(x))     # [batch, 2048]

        return x
