"""VGGish model and preprocessing for exported FAD.

This module contains:
- VGGishCore: The pure PyTorch neural network (exportable with torch.export)
- Preprocessing functions: Convert audio waveforms to mel-spectrogram patches
"""

import numpy as np
import torch
import torch.nn as nn
import resampy

# ============================================================================
# VGGish Parameters
# ============================================================================

NUM_FRAMES = 96  # Frames in input mel-spectrogram patch
NUM_BANDS = 64  # Frequency bands in input mel-spectrogram patch
EMBEDDING_SIZE = 128  # Size of embedding layer

SAMPLE_RATE = 16000
STFT_WINDOW_LENGTH_SECONDS = 0.025
STFT_HOP_LENGTH_SECONDS = 0.010
NUM_MEL_BINS = NUM_BANDS
MEL_MIN_HZ = 125
MEL_MAX_HZ = 7500
LOG_OFFSET = 0.01  # Offset used for stabilized log of input mel-spectrogram
EXAMPLE_WINDOW_SECONDS = 0.96  # Each example contains 96 10ms frames
EXAMPLE_HOP_SECONDS = 0.96  # with zero overlap

# Mel spectrum constants
_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 1127.0


# ============================================================================
# VGGishCore Model (Exportable)
# ============================================================================

def _make_layers():
    """Create VGG-style feature extraction layers."""
    layers = []
    in_channels = 1
    for v in [64, "M", 128, "M", 256, 256, "M", 512, 512, "M"]:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGGishCore(nn.Module):
    """Pure PyTorch VGGish model without preprocessing.

    This model takes preprocessed mel-spectrogram patches as input
    and outputs 128-dimensional embeddings.

    Input shape: [batch, 1, 96, 64] (log-mel spectrogram patches)
    Output shape: [batch, 128] (embeddings)

    Note: This version does NOT include the final ReLU activation,
    matching the default use_activation=False in frechet_audio_distance.
    """

    def __init__(self):
        super(VGGishCore, self).__init__()
        self.features = _make_layers()
        # Embedding layers without final ReLU (matches use_activation=False)
        self.embeddings = nn.Sequential(
            nn.Linear(512 * 4 * 6, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 128),
            # No final ReLU - matches frechet_audio_distance default
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [batch, 1, 96, 64]

        Returns:
            Embeddings tensor of shape [batch, 128]
        """
        x = self.features(x)
        # Transpose to remain compatible with vggish embeddings
        x = torch.transpose(x, 1, 3)
        x = torch.transpose(x, 1, 2)
        x = x.contiguous()
        x = x.view(x.size(0), -1)
        return self.embeddings(x)


# ============================================================================
# Preprocessing Functions
# ============================================================================

def _frame(data, window_length, hop_length):
    """Convert array into a sequence of successive possibly overlapping frames.

    Args:
        data: np.array of dimension N >= 1.
        window_length: Number of samples in each frame.
        hop_length: Advance (in samples) between each window.

    Returns:
        (N+1)-D np.array with as many rows as there are complete frames.
    """
    num_samples = data.shape[0]
    num_frames = 1 + int(np.floor((num_samples - window_length) / hop_length))
    shape = (num_frames, window_length) + data.shape[1:]
    strides = (data.strides[0] * hop_length,) + data.strides
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


def _periodic_hann(window_length):
    """Calculate a "periodic" Hann window."""
    return 0.5 - (0.5 * np.cos(2 * np.pi / window_length * np.arange(window_length)))


def _stft_magnitude(signal, fft_length, hop_length=None, window_length=None):
    """Calculate the short-time Fourier transform magnitude.

    Args:
        signal: 1D np.array of the input time-domain signal.
        fft_length: Size of the FFT to apply.
        hop_length: Advance (in samples) between each frame passed to FFT.
        window_length: Length of each block of samples to pass to FFT.

    Returns:
        2D np.array where each row contains the magnitudes of the fft_length/2+1
        unique values of the FFT for the corresponding frame.
    """
    frames = _frame(signal, window_length, hop_length)
    window = _periodic_hann(window_length)
    windowed_frames = frames * window
    return np.abs(np.fft.rfft(windowed_frames, int(fft_length)))


def _hertz_to_mel(frequencies_hertz):
    """Convert frequencies to mel scale using HTK formula."""
    return _MEL_HIGH_FREQUENCY_Q * np.log(
        1.0 + (frequencies_hertz / _MEL_BREAK_FREQUENCY_HERTZ))


def _spectrogram_to_mel_matrix(num_mel_bins=20,
                                num_spectrogram_bins=129,
                                audio_sample_rate=8000,
                                lower_edge_hertz=125.0,
                                upper_edge_hertz=3800.0):
    """Return a matrix that can post-multiply spectrogram rows to make mel.

    Returns a np.array matrix A that can be used to post-multiply a matrix S of
    spectrogram values (STFT magnitudes) arranged as frames x bins to generate a
    "mel spectrogram" M of frames x num_mel_bins. M = S A.
    """
    nyquist_hertz = audio_sample_rate / 2.0
    if lower_edge_hertz < 0.0:
        raise ValueError("lower_edge_hertz %.1f must be >= 0" % lower_edge_hertz)
    if lower_edge_hertz >= upper_edge_hertz:
        raise ValueError("lower_edge_hertz %.1f >= upper_edge_hertz %.1f" %
                        (lower_edge_hertz, upper_edge_hertz))
    if upper_edge_hertz > nyquist_hertz:
        raise ValueError("upper_edge_hertz %.1f is greater than Nyquist %.1f" %
                        (upper_edge_hertz, nyquist_hertz))

    spectrogram_bins_hertz = np.linspace(0.0, nyquist_hertz, num_spectrogram_bins)
    spectrogram_bins_mel = _hertz_to_mel(spectrogram_bins_hertz)

    band_edges_mel = np.linspace(
        _hertz_to_mel(lower_edge_hertz),
        _hertz_to_mel(upper_edge_hertz),
        num_mel_bins + 2)

    mel_weights_matrix = np.empty((num_spectrogram_bins, num_mel_bins))
    for i in range(num_mel_bins):
        lower_edge_mel, center_mel, upper_edge_mel = band_edges_mel[i:i + 3]
        lower_slope = ((spectrogram_bins_mel - lower_edge_mel) /
                      (center_mel - lower_edge_mel))
        upper_slope = ((upper_edge_mel - spectrogram_bins_mel) /
                      (upper_edge_mel - center_mel))
        mel_weights_matrix[:, i] = np.maximum(0.0, np.minimum(lower_slope, upper_slope))

    # HTK excludes the spectrogram DC bin
    mel_weights_matrix[0, :] = 0.0
    return mel_weights_matrix


def _log_mel_spectrogram(data,
                          audio_sample_rate=8000,
                          log_offset=0.0,
                          window_length_secs=0.025,
                          hop_length_secs=0.010,
                          **kwargs):
    """Convert waveform to a log magnitude mel-frequency spectrogram.

    Args:
        data: 1D np.array of waveform data.
        audio_sample_rate: The sampling rate of data.
        log_offset: Add this to values when taking log to avoid -Infs.
        window_length_secs: Duration of each window to analyze.
        hop_length_secs: Advance between successive analysis windows.
        **kwargs: Additional arguments to pass to spectrogram_to_mel_matrix.

    Returns:
        2D np.array of (num_frames, num_mel_bins) consisting of log mel filterbank
        magnitudes for successive frames.
    """
    window_length_samples = int(round(audio_sample_rate * window_length_secs))
    hop_length_samples = int(round(audio_sample_rate * hop_length_secs))
    fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))

    spectrogram = _stft_magnitude(
        data,
        fft_length=fft_length,
        hop_length=hop_length_samples,
        window_length=window_length_samples)

    mel_spectrogram = np.dot(spectrogram, _spectrogram_to_mel_matrix(
        num_spectrogram_bins=spectrogram.shape[1],
        audio_sample_rate=audio_sample_rate, **kwargs))

    return np.log(mel_spectrogram + log_offset)


def waveform_to_examples(data, sample_rate, return_tensor=True):
    """Converts audio waveform into an array of examples for VGGish.

    Args:
        data: np.array of either one dimension (mono) or two dimensions
            (multi-channel, with the outer dimension representing channels).
            Each sample is generally expected to lie in the range [-1.0, +1.0].
        sample_rate: Sample rate of data.
        return_tensor: Return data as a Pytorch tensor ready for VGGish.

    Returns:
        If return_tensor=True: torch.Tensor of shape [num_examples, 1, 96, 64]
        If return_tensor=False: np.array of shape [num_examples, 96, 64]
    """
    # Convert to mono
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    # Resample to the rate assumed by VGGish
    if sample_rate != SAMPLE_RATE:
        data = resampy.resample(data, sample_rate, SAMPLE_RATE)

    # Compute log mel spectrogram features
    log_mel = _log_mel_spectrogram(
        data,
        audio_sample_rate=SAMPLE_RATE,
        log_offset=LOG_OFFSET,
        window_length_secs=STFT_WINDOW_LENGTH_SECONDS,
        hop_length_secs=STFT_HOP_LENGTH_SECONDS,
        num_mel_bins=NUM_MEL_BINS,
        lower_edge_hertz=MEL_MIN_HZ,
        upper_edge_hertz=MEL_MAX_HZ)

    # Frame features into examples
    features_sample_rate = 1.0 / STFT_HOP_LENGTH_SECONDS
    example_window_length = int(round(EXAMPLE_WINDOW_SECONDS * features_sample_rate))
    example_hop_length = int(round(EXAMPLE_HOP_SECONDS * features_sample_rate))

    log_mel_examples = _frame(
        log_mel,
        window_length=example_window_length,
        hop_length=example_hop_length)

    if return_tensor:
        # Shape: [num_examples, 1, 96, 64]
        # Use np.array() to ensure contiguous memory before converting to tensor
        log_mel_examples = np.ascontiguousarray(log_mel_examples)
        log_mel_examples = torch.from_numpy(log_mel_examples)[:, None, :, :].float()

    return log_mel_examples
