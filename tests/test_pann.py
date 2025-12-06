"""Tests for PANN model and preprocessing.

These tests do NOT require frechet_audio_distance to be installed.
They verify basic functionality of the PANN implementation.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_test_audio(duration: float, freq: float, sample_rate: int) -> np.ndarray:
    """Generate a sine wave test audio."""
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    audio = (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32)
    return audio


class TestPANNPreprocessing:
    """Test PANN preprocessing functions."""

    @pytest.mark.parametrize("sample_rate", [8000, 16000, 32000])
    def test_waveform_to_logmel_shape(self, sample_rate):
        """Test that preprocessing produces correct output shape."""
        from frechet_audio_distance_exported.models.pann import waveform_to_logmel

        # 2 second audio
        audio = generate_test_audio(2.0, 440.0, sample_rate)
        logmel = waveform_to_logmel(audio, sample_rate, target_sample_rate=sample_rate, return_tensor=True)

        # Should be [1, 1, time_steps, 64]
        assert logmel.dim() == 4
        assert logmel.shape[0] == 1  # batch
        assert logmel.shape[1] == 1  # channel
        assert logmel.shape[3] == 64  # mel bins
        # Time steps vary by sample rate, but should be > 0
        assert logmel.shape[2] > 0

    @pytest.mark.parametrize("sample_rate", [8000, 16000, 32000])
    def test_waveform_to_logmel_numpy(self, sample_rate):
        """Test preprocessing with numpy output."""
        from frechet_audio_distance_exported.models.pann import waveform_to_logmel

        audio = generate_test_audio(2.0, 440.0, sample_rate)
        logmel = waveform_to_logmel(audio, sample_rate, target_sample_rate=sample_rate, return_tensor=False)

        assert isinstance(logmel, np.ndarray)
        assert len(logmel.shape) == 2  # [time_steps, 64]
        assert logmel.shape[1] == 64  # mel bins

    def test_waveform_to_logmel_resampling(self):
        """Test that audio is resampled to target sample rate."""
        from frechet_audio_distance_exported.models.pann import waveform_to_logmel

        # Generate at 44100Hz, target 16000Hz
        audio = generate_test_audio(2.0, 440.0, 44100)
        logmel = waveform_to_logmel(audio, 44100, target_sample_rate=16000, return_tensor=True)

        # Should produce valid output
        assert logmel.shape[0] == 1
        assert logmel.shape[1] == 1
        assert logmel.shape[3] == 64

    def test_waveform_to_logmel_stereo(self):
        """Test that stereo audio is converted to mono."""
        from frechet_audio_distance_exported.models.pann import waveform_to_logmel

        # Create stereo audio
        mono = generate_test_audio(2.0, 440.0, 16000)
        stereo = np.column_stack([mono, mono])

        logmel = waveform_to_logmel(stereo, 16000, target_sample_rate=16000, return_tensor=True)

        assert logmel.shape[0] == 1
        assert logmel.shape[1] == 1
        assert logmel.shape[3] == 64

    def test_invalid_sample_rate(self):
        """Test that invalid sample rates raise an error."""
        from frechet_audio_distance_exported.models.pann import waveform_to_logmel

        audio = generate_test_audio(2.0, 440.0, 16000)

        with pytest.raises(ValueError):
            waveform_to_logmel(audio, 16000, target_sample_rate=22050)


class TestPANNCore:
    """Test PANNCore model."""

    def test_model_creation(self):
        """Test that model can be created."""
        from frechet_audio_distance_exported.models.pann import PANNCore

        model = PANNCore()
        assert model is not None

    def test_model_forward_shape(self):
        """Test that model forward produces correct output shape."""
        from frechet_audio_distance_exported.models.pann import PANNCore

        model = PANNCore()
        model.eval()

        # Create random input matching expected shape
        x = torch.randn(5, 1, 200, 64)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (5, 2048)  # 5 samples, 2048 embedding dim

    @pytest.mark.parametrize("batch_size", [1, 2, 10, 32])
    def test_model_different_batch_sizes(self, batch_size):
        """Test model with different batch sizes."""
        from frechet_audio_distance_exported.models.pann import PANNCore

        model = PANNCore()
        model.eval()

        x = torch.randn(batch_size, 1, 200, 64)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (batch_size, 2048)

    @pytest.mark.parametrize("time_steps", [100, 200, 500, 1000])
    def test_model_different_time_lengths(self, time_steps):
        """Test model with different time lengths."""
        from frechet_audio_distance_exported.models.pann import PANNCore

        model = PANNCore()
        model.eval()

        x = torch.randn(2, 1, time_steps, 64)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (2, 2048)

    def test_model_deterministic(self):
        """Test that model is deterministic in eval mode."""
        from frechet_audio_distance_exported.models.pann import PANNCore

        model = PANNCore()
        model.eval()

        x = torch.randn(2, 1, 200, 64)
        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)

        np.testing.assert_array_equal(output1.numpy(), output2.numpy())


class TestPANNEndToEnd:
    """End-to-end tests for PANN preprocessing + model."""

    @pytest.mark.parametrize("sample_rate", [8000, 16000, 32000])
    def test_preprocessing_to_model(self, sample_rate):
        """Test that preprocessing output works with model."""
        from frechet_audio_distance_exported.models.pann import PANNCore, waveform_to_logmel

        model = PANNCore()
        model.eval()

        # Generate audio and preprocess
        audio = generate_test_audio(2.0, 440.0, sample_rate)
        logmel = waveform_to_logmel(audio, sample_rate, target_sample_rate=sample_rate, return_tensor=True)

        # Run through model
        with torch.no_grad():
            output = model(logmel)

        # Should produce embedding
        assert output.shape == (1, 2048)

    def test_different_audio_lengths(self):
        """Test with various audio lengths."""
        from frechet_audio_distance_exported.models.pann import PANNCore, waveform_to_logmel

        model = PANNCore()
        model.eval()

        for duration in [0.5, 1.0, 2.0, 5.0]:
            audio = generate_test_audio(duration, 440.0, 16000)
            logmel = waveform_to_logmel(audio, 16000, target_sample_rate=16000, return_tensor=True)

            with torch.no_grad():
                output = model(logmel)

            assert output.shape == (1, 2048), f"Failed for duration {duration}s"


class TestConvBlock:
    """Test ConvBlock component."""

    def test_conv_block_creation(self):
        """Test ConvBlock creation."""
        from frechet_audio_distance_exported.models.pann import ConvBlock

        block = ConvBlock(in_channels=64, out_channels=128)
        assert block is not None

    def test_conv_block_forward(self):
        """Test ConvBlock forward pass."""
        from frechet_audio_distance_exported.models.pann import ConvBlock

        block = ConvBlock(in_channels=64, out_channels=128)

        x = torch.randn(2, 64, 100, 32)
        output = block(x, pool_size=(2, 2))

        # Spatial dimensions should be halved
        assert output.shape == (2, 128, 50, 16)

    def test_conv_block_no_pooling(self):
        """Test ConvBlock with pool_size=(1,1)."""
        from frechet_audio_distance_exported.models.pann import ConvBlock

        block = ConvBlock(in_channels=64, out_channels=128)

        x = torch.randn(2, 64, 100, 32)
        output = block(x, pool_size=(1, 1))

        # Spatial dimensions should be unchanged
        assert output.shape == (2, 128, 100, 32)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
