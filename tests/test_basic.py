"""Basic tests for frechet-audio-distance-exported.

These tests do NOT require frechet_audio_distance to be installed.
They verify basic functionality of the package.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_test_audio(duration: float, freq: float, sample_rate: int = 16000) -> np.ndarray:
    """Generate a sine wave test audio."""
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    audio = (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32)
    return audio


class TestPreprocessing:
    """Test preprocessing functions."""

    def test_waveform_to_examples_shape(self):
        """Test that preprocessing produces correct output shape."""
        from frechet_audio_distance_exported.models.vggish import waveform_to_examples

        # 2 second audio should produce 2 patches (96 frames each, 0.96s per patch)
        audio = generate_test_audio(2.0, 440.0)
        patches = waveform_to_examples(audio, 16000, return_tensor=True)

        assert patches.shape[0] == 2  # 2 patches
        assert patches.shape[1] == 1  # 1 channel
        assert patches.shape[2] == 96  # 96 frames
        assert patches.shape[3] == 64  # 64 mel bands

    def test_waveform_to_examples_numpy(self):
        """Test preprocessing with numpy output."""
        from frechet_audio_distance_exported.models.vggish import waveform_to_examples

        audio = generate_test_audio(2.0, 440.0)
        patches = waveform_to_examples(audio, 16000, return_tensor=False)

        assert isinstance(patches, np.ndarray)
        assert patches.shape[0] == 2  # 2 patches
        assert patches.shape[1] == 96  # 96 frames
        assert patches.shape[2] == 64  # 64 mel bands

    def test_waveform_to_examples_short_audio(self):
        """Test preprocessing with short audio (< 0.96s)."""
        from frechet_audio_distance_exported.models.vggish import waveform_to_examples

        # 0.5 second audio - too short for a full patch
        audio = generate_test_audio(0.5, 440.0)
        patches = waveform_to_examples(audio, 16000, return_tensor=True)

        # Should still produce at least partial patches or empty
        assert patches.shape[1] == 1  # 1 channel
        assert patches.shape[2] == 96  # 96 frames
        assert patches.shape[3] == 64  # 64 mel bands

    def test_waveform_to_examples_resampling(self):
        """Test that different sample rates are handled."""
        from frechet_audio_distance_exported.models.vggish import waveform_to_examples

        # Generate at 44.1kHz
        t = np.linspace(0, 2.0, int(44100 * 2.0), dtype=np.float32)
        audio = (np.sin(2 * np.pi * 440.0 * t) * 0.5).astype(np.float32)

        patches = waveform_to_examples(audio, 44100, return_tensor=True)

        # Should still produce correct output shape
        assert patches.shape[1] == 1  # 1 channel
        assert patches.shape[2] == 96  # 96 frames
        assert patches.shape[3] == 64  # 64 mel bands


class TestVGGishCore:
    """Test VGGishCore model."""

    def test_model_creation(self):
        """Test that model can be created."""
        from frechet_audio_distance_exported.models.vggish import VGGishCore

        model = VGGishCore()
        assert model is not None

    def test_model_forward_shape(self):
        """Test that model forward produces correct output shape."""
        import torch
        from frechet_audio_distance_exported.models.vggish import VGGishCore

        model = VGGishCore()
        model.eval()

        # Create random input
        x = torch.randn(5, 1, 96, 64)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (5, 128)  # 5 samples, 128 embedding dim

    def test_model_different_batch_sizes(self):
        """Test model with different batch sizes."""
        import torch
        from frechet_audio_distance_exported.models.vggish import VGGishCore

        model = VGGishCore()
        model.eval()

        for batch_size in [1, 2, 10, 32]:
            x = torch.randn(batch_size, 1, 96, 64)
            with torch.no_grad():
                output = model(x)
            assert output.shape == (batch_size, 128)


class TestFADClass:
    """Test FrechetAudioDistance class (without exported model)."""

    def test_frechet_distance_calculation(self):
        """Test Frechet distance calculation."""
        from frechet_audio_distance_exported.fad import FrechetAudioDistance

        # Create a mock FAD instance (won't load model, just test the math)
        # We'll test the calculate_frechet_distance method directly

        # Create instance without loading model
        class MockFAD(FrechetAudioDistance):
            def _load_model(self):
                pass  # Skip model loading

        fad = MockFAD.__new__(MockFAD)
        fad.model_name = "vggish"

        # Same distributions should have distance 0
        mu1 = np.array([1.0, 2.0, 3.0])
        sigma1 = np.eye(3)
        mu2 = np.array([1.0, 2.0, 3.0])
        sigma2 = np.eye(3)

        distance = fad.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        assert abs(distance) < 1e-6

    def test_frechet_distance_different_means(self):
        """Test Frechet distance with different means."""
        from frechet_audio_distance_exported.fad import FrechetAudioDistance

        class MockFAD(FrechetAudioDistance):
            def _load_model(self):
                pass

        fad = MockFAD.__new__(MockFAD)
        fad.model_name = "vggish"

        # Different means should give positive distance
        mu1 = np.array([0.0, 0.0, 0.0])
        sigma1 = np.eye(3)
        mu2 = np.array([1.0, 1.0, 1.0])
        sigma2 = np.eye(3)

        distance = fad.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        assert distance > 0

    def test_embedding_statistics(self):
        """Test embedding statistics calculation."""
        from frechet_audio_distance_exported.fad import FrechetAudioDistance

        class MockFAD(FrechetAudioDistance):
            def _load_model(self):
                pass

        fad = MockFAD.__new__(MockFAD)
        fad.model_name = "vggish"

        # Create random embeddings
        np.random.seed(42)
        embeddings = np.random.randn(100, 128)

        mu, sigma = fad.calculate_embd_statistics(embeddings)

        assert mu.shape == (128,)
        assert sigma.shape == (128, 128)


class TestLoadAudio:
    """Test audio loading functionality."""

    def test_load_audio_basic(self):
        """Test basic audio loading."""
        from frechet_audio_distance_exported.fad import load_audio

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            audio = generate_test_audio(1.0, 440.0)
            sf.write(f.name, audio, 16000)

            loaded = load_audio(f.name, 16000, 1)

            assert len(loaded) == len(audio)
            # WAV encoding/decoding introduces small precision differences
            np.testing.assert_array_almost_equal(loaded, audio, decimal=4)

            os.unlink(f.name)

    def test_load_audio_resampling(self):
        """Test audio loading with resampling."""
        from frechet_audio_distance_exported.fad import load_audio

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            # Create audio at 44.1kHz
            t = np.linspace(0, 1.0, 44100, dtype=np.float32)
            audio = (np.sin(2 * np.pi * 440.0 * t) * 0.5).astype(np.float32)
            sf.write(f.name, audio, 44100)

            # Load at 16kHz
            loaded = load_audio(f.name, 16000, 1)

            # Should be resampled to 16kHz
            assert len(loaded) == 16000

            os.unlink(f.name)

    def test_load_audio_stereo_to_mono(self):
        """Test stereo to mono conversion."""
        from frechet_audio_distance_exported.fad import load_audio

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            # Create stereo audio
            mono = generate_test_audio(1.0, 440.0)
            stereo = np.column_stack([mono, mono])
            sf.write(f.name, stereo, 16000)

            loaded = load_audio(f.name, 16000, 1)

            # Should be mono
            assert len(loaded.shape) == 1
            # WAV encoding/decoding introduces small precision differences
            np.testing.assert_array_almost_equal(loaded, mono, decimal=4)

            os.unlink(f.name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
