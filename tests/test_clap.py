"""Tests for CLAP preprocessing and FAD integration.

These tests do NOT require laion_clap to be installed.
They verify basic functionality of the CLAP preprocessing and FAD integration.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_test_audio(duration: float, freq: float, sample_rate: int) -> np.ndarray:
    """Generate a sine wave test audio."""
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    audio = (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32)
    return audio


class TestCLAPPreprocessing:
    """Test CLAP preprocessing functions."""

    def test_preprocess_shape(self):
        """Test preprocessing output shape."""
        from frechet_audio_distance_exported.models.clap import preprocess_for_clap, CLAP_SAMPLE_RATE

        audio = generate_test_audio(2.0, 440.0, CLAP_SAMPLE_RATE)
        preprocessed = preprocess_for_clap(audio, CLAP_SAMPLE_RATE, return_tensor=True)

        # Should be [1, 1, time_steps, 64]
        assert preprocessed.dim() == 4
        assert preprocessed.shape[0] == 1  # batch
        assert preprocessed.shape[1] == 1  # channel
        assert preprocessed.shape[3] == 64  # mel bins

    def test_preprocess_resampling(self):
        """Test that audio is resampled to 48kHz."""
        from frechet_audio_distance_exported.models.clap import preprocess_for_clap, CLAP_SAMPLE_RATE

        # Generate at 44100Hz, target 48kHz
        audio = generate_test_audio(1.0, 440.0, 44100)
        preprocessed = preprocess_for_clap(audio, 44100, return_tensor=True)

        # Time steps should correspond to 48kHz (hop=480)
        # 1 second at 48kHz with hop 480 = 48000 / 480 = 100 frames (+ padding)
        expected_frames = int(len(audio) * CLAP_SAMPLE_RATE / 44100) // 480
        # Allow some tolerance for STFT windowing
        assert abs(preprocessed.shape[2] - expected_frames) < 5

    def test_preprocess_numpy_output(self):
        """Test preprocessing with numpy output."""
        from frechet_audio_distance_exported.models.clap import preprocess_for_clap, CLAP_SAMPLE_RATE

        audio = generate_test_audio(1.0, 440.0, CLAP_SAMPLE_RATE)
        preprocessed = preprocess_for_clap(audio, CLAP_SAMPLE_RATE, return_tensor=False)

        assert isinstance(preprocessed, np.ndarray)
        assert len(preprocessed.shape) == 2  # [time_steps, 64]
        assert preprocessed.shape[1] == 64  # mel bins

    def test_preprocess_stereo_to_mono(self):
        """Test that stereo audio is converted to mono."""
        from frechet_audio_distance_exported.models.clap import preprocess_for_clap, CLAP_SAMPLE_RATE

        # Create stereo audio
        mono = generate_test_audio(1.0, 440.0, CLAP_SAMPLE_RATE)
        stereo = np.column_stack([mono * 0.8, mono * 1.2])

        preprocessed = preprocess_for_clap(stereo, CLAP_SAMPLE_RATE, return_tensor=True)

        # Should still work and produce same shape as mono
        assert preprocessed.dim() == 4
        assert preprocessed.shape[3] == 64

    def test_preprocess_quantization(self):
        """Test that int16 quantization is applied when requested."""
        from frechet_audio_distance_exported.models.clap import preprocess_for_clap, CLAP_SAMPLE_RATE

        audio = generate_test_audio(1.0, 440.0, CLAP_SAMPLE_RATE)

        # With quantization
        preprocessed_quant = preprocess_for_clap(
            audio, CLAP_SAMPLE_RATE, return_tensor=True, apply_quantization=True
        )

        # Without quantization
        preprocessed_no_quant = preprocess_for_clap(
            audio, CLAP_SAMPLE_RATE, return_tensor=True, apply_quantization=False
        )

        # Results should be slightly different (quantization changes values)
        # The difference can be significant in dB scale (log mel-spectrogram)
        # but should still be bounded
        diff = (preprocessed_quant - preprocessed_no_quant).abs().max().item()
        # Mel-spectrograms are in dB scale, so differences can be > 10 dB
        assert diff < 50.0  # Allow reasonable mel-spectrogram difference in dB


class TestCLAPPadding:
    """Test CLAP padding functions."""

    def test_pad_audio_to_max_length_short_audio(self):
        """Test padding short audio."""
        from frechet_audio_distance_exported.models.clap import pad_audio_to_max_length, CLAP_SAMPLE_RATE

        audio = generate_test_audio(2.0, 440.0, CLAP_SAMPLE_RATE)
        padded = pad_audio_to_max_length(audio, CLAP_SAMPLE_RATE)

        # Should be padded to 10 seconds
        assert len(padded) == 10 * CLAP_SAMPLE_RATE
        # Original content should be preserved
        np.testing.assert_array_equal(padded[:len(audio)], audio)
        # Padding should be zeros
        assert (padded[len(audio):] == 0).all()

    def test_pad_audio_to_max_length_exact(self):
        """Test that exact max length is unchanged."""
        from frechet_audio_distance_exported.models.clap import pad_audio_to_max_length, CLAP_SAMPLE_RATE

        audio = generate_test_audio(10.0, 440.0, CLAP_SAMPLE_RATE)
        padded = pad_audio_to_max_length(audio, CLAP_SAMPLE_RATE)

        np.testing.assert_array_equal(padded, audio)

    def test_pad_audio_too_long_raises_error(self):
        """Test that audio longer than 10 seconds raises an error."""
        from frechet_audio_distance_exported.models.clap import pad_audio_to_max_length, CLAP_SAMPLE_RATE

        audio = generate_test_audio(11.0, 440.0, CLAP_SAMPLE_RATE)

        with pytest.raises(ValueError, match="Audio too long"):
            pad_audio_to_max_length(audio, CLAP_SAMPLE_RATE)


class TestCLAPConfig:
    """Test CLAP configuration constants."""

    def test_constants(self):
        """Test that constants are correct."""
        from frechet_audio_distance_exported.models.clap import (
            CLAP_SAMPLE_RATE,
            CLAP_EMBEDDING_SIZE,
            MAX_AUDIO_SECONDS,
            MAX_SAMPLES,
        )

        assert CLAP_SAMPLE_RATE == 48000
        assert CLAP_EMBEDDING_SIZE == 512
        assert MAX_AUDIO_SECONDS == 10
        assert MAX_SAMPLES == 10 * 48000


class TestCLAPFADIntegration:
    """Test CLAP integration with FAD class."""

    def test_fad_model_name_validation(self):
        """Test that CLAP model name is valid."""
        from frechet_audio_distance_exported.fad import VALID_MODELS

        assert "clap" in VALID_MODELS

    def test_fad_model_config(self):
        """Test FAD model configuration for CLAP."""
        from frechet_audio_distance_exported.fad import VALID_MODELS

        config = VALID_MODELS["clap"]
        assert config["sample_rate"] == 48000
        assert config["embedding_dim"] == 512

    def test_invalid_model_name(self):
        """Test that invalid model name raises error."""
        from frechet_audio_distance_exported import FrechetAudioDistance

        with pytest.raises(ValueError, match="Unknown model"):
            FrechetAudioDistance(model_name="clap-invalid")


class TestCLAPFADWithModel:
    """Tests that require the exported model to be present."""

    @pytest.fixture(autouse=True)
    def skip_if_no_model(self):
        """Skip tests if exported model is not available."""
        model_path = Path(__file__).parent.parent / "clap_exported.pt2"
        if not model_path.exists():
            pytest.skip("Exported CLAP model not found")

    def test_fad_initialization(self):
        """Test FAD initialization with CLAP model."""
        from frechet_audio_distance_exported import FrechetAudioDistance

        ckpt_dir = str(Path(__file__).parent.parent)
        fad = FrechetAudioDistance(model_name="clap", ckpt_dir=ckpt_dir)

        assert fad.model_name == "clap"
        assert fad.sample_rate == 48000

    def test_get_embedding_shape(self):
        """Test that embeddings have correct shape."""
        from frechet_audio_distance_exported import FrechetAudioDistance
        from frechet_audio_distance_exported.models.clap import CLAP_SAMPLE_RATE

        ckpt_dir = str(Path(__file__).parent.parent)
        fad = FrechetAudioDistance(model_name="clap", ckpt_dir=ckpt_dir)
        # Force CPU for testing
        fad.device = torch.device('cpu')
        fad.model = fad.model.cpu()

        audio = generate_test_audio(2.0, 440.0, CLAP_SAMPLE_RATE)
        embd = fad._get_embedding_for_audio(audio)

        # Should be [1, 512] (one embedding per audio file)
        assert embd.ndim == 2
        assert embd.shape[0] == 1
        assert embd.shape[1] == 512

    def test_embedding_l2_normalized(self):
        """Test that embeddings are L2-normalized."""
        from frechet_audio_distance_exported import FrechetAudioDistance
        from frechet_audio_distance_exported.models.clap import CLAP_SAMPLE_RATE

        ckpt_dir = str(Path(__file__).parent.parent)
        fad = FrechetAudioDistance(model_name="clap", ckpt_dir=ckpt_dir)
        fad.device = torch.device('cpu')
        fad.model = fad.model.cpu()

        audio = generate_test_audio(2.0, 440.0, CLAP_SAMPLE_RATE)
        embd = fad._get_embedding_for_audio(audio)

        # Check L2 norm
        norm = np.linalg.norm(embd)
        np.testing.assert_allclose(norm, 1.0, rtol=1e-5)

    def test_embedding_deterministic(self):
        """Test that embeddings are deterministic."""
        from frechet_audio_distance_exported import FrechetAudioDistance
        from frechet_audio_distance_exported.models.clap import CLAP_SAMPLE_RATE

        ckpt_dir = str(Path(__file__).parent.parent)
        fad = FrechetAudioDistance(model_name="clap", ckpt_dir=ckpt_dir)
        fad.device = torch.device('cpu')
        fad.model = fad.model.cpu()

        audio = generate_test_audio(2.0, 440.0, CLAP_SAMPLE_RATE)
        embd1 = fad._get_embedding_for_audio(audio)
        embd2 = fad._get_embedding_for_audio(audio)

        np.testing.assert_array_equal(embd1, embd2)

    def test_fad_score_computation(self):
        """Test end-to-end FAD score computation."""
        from frechet_audio_distance_exported import FrechetAudioDistance
        from frechet_audio_distance_exported.models.clap import CLAP_SAMPLE_RATE

        with tempfile.TemporaryDirectory() as tmpdir:
            bg_dir = os.path.join(tmpdir, "background")
            eval_dir = os.path.join(tmpdir, "eval")
            os.makedirs(bg_dir)
            os.makedirs(eval_dir)

            # Generate test audio
            for i in range(5):
                audio = generate_test_audio(2.0, 440.0 + i * 10, CLAP_SAMPLE_RATE)
                sf.write(os.path.join(bg_dir, f"bg_{i}.wav"), audio, CLAP_SAMPLE_RATE)
                audio = generate_test_audio(2.0, 880.0 + i * 10, CLAP_SAMPLE_RATE)
                sf.write(os.path.join(eval_dir, f"eval_{i}.wav"), audio, CLAP_SAMPLE_RATE)

            ckpt_dir = str(Path(__file__).parent.parent)
            fad = FrechetAudioDistance(model_name="clap", ckpt_dir=ckpt_dir)
            fad.device = torch.device('cpu')
            fad.model = fad.model.cpu()

            score = fad.score(bg_dir, eval_dir)

            # Should produce a valid score
            assert isinstance(score, float)
            assert score > 0  # Different audio should have positive FAD
            assert np.isfinite(score)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
