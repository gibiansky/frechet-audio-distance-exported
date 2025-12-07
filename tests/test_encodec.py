"""Tests for Encodec preprocessing and FAD integration.

These tests do NOT require encodec to be installed.
They verify basic functionality of the Encodec preprocessing and FAD integration.
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


class TestEncodecPreprocessing:
    """Test Encodec preprocessing functions."""

    @pytest.mark.parametrize("sample_rate", [24000, 48000])
    def test_preprocess_shape_mono_input(self, sample_rate):
        """Test preprocessing with mono input."""
        from frechet_audio_distance_exported.models.encodec import preprocess_for_encodec, ENCODEC_CONFIGS

        config = ENCODEC_CONFIGS[sample_rate]
        target_channels = config["channels"]

        audio = generate_test_audio(2.0, 440.0, sample_rate)
        preprocessed = preprocess_for_encodec(
            audio, sample_rate,
            target_sample_rate=sample_rate,
            target_channels=target_channels,
            return_tensor=True
        )

        # Should be [1, channels, samples]
        assert preprocessed.dim() == 3
        assert preprocessed.shape[0] == 1  # batch
        assert preprocessed.shape[1] == target_channels
        assert preprocessed.shape[2] == len(audio)

    def test_preprocess_mono_to_stereo(self):
        """Test mono to stereo conversion for 48kHz."""
        from frechet_audio_distance_exported.models.encodec import preprocess_for_encodec

        audio = generate_test_audio(1.0, 440.0, 48000)
        preprocessed = preprocess_for_encodec(
            audio, 48000,
            target_sample_rate=48000,
            target_channels=2,
            return_tensor=True
        )

        # Should duplicate channels
        assert preprocessed.shape[1] == 2
        assert torch.allclose(preprocessed[0, 0], preprocessed[0, 1])

    def test_preprocess_stereo_to_mono(self):
        """Test stereo to mono conversion for 24kHz."""
        from frechet_audio_distance_exported.models.encodec import preprocess_for_encodec

        # Create stereo audio
        mono = generate_test_audio(1.0, 440.0, 24000)
        stereo = np.column_stack([mono * 0.8, mono * 1.2])

        preprocessed = preprocess_for_encodec(
            stereo, 24000,
            target_sample_rate=24000,
            target_channels=1,
            return_tensor=True
        )

        # Should average channels
        assert preprocessed.shape[1] == 1
        # Output should be average of the two channels
        expected = (mono * 0.8 + mono * 1.2) / 2
        np.testing.assert_allclose(preprocessed[0, 0].numpy(), expected, rtol=1e-5)

    def test_preprocess_numpy_output(self):
        """Test preprocessing with numpy output."""
        from frechet_audio_distance_exported.models.encodec import preprocess_for_encodec

        audio = generate_test_audio(1.0, 440.0, 24000)
        preprocessed = preprocess_for_encodec(
            audio, 24000,
            target_sample_rate=24000,
            target_channels=1,
            return_tensor=False
        )

        assert isinstance(preprocessed, np.ndarray)
        assert len(preprocessed.shape) == 2  # [channels, samples]
        assert preprocessed.shape[0] == 1  # mono

    def test_preprocess_resampling(self):
        """Test that audio is resampled to target sample rate."""
        from frechet_audio_distance_exported.models.encodec import preprocess_for_encodec

        # Generate at 44100Hz, target 24000Hz
        audio = generate_test_audio(1.0, 440.0, 44100)
        preprocessed = preprocess_for_encodec(
            audio, 44100,
            target_sample_rate=24000,
            target_channels=1,
            return_tensor=True
        )

        # Samples should be resampled
        expected_samples = int(len(audio) * 24000 / 44100)
        assert abs(preprocessed.shape[2] - expected_samples) < 2  # Allow small rounding difference

    def test_invalid_sample_rate(self):
        """Test that invalid sample rates raise an error."""
        from frechet_audio_distance_exported.models.encodec import preprocess_for_encodec

        audio = generate_test_audio(1.0, 440.0, 24000)

        with pytest.raises(ValueError):
            preprocess_for_encodec(audio, 24000, target_sample_rate=22050)


class TestEncodecPadding:
    """Test Encodec padding functions."""

    @pytest.mark.parametrize("sample_rate,max_samples", [(24000, 240000), (48000, 480000)])
    def test_pad_to_fixed_length(self, sample_rate, max_samples):
        """Test padding to fixed length."""
        from frechet_audio_distance_exported.models.encodec import pad_to_fixed_length

        # Create tensor shorter than max
        original_samples = 48000  # 2 seconds for 24kHz, 1 second for 48kHz
        x = torch.randn(1, 1, original_samples)

        padded = pad_to_fixed_length(x, sample_rate)

        assert padded.shape == (1, 1, max_samples)
        # Original content should be preserved
        assert torch.allclose(padded[:, :, :original_samples], x)
        # Padding should be zeros
        assert (padded[:, :, original_samples:] == 0).all()

    def test_pad_exactly_max_length(self):
        """Test that exact max length is not changed."""
        from frechet_audio_distance_exported.models.encodec import pad_to_fixed_length

        x = torch.randn(1, 1, 240000)  # Exactly 10 seconds at 24kHz
        padded = pad_to_fixed_length(x, 24000)

        assert torch.allclose(padded, x)

    def test_pad_too_long_raises_error(self):
        """Test that audio longer than max raises an error."""
        from frechet_audio_distance_exported.models.encodec import pad_to_fixed_length

        x = torch.randn(1, 1, 300000)  # Longer than 10 seconds at 24kHz

        with pytest.raises(ValueError, match="Audio too long"):
            pad_to_fixed_length(x, 24000)

    def test_pad_to_valid_encodec_length(self):
        """Test padding to be divisible by hop length."""
        from frechet_audio_distance_exported.models.encodec import pad_to_valid_encodec_length

        # Create tensor not divisible by 320
        x = torch.randn(1, 1, 1000)
        padded = pad_to_valid_encodec_length(x)

        # Should be divisible by 320
        assert padded.shape[2] % 320 == 0
        # Should be >= original
        assert padded.shape[2] >= 1000

    @pytest.mark.parametrize("samples", [320, 640, 960, 1600])
    def test_pad_already_divisible(self, samples):
        """Test that already divisible lengths are unchanged."""
        from frechet_audio_distance_exported.models.encodec import pad_to_valid_encodec_length

        x = torch.randn(1, 1, samples)
        padded = pad_to_valid_encodec_length(x)

        assert padded.shape == x.shape


class TestEncodecConfig:
    """Test Encodec configuration constants."""

    def test_configs_exist(self):
        """Test that configs exist for both sample rates."""
        from frechet_audio_distance_exported.models.encodec import ENCODEC_CONFIGS

        assert 24000 in ENCODEC_CONFIGS
        assert 48000 in ENCODEC_CONFIGS

    def test_config_values(self):
        """Test config values are correct."""
        from frechet_audio_distance_exported.models.encodec import ENCODEC_CONFIGS

        config_24k = ENCODEC_CONFIGS[24000]
        assert config_24k["channels"] == 1  # mono
        assert config_24k["embedding_dim"] == 128
        assert config_24k["hop_length"] == 320

        config_48k = ENCODEC_CONFIGS[48000]
        assert config_48k["channels"] == 2  # stereo
        assert config_48k["embedding_dim"] == 128
        assert config_48k["hop_length"] == 320

    def test_max_samples_constants(self):
        """Test max samples constants are correct."""
        from frechet_audio_distance_exported.models.encodec import (
            MAX_AUDIO_SECONDS,
            MAX_SAMPLES_24K,
            MAX_SAMPLES_48K,
        )

        assert MAX_AUDIO_SECONDS == 10
        assert MAX_SAMPLES_24K == 10 * 24000
        assert MAX_SAMPLES_48K == 10 * 48000


class TestEncodecFADIntegration:
    """Test Encodec integration with FAD class."""

    def test_fad_model_name_validation(self):
        """Test that Encodec model names are valid."""
        from frechet_audio_distance_exported.fad import VALID_MODELS

        assert "encodec-24k" in VALID_MODELS
        assert "encodec-48k" in VALID_MODELS

    def test_fad_model_config(self):
        """Test FAD model configuration for Encodec."""
        from frechet_audio_distance_exported.fad import VALID_MODELS

        config_24k = VALID_MODELS["encodec-24k"]
        assert config_24k["sample_rate"] == 24000
        assert config_24k["embedding_dim"] == 128
        assert config_24k["channels"] == 1

        config_48k = VALID_MODELS["encodec-48k"]
        assert config_48k["sample_rate"] == 48000
        assert config_48k["embedding_dim"] == 128
        assert config_48k["channels"] == 2

    def test_fad_sample_rate_mapping(self):
        """Test sample rate mapping for Encodec."""
        from frechet_audio_distance_exported.fad import ENCODEC_SAMPLE_RATES

        assert ENCODEC_SAMPLE_RATES["encodec-24k"] == 24000
        assert ENCODEC_SAMPLE_RATES["encodec-48k"] == 48000

    def test_invalid_model_name(self):
        """Test that invalid model name raises error."""
        from frechet_audio_distance_exported import FrechetAudioDistance

        with pytest.raises(ValueError, match="Unknown model"):
            FrechetAudioDistance(model_name="encodec-invalid")


class TestEncodecFADWithModel:
    """Tests that require the exported model to be present."""

    @pytest.fixture(autouse=True)
    def skip_if_no_model(self):
        """Skip tests if exported model is not available."""
        model_path = Path(__file__).parent.parent / "encodec_24k_exported.pt"
        if not model_path.exists():
            pytest.skip("Exported Encodec model not found")

    def test_fad_initialization(self):
        """Test FAD initialization with Encodec model."""
        from frechet_audio_distance_exported import FrechetAudioDistance

        ckpt_dir = str(Path(__file__).parent.parent)
        fad = FrechetAudioDistance(model_name="encodec-24k", ckpt_dir=ckpt_dir)

        assert fad.model_name == "encodec-24k"
        assert fad.sample_rate == 24000

    def test_get_embedding_shape(self):
        """Test that embeddings have correct shape."""
        from frechet_audio_distance_exported import FrechetAudioDistance

        ckpt_dir = str(Path(__file__).parent.parent)
        fad = FrechetAudioDistance(model_name="encodec-24k", ckpt_dir=ckpt_dir)
        # Force CPU for testing
        fad.device = torch.device('cpu')
        fad.model = fad.model.cpu()

        audio = generate_test_audio(2.0, 440.0, 24000)
        embd = fad._get_embedding_for_audio(audio)

        # Should be [time_frames, 128]
        assert embd.ndim == 2
        assert embd.shape[1] == 128
        # 2 seconds at 24kHz with hop 320 = 2*24000/320 = 150 frames
        expected_frames = len(audio) // 320
        assert embd.shape[0] == expected_frames

    def test_embedding_deterministic(self):
        """Test that embeddings are deterministic."""
        from frechet_audio_distance_exported import FrechetAudioDistance

        ckpt_dir = str(Path(__file__).parent.parent)
        fad = FrechetAudioDistance(model_name="encodec-24k", ckpt_dir=ckpt_dir)
        fad.device = torch.device('cpu')
        fad.model = fad.model.cpu()

        audio = generate_test_audio(2.0, 440.0, 24000)
        embd1 = fad._get_embedding_for_audio(audio)
        embd2 = fad._get_embedding_for_audio(audio)

        np.testing.assert_array_equal(embd1, embd2)

    def test_fad_score_computation(self):
        """Test end-to-end FAD score computation."""
        from frechet_audio_distance_exported import FrechetAudioDistance

        with tempfile.TemporaryDirectory() as tmpdir:
            bg_dir = os.path.join(tmpdir, "background")
            eval_dir = os.path.join(tmpdir, "eval")
            os.makedirs(bg_dir)
            os.makedirs(eval_dir)

            # Generate test audio
            for i in range(5):
                audio = generate_test_audio(2.0, 440.0 + i * 10, 24000)
                sf.write(os.path.join(bg_dir, f"bg_{i}.wav"), audio, 24000)
                audio = generate_test_audio(2.0, 880.0 + i * 10, 24000)
                sf.write(os.path.join(eval_dir, f"eval_{i}.wav"), audio, 24000)

            ckpt_dir = str(Path(__file__).parent.parent)
            fad = FrechetAudioDistance(model_name="encodec-24k", ckpt_dir=ckpt_dir)
            fad.device = torch.device('cpu')
            fad.model = fad.model.cpu()

            score = fad.score(bg_dir, eval_dir)

            # Should produce a valid score
            assert isinstance(score, float)
            assert score > 0  # Different audio should have positive FAD
            assert np.isfinite(score)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
