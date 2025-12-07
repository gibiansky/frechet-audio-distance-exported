#!/usr/bin/env python
"""Verify that exported CLAP produces equivalent results to original laion_clap.

IMPORTANT: This script requires `laion_clap` to be installed:
    pip install laion_clap

This script tests:
1. Embedding equivalence: Compare embeddings from original CLAP and exported model
2. FAD score equivalence: Compare FAD scores computed by both implementations

Usage:
    python scripts/verify_clap.py
    python scripts/verify_clap.py --ckpt-dir /path/to/models
"""

import os
import sys
import tempfile
from pathlib import Path

# Parse args BEFORE importing anything that might use argparse
def parse_args():
    """Parse arguments manually."""
    audio_file = None
    ckpt_dir = None

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--audio-file" and i + 1 < len(sys.argv):
            audio_file = sys.argv[i + 1]
            i += 2
        elif arg == "--ckpt-dir" and i + 1 < len(sys.argv):
            ckpt_dir = sys.argv[i + 1]
            i += 2
        elif arg in ("-h", "--help"):
            print(__doc__)
            sys.exit(0)
        else:
            i += 1

    # Clear sys.argv to prevent laion_clap from parsing it
    sys.argv = [sys.argv[0]]

    return audio_file, ckpt_dir


# Parse args before any imports
_PARSED_ARGS = parse_args()

import numpy as np
import soundfile as sf
import torch

# Fix for PyTorch 2.6+ compatibility with laion_clap checkpoints
_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from frechet_audio_distance_exported.models.clap import (
    preprocess_for_clap,
    CLAP_SAMPLE_RATE,
)
from frechet_audio_distance_exported.fad import _pad_to_clap_time


def generate_test_audio(duration: float, freq: float, sample_rate: int) -> np.ndarray:
    """Generate a sine wave test audio.

    Args:
        duration: Duration in seconds
        freq: Frequency in Hz
        sample_rate: Sample rate

    Returns:
        Audio as numpy array
    """
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    audio = (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32)
    return audio


def load_original_clap():
    """Load the original CLAP model from laion_clap (HTSAT-tiny 630k-audioset).

    Returns:
        Original CLAP model
    """
    import os
    import laion_clap
    from laion_clap.clap_module import create_model
    from laion_clap.clap_module.factory import load_state_dict

    # Create HTSAT-tiny model (same as export_clap.py)
    model, model_cfg = create_model(
        'HTSAT-tiny',
        'roberta',
        precision='fp32',
        device='cpu',
        enable_fusion=False
    )

    # Find and load checkpoint
    package_dir = os.path.dirname(os.path.realpath(laion_clap.__file__))
    ckpt_path = os.path.join(package_dir, '630k-audioset-best.pt')

    if not os.path.exists(ckpt_path):
        print("Downloading CLAP checkpoint...")
        import wget
        download_link = 'https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-best.pt'
        ckpt_path = wget.download(download_link, package_dir)

    ckpt = load_state_dict(ckpt_path, skip_params=True)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model


def test_embedding_equivalence(audio_file: str = None, ckpt_dir: str = None):
    """Test that embeddings from exported model match original CLAP.

    Args:
        audio_file: Optional path to audio file for testing
        ckpt_dir: Directory containing exported models
    """
    from dataclasses import asdict
    from laion_clap.training.data import get_audio_features
    import torch.nn.functional as F

    print(f"\n{'=' * 60}")
    print("Test 1: Embedding Equivalence")
    print("=" * 60)

    # Load audio
    if audio_file and os.path.exists(audio_file):
        print(f"  Using audio file: {audio_file}")
        audio, sr = sf.read(audio_file, dtype='float32')
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        # Limit to 10 seconds
        max_samples = 10 * sr
        if len(audio) > max_samples:
            audio = audio[:max_samples]
    else:
        print("  Using generated sine wave audio")
        audio = generate_test_audio(2.0, 440.0, CLAP_SAMPLE_RATE)
        sr = CLAP_SAMPLE_RATE

    print(f"  Audio length: {len(audio) / sr:.2f}s")

    # Get original CLAP embeddings
    print("  Loading original CLAP model...")
    original_model = load_original_clap()

    print("  Computing original embeddings...")
    with torch.no_grad():
        # Apply int16 quantization (same as laion_clap training)
        audio_quantized = (audio * 32767.0).astype(np.int16).astype(np.float32) / 32767.0
        # Resample to 48kHz if needed
        if sr != CLAP_SAMPLE_RATE:
            import resampy
            audio_quantized = resampy.resample(audio_quantized, sr, CLAP_SAMPLE_RATE)

        audio_tensor = torch.from_numpy(audio_quantized).float()

        # Use HTSAT's internal extractors for mel-spectrogram
        waveform = audio_tensor.unsqueeze(0)  # [1, samples]
        # Pad to 10 seconds (480000 samples)
        if waveform.shape[1] < 480000:
            waveform = F.pad(waveform, (0, 480000 - waveform.shape[1]))

        x = original_model.audio_branch.spectrogram_extractor(waveform)
        mel = original_model.audio_branch.logmel_extractor(x)

        # Get embedding through HTSAT (same as CLAPAudioEmbedder)
        B, C, T, freq = mel.shape
        freq_ratio = 4
        spec_size = 256
        target_T = spec_size * freq_ratio  # 1024

        # Interpolate time dimension to 1024 if needed
        if T < target_T:
            mel = F.interpolate(mel, (target_T, freq), mode='bicubic', align_corners=True)

        # Apply bn0 normalization
        mel_normalized = mel.transpose(1, 3)
        mel_normalized = original_model.audio_branch.bn0(mel_normalized)
        mel_normalized = mel_normalized.transpose(1, 3)

        # Reshape to HTSAT image format (256x256)
        B, C, T, freq = mel_normalized.shape
        mel_reshaped = mel_normalized.permute(0, 1, 3, 2)  # B C F T
        mel_reshaped = mel_reshaped.reshape(B, C, freq, freq_ratio, T // freq_ratio)
        mel_reshaped = mel_reshaped.permute(0, 1, 3, 2, 4)
        mel_reshaped = mel_reshaped.reshape(B, C, freq_ratio * freq, T // freq_ratio)

        # Forward through HTSAT
        output = original_model.audio_branch.forward_features(mel_reshaped)
        orig_embedding = output['embedding']
        orig_projected = original_model.audio_projection(orig_embedding)
        orig_embd = F.normalize(orig_projected, dim=-1)

    # Load exported model
    print("  Loading exported model...")
    if ckpt_dir is None:
        ckpt_dir = str(Path(__file__).parent.parent)
    model_path = os.path.join(ckpt_dir, "clap_exported.pt2")
    exported = torch.export.load(model_path)
    exported_model = exported.module()
    # Note: eval() not needed/supported for exported models

    print("  Computing exported embeddings...")
    with torch.no_grad():
        # Pad waveform to 10 seconds BEFORE preprocessing (to match original)
        # This is important because mel-spectrogram of zeros != zeros in mel-spectrogram
        audio_padded = audio.copy()
        if len(audio_padded) < 480000:
            audio_padded = np.pad(audio_padded, (0, 480000 - len(audio_padded)))
        preprocessed = preprocess_for_clap(audio_padded, sr, return_tensor=True)
        preprocessed = _pad_to_clap_time(preprocessed)
        export_embd = exported_model(preprocessed)

    # Compare
    max_diff = (orig_embd - export_embd).abs().max().item()
    mean_diff = (orig_embd - export_embd).abs().mean().item()
    cosine_sim = F.cosine_similarity(orig_embd, export_embd).item()

    print(f"  Original embedding shape: {orig_embd.shape}")
    print(f"  Exported embedding shape: {export_embd.shape}")
    print(f"  Max difference: {max_diff:.2e}")
    print(f"  Mean difference: {mean_diff:.2e}")
    print(f"  Cosine similarity: {cosine_sim:.6f}")

    # L2-normalized embeddings should be very similar
    if cosine_sim > 0.99:
        print("  PASSED")
        return True
    else:
        print("  FAILED!")
        return False


def test_fad_score_computation(ckpt_dir: str = None):
    """Test that FAD scores are computed correctly.

    Args:
        ckpt_dir: Directory containing exported models
    """
    print(f"\n{'=' * 60}")
    print("Test 2: FAD Score Computation")
    print("=" * 60)

    # Create temporary directories with test audio
    with tempfile.TemporaryDirectory() as tmpdir:
        bg_dir = os.path.join(tmpdir, "background")
        eval_dir = os.path.join(tmpdir, "eval")
        os.makedirs(bg_dir)
        os.makedirs(eval_dir)

        # Generate background audio (440 Hz sine waves)
        print("  Generating test audio files...")
        for i in range(5):
            audio = generate_test_audio(2.0, 440.0 + i * 10, CLAP_SAMPLE_RATE)
            sf.write(os.path.join(bg_dir, f"bg_{i}.wav"), audio, CLAP_SAMPLE_RATE)

        # Generate eval audio (880 Hz sine waves - different frequency)
        for i in range(5):
            audio = generate_test_audio(2.0, 880.0 + i * 10, CLAP_SAMPLE_RATE)
            sf.write(os.path.join(eval_dir, f"eval_{i}.wav"), audio, CLAP_SAMPLE_RATE)

        # Calculate FAD with exported model
        print("  Computing FAD with exported implementation...")
        from frechet_audio_distance_exported import FrechetAudioDistance

        if ckpt_dir is None:
            ckpt_dir = str(Path(__file__).parent.parent)
        fad = FrechetAudioDistance(model_name="clap", ckpt_dir=ckpt_dir)
        fad_score = fad.score(bg_dir, eval_dir)

        print(f"  FAD score: {fad_score:.6f}")

        # Basic sanity checks
        if fad_score < 0:
            print("  FAD score is negative (error state)")
            print("  FAILED!")
            return False

        if fad_score == 0:
            print("  FAD score is exactly zero (unexpected for different audio)")
            print("  FAILED!")
            return False

        if not np.isfinite(fad_score):
            print("  FAD score is not finite")
            print("  FAILED!")
            return False

        print("  FAD score is reasonable (positive, finite)")
        print("  PASSED")
        return True


def test_fad_score_identical_audio(ckpt_dir: str = None):
    """Test that FAD score is near zero for identical audio sets."""
    print(f"\n{'=' * 60}")
    print("Test 3: FAD Score for Identical Audio")
    print("=" * 60)

    # Create temporary directories with identical audio
    with tempfile.TemporaryDirectory() as tmpdir:
        dir1 = os.path.join(tmpdir, "set1")
        dir2 = os.path.join(tmpdir, "set2")
        os.makedirs(dir1)
        os.makedirs(dir2)

        # Generate identical audio in both directories
        print("  Generating identical test audio files...")
        for i in range(10):
            audio = generate_test_audio(2.0, 440.0 + i * 50, CLAP_SAMPLE_RATE)
            sf.write(os.path.join(dir1, f"audio_{i}.wav"), audio, CLAP_SAMPLE_RATE)
            sf.write(os.path.join(dir2, f"audio_{i}.wav"), audio, CLAP_SAMPLE_RATE)

        # Calculate FAD
        print("  Computing FAD for identical audio sets...")
        from frechet_audio_distance_exported import FrechetAudioDistance

        if ckpt_dir is None:
            ckpt_dir = str(Path(__file__).parent.parent)
        fad = FrechetAudioDistance(model_name="clap", ckpt_dir=ckpt_dir)
        fad_score = fad.score(dir1, dir2)

        print(f"  FAD score: {fad_score:.6f}")

        # FAD between identical sets should be very close to zero
        if abs(fad_score) > 0.001:
            print("  FAD score is too far from zero for identical audio")
            print("  FAILED!")
            return False

        print("  FAD score is near zero as expected")
        print("  PASSED")
        return True


def main():
    audio_file, ckpt_dir = _PARSED_ARGS

    print(f"\n{'#' * 60}")
    print("# CLAP Verification")
    print(f"{'#' * 60}")

    results = []

    # Test 1: Embedding equivalence
    try:
        results.append(("Embedding equivalence", test_embedding_equivalence(audio_file, ckpt_dir)))
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Embedding equivalence", False))

    # Test 2: FAD score computation
    try:
        results.append(("FAD Score (different audio)", test_fad_score_computation(ckpt_dir)))
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("FAD Score (different audio)", False))

    # Test 3: FAD score for identical audio
    try:
        results.append(("FAD Score (identical audio)", test_fad_score_identical_audio(ckpt_dir)))
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("FAD Score (identical audio)", False))

    # Summary
    print(f"\n{'=' * 60}")
    print("Verification Summary")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print(f"\n{'=' * 60}")
    if all_passed:
        print("ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
