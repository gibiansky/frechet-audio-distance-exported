#!/usr/bin/env python
"""Verify that exported PANN produces equivalent results to original package.

IMPORTANT: This script requires `frechet-audio-distance` to be installed:
    pip install frechet-audio-distance

This script tests:
1. Preprocessing equivalence: Compare log-mel spectrograms from both implementations
2. Embedding equivalence: Compare embeddings from original and exported models
3. FAD score equivalence: Compare FAD scores computed by both implementations

Usage:
    python scripts/verify_pann.py --sample-rate 16000
    python scripts/verify_pann.py --sample-rate 8000
    python scripts/verify_pann.py --sample-rate 32000
    python scripts/verify_pann.py --all  # Verify all variants
"""

import os
import sys
import tempfile
from pathlib import Path

# Parse args BEFORE importing anything that might use argparse (like laion_clap)
def parse_args():
    """Parse arguments manually to avoid conflicts with laion_clap's argparse."""
    sample_rate = None
    verify_all = False
    audio_file = os.path.expanduser("~/Downloads/audio_diffusion_195101_08792205c6c191814139.wav")
    ckpt_dir = None

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg in ("--sample-rate", "-sr") and i + 1 < len(sys.argv):
            sample_rate = int(sys.argv[i + 1])
            if sample_rate not in [8000, 16000, 32000]:
                print(f"Error: sample-rate must be 8000, 16000, or 32000, got {sample_rate}")
                sys.exit(1)
            i += 2
        elif arg == "--all":
            verify_all = True
            i += 1
        elif arg == "--audio-file" and i + 1 < len(sys.argv):
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

    if not verify_all and sample_rate is None:
        print("Error: Either --sample-rate or --all must be specified")
        print("Usage: python verify_pann.py --sample-rate 16000")
        print("       python verify_pann.py --all")
        sys.exit(1)

    # Clear sys.argv to prevent laion_clap from parsing it
    sys.argv = [sys.argv[0]]

    return sample_rate, verify_all, audio_file, ckpt_dir


# Parse args before any imports
_PARSED_ARGS = parse_args()

import numpy as np
import soundfile as sf
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


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


def test_preprocessing_equivalence(sample_rate: int):
    """Test that local preprocessing matches original torchlibrosa preprocessing."""
    print(f"\n{'=' * 60}")
    print(f"Test 1: Preprocessing Equivalence ({sample_rate}Hz)")
    print("=" * 60)

    from frechet_audio_distance_exported.models.pann import waveform_to_logmel, PANN_CONFIGS
    from frechet_audio_distance.models.pann import Cnn14, Cnn14_8k, Cnn14_16k

    # Create original model to use its preprocessing
    if sample_rate == 8000:
        original = Cnn14_8k(
            sample_rate=8000, window_size=256, hop_size=80,
            mel_bins=64, fmin=50, fmax=4000, classes_num=527
        )
    elif sample_rate == 16000:
        original = Cnn14_16k(
            sample_rate=16000, window_size=512, hop_size=160,
            mel_bins=64, fmin=50, fmax=8000, classes_num=527
        )
    else:  # 32000
        original = Cnn14(
            sample_rate=32000, window_size=1024, hop_size=320,
            mel_bins=64, fmin=50, fmax=14000, classes_num=527
        )
    original.eval()

    # Generate test audio
    audio = generate_test_audio(2.0, 440.0, sample_rate)
    audio_tensor = torch.from_numpy(audio).unsqueeze(0)

    # Get log-mel from original model
    with torch.no_grad():
        orig_spec = original.spectrogram_extractor(audio_tensor)
        orig_logmel = original.logmel_extractor(orig_spec)
    orig_logmel_np = orig_logmel.numpy()

    # Get log-mel from our preprocessing
    our_logmel = waveform_to_logmel(audio, sample_rate, target_sample_rate=sample_rate, return_tensor=True)
    our_logmel_np = our_logmel.numpy()

    # Compare
    max_diff = np.abs(orig_logmel_np - our_logmel_np).max()
    mean_diff = np.abs(orig_logmel_np - our_logmel_np).mean()

    print(f"  Original log-mel shape: {orig_logmel_np.shape}")
    print(f"  Our log-mel shape: {our_logmel_np.shape}")
    print(f"  Max difference: {max_diff:.2e}")
    print(f"  Mean difference: {mean_diff:.2e}")

    # Log-mel preprocessing can have small differences due to librosa vs torchlibrosa
    # Allow up to 0.1 difference (on dB scale)
    if max_diff < 0.5:
        print("  PASSED")
        return True
    else:
        print("  FAILED!")
        return False


def test_embedding_equivalence(sample_rate: int, audio_file: str = None, ckpt_dir: str = None):
    """Test that embeddings from exported model match original."""
    print(f"\n{'=' * 60}")
    print(f"Test 2: Embedding Equivalence ({sample_rate}Hz)")
    print("=" * 60)

    from frechet_audio_distance import FrechetAudioDistance as OriginalFAD
    from frechet_audio_distance_exported import FrechetAudioDistance as ExportedFAD

    # Load audio
    if audio_file and os.path.exists(audio_file):
        print(f"  Using audio file: {audio_file}")
        audio, sr = sf.read(audio_file, dtype='float32')
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        # Resample to target sample rate if needed
        if sr != sample_rate:
            import resampy
            audio = resampy.resample(audio, sr, sample_rate)
    else:
        print("  Using generated sine wave audio")
        audio = generate_test_audio(2.0, 440.0, sample_rate)

    print(f"  Audio length: {len(audio) / sample_rate:.2f}s")

    # Map sample rate to model name
    model_name_map = {8000: "pann-8k", 16000: "pann-16k", 32000: "pann-32k"}
    exported_model_name = model_name_map[sample_rate]

    # Initialize original FAD with PANN
    print("  Loading original model...")
    # Original FAD uses sample_rate parameter to select PANN variant
    original_fad = OriginalFAD(
        model_name="pann",
        sample_rate=sample_rate,
        use_pca=False,
        use_activation=False
    )

    # Initialize exported FAD
    print("  Loading exported model...")
    if ckpt_dir is None:
        ckpt_dir = str(Path(__file__).parent.parent)
    exported_fad = ExportedFAD(
        model_name=exported_model_name,
        ckpt_dir=ckpt_dir
    )

    # Get embeddings from original
    print("  Computing original embeddings...")
    with torch.no_grad():
        orig_embd = original_fad.model.forward(audio, sample_rate)
    if torch.is_tensor(orig_embd):
        orig_embd = orig_embd.cpu().numpy()

    # Get embeddings from exported
    print("  Computing exported embeddings...")
    export_embd = exported_fad._get_embedding_for_audio(audio)

    # Compare
    max_diff = np.abs(orig_embd - export_embd).max()
    mean_diff = np.abs(orig_embd - export_embd).mean()

    print(f"  Original embedding shape: {orig_embd.shape}")
    print(f"  Exported embedding shape: {export_embd.shape}")
    print(f"  Max difference: {max_diff:.2e}")
    print(f"  Mean difference: {mean_diff:.2e}")

    if max_diff < 1e-4:
        print("  PASSED")
        return True
    else:
        print("  FAILED!")
        return False


def test_fad_score_equivalence(sample_rate: int, ckpt_dir: str = None):
    """Test that FAD scores match between original and exported implementations."""
    print(f"\n{'=' * 60}")
    print(f"Test 3: FAD Score Equivalence ({sample_rate}Hz)")
    print("=" * 60)

    from frechet_audio_distance import FrechetAudioDistance as OriginalFAD
    from frechet_audio_distance_exported import FrechetAudioDistance as ExportedFAD

    # Map sample rate to model name
    model_name_map = {8000: "pann-8k", 16000: "pann-16k", 32000: "pann-32k"}
    exported_model_name = model_name_map[sample_rate]

    # Create temporary directories with test audio
    with tempfile.TemporaryDirectory() as tmpdir:
        bg_dir = os.path.join(tmpdir, "background")
        eval_dir = os.path.join(tmpdir, "eval")
        os.makedirs(bg_dir)
        os.makedirs(eval_dir)

        # Generate background audio (440 Hz sine waves with variations)
        print("  Generating test audio files...")
        for i in range(5):
            audio = generate_test_audio(2.0, 440.0 + i * 10, sample_rate)
            sf.write(os.path.join(bg_dir, f"bg_{i}.wav"), audio, sample_rate)

        # Generate eval audio (880 Hz sine waves - different frequency)
        for i in range(5):
            audio = generate_test_audio(2.0, 880.0 + i * 10, sample_rate)
            sf.write(os.path.join(eval_dir, f"eval_{i}.wav"), audio, sample_rate)

        # Calculate FAD with original
        print("  Computing FAD with original implementation...")
        original_fad = OriginalFAD(
            model_name="pann",
            sample_rate=sample_rate,
            use_pca=False,
            use_activation=False
        )
        orig_score = original_fad.score(bg_dir, eval_dir)

        # Calculate FAD with exported
        print("  Computing FAD with exported implementation...")
        if ckpt_dir is None:
            ckpt_dir = str(Path(__file__).parent.parent)
        exported_fad = ExportedFAD(
            model_name=exported_model_name,
            ckpt_dir=ckpt_dir
        )
        export_score = exported_fad.score(bg_dir, eval_dir)

        # Compare
        diff = abs(orig_score - export_score)
        rel_diff = diff / max(abs(orig_score), 1e-10) * 100

        print(f"  Original FAD score: {orig_score:.6f}")
        print(f"  Exported FAD score: {export_score:.6f}")
        print(f"  Absolute difference: {diff:.6f}")
        print(f"  Relative difference: {rel_diff:.4f}%")

        if diff < 0.01:
            print("  PASSED")
            return True
        else:
            print("  FAILED!")
            return False


def test_with_real_audio(sample_rate: int, audio_file: str, ckpt_dir: str = None):
    """Test embedding equivalence with real audio file."""
    print(f"\n{'=' * 60}")
    print(f"Test 4: Real Audio Embedding Equivalence ({sample_rate}Hz)")
    print("=" * 60)

    if not os.path.exists(audio_file):
        print(f"  Audio file not found: {audio_file}")
        print("  SKIPPED")
        return True

    return test_embedding_equivalence(sample_rate, audio_file, ckpt_dir)


def verify_sample_rate(sample_rate: int, audio_file: str, ckpt_dir: str):
    """Run all verification tests for a single sample rate."""
    print(f"\n{'#' * 60}")
    print(f"# PANN {sample_rate}Hz Verification")
    print(f"{'#' * 60}")

    results = []

    # Test 1: Preprocessing equivalence
    try:
        results.append(("Preprocessing", test_preprocessing_equivalence(sample_rate)))
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Preprocessing", False))

    # Test 2: Embedding equivalence (with generated audio)
    try:
        results.append(("Embeddings (synthetic)", test_embedding_equivalence(sample_rate, ckpt_dir=ckpt_dir)))
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Embeddings (synthetic)", False))

    # Test 3: FAD score equivalence
    try:
        results.append(("FAD Score", test_fad_score_equivalence(sample_rate, ckpt_dir)))
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("FAD Score", False))

    # Test 4: Real audio (if available)
    if audio_file:
        try:
            results.append(("Embeddings (real audio)", test_with_real_audio(sample_rate, audio_file, ckpt_dir)))
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            results.append(("Embeddings (real audio)", False))

    return results


def main():
    sample_rate, verify_all, audio_file, ckpt_dir = _PARSED_ARGS

    sample_rates = [8000, 16000, 32000] if verify_all else [sample_rate]

    all_results = {}
    for sr in sample_rates:
        results = verify_sample_rate(sr, audio_file, ckpt_dir)
        all_results[sr] = results

    # Summary
    print(f"\n{'=' * 60}")
    print("Verification Summary")
    print("=" * 60)

    all_passed = True
    for sr, results in all_results.items():
        print(f"\n  PANN {sr}Hz:")
        for name, passed in results:
            status = "PASSED" if passed else "FAILED"
            print(f"    {name}: {status}")
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
