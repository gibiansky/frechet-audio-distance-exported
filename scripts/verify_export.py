#!/usr/bin/env python
"""Verify that exported FAD produces equivalent results to original package.

IMPORTANT: This script requires `frechet-audio-distance` to be installed:
    pip install frechet-audio-distance

This script tests:
1. Embedding equivalence: Compare embeddings from original and exported models
2. FAD score equivalence: Compare FAD scores computed by both implementations
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_test_audio(duration: float, freq: float, sample_rate: int = 16000) -> np.ndarray:
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


def test_preprocessing_equivalence():
    """Test that local preprocessing matches original torchvggish preprocessing."""
    print("\n" + "=" * 60)
    print("Test 1: Preprocessing Equivalence")
    print("=" * 60)

    from frechet_audio_distance_exported.models.vggish import waveform_to_examples

    # Import original torchvggish (loaded via torch.hub)
    import torch
    original = torch.hub.load('harritaylor/torchvggish', 'vggish')

    # Generate test audio
    audio = generate_test_audio(2.0, 440.0)

    # Get preprocessed patches from both
    local_patches = waveform_to_examples(audio, 16000, return_tensor=True)

    # Original preprocessing
    from torchvggish import vggish_input
    orig_patches = vggish_input.waveform_to_examples(audio, 16000, return_tensor=True)

    # Compare
    local_np = local_patches.detach().numpy()
    orig_np = orig_patches.detach().numpy()

    max_diff = np.abs(local_np - orig_np).max()
    mean_diff = np.abs(local_np - orig_np).mean()

    print(f"  Local patches shape: {local_np.shape}")
    print(f"  Original patches shape: {orig_np.shape}")
    print(f"  Max difference: {max_diff:.2e}")
    print(f"  Mean difference: {mean_diff:.2e}")

    if max_diff < 1e-5:
        print("  PASSED")
        return True
    else:
        print("  FAILED!")
        return False


def test_embedding_equivalence(audio_file: str = None, ckpt_dir: str = None):
    """Test that embeddings from exported model match original."""
    print("\n" + "=" * 60)
    print("Test 2: Embedding Equivalence")
    print("=" * 60)

    import torch
    from frechet_audio_distance import FrechetAudioDistance as OriginalFAD
    from frechet_audio_distance_exported import FrechetAudioDistance as ExportedFAD
    from frechet_audio_distance_exported.models.vggish import waveform_to_examples

    # Load audio
    if audio_file and os.path.exists(audio_file):
        print(f"  Using audio file: {audio_file}")
        audio, sr = sf.read(audio_file, dtype='float32')
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        if sr != 16000:
            import resampy
            audio = resampy.resample(audio, sr, 16000)
    else:
        print("  Using generated sine wave audio")
        audio = generate_test_audio(2.0, 440.0)

    print(f"  Audio length: {len(audio) / 16000:.2f}s")

    # Initialize original FAD
    print("  Loading original model...")
    original_fad = OriginalFAD(model_name="vggish", use_pca=False, use_activation=False)

    # Initialize exported FAD
    print("  Loading exported model...")
    if ckpt_dir is None:
        ckpt_dir = str(Path(__file__).parent.parent)
    exported_fad = ExportedFAD(
        model_name="vggish",
        ckpt_dir=ckpt_dir
    )

    # Get embeddings from original
    print("  Computing original embeddings...")
    with torch.no_grad():
        orig_embd = original_fad.model.forward(audio, 16000)
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


def test_fad_score_equivalence(ckpt_dir: str = None):
    """Test that FAD scores match between original and exported implementations."""
    print("\n" + "=" * 60)
    print("Test 3: FAD Score Equivalence")
    print("=" * 60)

    from frechet_audio_distance import FrechetAudioDistance as OriginalFAD
    from frechet_audio_distance_exported import FrechetAudioDistance as ExportedFAD

    # Create temporary directories with test audio
    with tempfile.TemporaryDirectory() as tmpdir:
        bg_dir = os.path.join(tmpdir, "background")
        eval_dir = os.path.join(tmpdir, "eval")
        os.makedirs(bg_dir)
        os.makedirs(eval_dir)

        # Generate background audio (440 Hz sine waves with variations)
        print("  Generating test audio files...")
        for i in range(5):
            audio = generate_test_audio(2.0, 440.0 + i * 10)
            sf.write(os.path.join(bg_dir, f"bg_{i}.wav"), audio, 16000)

        # Generate eval audio (880 Hz sine waves - different frequency)
        for i in range(5):
            audio = generate_test_audio(2.0, 880.0 + i * 10)
            sf.write(os.path.join(eval_dir, f"eval_{i}.wav"), audio, 16000)

        # Calculate FAD with original
        print("  Computing FAD with original implementation...")
        original_fad = OriginalFAD(model_name="vggish", use_pca=False, use_activation=False)
        orig_score = original_fad.score(bg_dir, eval_dir)

        # Calculate FAD with exported
        print("  Computing FAD with exported implementation...")
        if ckpt_dir is None:
            ckpt_dir = str(Path(__file__).parent.parent)
        exported_fad = ExportedFAD(
            model_name="vggish",
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


def test_with_real_audio(audio_file: str, ckpt_dir: str = None):
    """Test embedding equivalence with real audio file."""
    print("\n" + "=" * 60)
    print("Test 4: Real Audio Embedding Equivalence")
    print("=" * 60)

    if not os.path.exists(audio_file):
        print(f"  Audio file not found: {audio_file}")
        print("  SKIPPED")
        return True

    return test_embedding_equivalence(audio_file, ckpt_dir)


def main():
    # Parse args manually to avoid conflicts with laion_clap's argparse
    import sys
    audio_file = os.path.expanduser("~/Downloads/audio_diffusion_195101_08792205c6c191814139.wav")
    skip_preprocessing = False
    ckpt_dir = None

    # Simple argument parsing
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--audio-file" and i + 1 < len(sys.argv):
            audio_file = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--ckpt-dir" and i + 1 < len(sys.argv):
            ckpt_dir = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--no-preprocessing":
            skip_preprocessing = True
            i += 1
        else:
            i += 1

    print("=" * 60)
    print("Exported FAD Verification")
    print("=" * 60)

    results = []

    # Test 1: Preprocessing equivalence
    if not skip_preprocessing:
        try:
            results.append(("Preprocessing", test_preprocessing_equivalence()))
        except Exception as e:
            print(f"  Error: {e}")
            results.append(("Preprocessing", False))

    # Test 2: Embedding equivalence (with generated audio)
    try:
        results.append(("Embeddings (synthetic)", test_embedding_equivalence(ckpt_dir=ckpt_dir)))
    except Exception as e:
        print(f"  Error: {e}")
        results.append(("Embeddings (synthetic)", False))

    # Test 3: FAD score equivalence
    try:
        results.append(("FAD Score", test_fad_score_equivalence(ckpt_dir=ckpt_dir)))
    except Exception as e:
        print(f"  Error: {e}")
        results.append(("FAD Score", False))

    # Test 4: Real audio (if available)
    if audio_file:
        try:
            results.append(("Embeddings (real audio)", test_with_real_audio(audio_file, ckpt_dir)))
        except Exception as e:
            print(f"  Error: {e}")
            results.append(("Embeddings (real audio)", False))

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
