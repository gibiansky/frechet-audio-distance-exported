#!/usr/bin/env python
"""Export Encodec encoder using torch.export for dependency-free inference.

IMPORTANT: This script requires `encodec` to be installed:
    pip install encodec

This script:
1. Loads the original Encodec model from Meta's encodec package
2. Extracts the encoder portion
3. Exports using torch.export.export() with dynamic batch and sample dimensions
4. Saves the exported model as a .pt2 file

Usage:
    python scripts/export_encodec.py --sample-rate 24000
    python scripts/export_encodec.py --sample-rate 48000
    python scripts/export_encodec.py --all  # Export all variants
"""

import os
import sys
from pathlib import Path

# Parse args BEFORE importing anything that might use argparse
def parse_args():
    """Parse arguments manually."""
    sample_rate = None
    export_all = False
    output_dir = Path(__file__).parent.parent

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg in ("--sample-rate", "-sr") and i + 1 < len(sys.argv):
            sample_rate = int(sys.argv[i + 1])
            if sample_rate not in [24000, 48000]:
                print(f"Error: sample-rate must be 24000 or 48000, got {sample_rate}")
                sys.exit(1)
            i += 2
        elif arg == "--all":
            export_all = True
            i += 1
        elif arg in ("--output-dir", "-o") and i + 1 < len(sys.argv):
            output_dir = Path(sys.argv[i + 1])
            i += 2
        elif arg in ("-h", "--help"):
            print(__doc__)
            sys.exit(0)
        else:
            i += 1

    if not export_all and sample_rate is None:
        print("Error: Either --sample-rate or --all must be specified")
        print("Usage: python export_encodec.py --sample-rate 24000")
        print("       python export_encodec.py --all")
        sys.exit(1)

    return sample_rate, export_all, output_dir


# Parse args before any imports
_PARSED_ARGS = parse_args()

import numpy as np
import torch
import torch.nn as nn
from torch.export import export, Dim

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# Maximum audio length for traced model (10 seconds)
# All inputs must be padded to this length before passing to the traced model.
# The output can then be trimmed based on the original input length.
MAX_AUDIO_SECONDS = 10
MAX_SAMPLES_24K = MAX_AUDIO_SECONDS * 24000  # 240000 samples
MAX_SAMPLES_48K = MAX_AUDIO_SECONDS * 48000  # 480000 samples

from frechet_audio_distance_exported.models.encodec import (
    ENCODEC_CONFIGS,
    preprocess_for_encodec,
    pad_to_valid_encodec_length,
)


def load_original_encodec(sample_rate: int):
    """Load the original Encodec model.

    Args:
        sample_rate: One of 24000 or 48000

    Returns:
        Encodec model from the encodec package
    """
    from encodec import EncodecModel

    if sample_rate == 24000:
        print("Loading EncodecModel.encodec_model_24khz()...")
        model = EncodecModel.encodec_model_24khz()
    elif sample_rate == 48000:
        print("Loading EncodecModel.encodec_model_48khz()...")
        model = EncodecModel.encodec_model_48khz()
    else:
        raise ValueError(f"Unsupported sample rate: {sample_rate}")

    model.set_target_bandwidth(24.0)  # Use highest quality
    model.eval()
    return model


def validate_encoder(encoder, sample_rate: int):
    """Validate that the encoder produces expected output.

    Args:
        encoder: Encodec encoder module
        sample_rate: Sample rate for generating test audio

    Returns:
        True if validation passes
    """
    print("Validating encoder output...")

    config = ENCODEC_CONFIGS[sample_rate]
    channels = config["channels"]
    hop_length = config["hop_length"]

    # Generate test audio (2 seconds)
    duration = 2.0
    num_samples = int(sample_rate * duration)
    # Make divisible by hop_length
    num_samples = (num_samples // hop_length) * hop_length

    # Generate sine wave
    t = np.linspace(0, duration, num_samples, dtype=np.float32)
    audio = np.sin(2 * np.pi * 440 * t) * 0.5

    # Preprocess
    preprocessed = preprocess_for_encodec(
        audio, sample_rate, target_sample_rate=sample_rate, target_channels=channels
    )

    # Run encoder
    with torch.no_grad():
        output = encoder(preprocessed)

    # Expected output shape: [batch, 128, time_frames]
    expected_time = num_samples // hop_length
    expected_shape = (1, 128, expected_time)

    print(f"  Input shape: {preprocessed.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected shape: {expected_shape}")

    if output.shape != expected_shape:
        print(f"  Shape mismatch! Expected {expected_shape}, got {output.shape}")
        return False

    # Check output is not all zeros or NaN
    if torch.isnan(output).any():
        print("  Output contains NaN values!")
        return False
    if (output == 0).all():
        print("  Output is all zeros!")
        return False

    print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")
    print("  Validation PASSED")
    return True


def export_encoder(encoder, output_path, sample_rate: int, validate: bool = True):
    """Export the encoder using torch.jit.trace.

    Note: torch.export.export() fails for Encodec due to math.ceil operations
    in the padding logic. We use torch.jit.trace instead.

    The traced model expects inputs padded to a fixed length (10 seconds).
    Shorter inputs should be zero-padded before passing to the model,
    and the output should be trimmed based on the original length.

    Args:
        encoder: Encodec encoder module
        output_path: Path to save the exported .pt file
        sample_rate: Sample rate (for validation)
        validate: Whether to validate the export

    Returns:
        True if export succeeds
    """
    print(f"Exporting encoder to {output_path}...")

    encoder.eval()
    config = ENCODEC_CONFIGS[sample_rate]
    channels = config["channels"]
    hop_length = config["hop_length"]

    # Use fixed length for tracing (10 seconds)
    if sample_rate == 24000:
        max_samples = MAX_SAMPLES_24K
    else:
        max_samples = MAX_SAMPLES_48K

    # Example input for tracing at max length
    example_input = torch.randn(1, channels, max_samples)

    # Use JIT trace directly (torch.export fails due to math.ceil in Encodec)
    print("  Running torch.jit.trace...")
    try:
        traced = torch.jit.trace(encoder, example_input)
        # Change extension from .pt2 to .pt for JIT models
        jit_path = str(output_path).replace('.pt2', '.pt')
        traced.save(jit_path)
        output_path = Path(jit_path)
        print(f"  Saved traced model to: {output_path}")
    except Exception as e:
        print(f"  torch.jit.trace failed: {e}")
        return False

    print(f"  File size: {output_path.stat().st_size / (1024*1024):.1f} MB")
    print(f"  Max input length: {max_samples} samples ({MAX_AUDIO_SECONDS} seconds)")

    if validate:
        print("  Validating exported model...")
        loaded_model = torch.jit.load(str(output_path))

        # Test with the same fixed length (traced model only works with this length)
        test_input = torch.randn(2, channels, max_samples)
        with torch.no_grad():
            orig_out = encoder(test_input)
            loaded_out = loaded_model(test_input)

        max_diff = (orig_out - loaded_out).abs().max().item()
        print(f"    samples={max_samples}: max diff = {max_diff:.2e}")

        if max_diff > 1e-5:
            print("    Export validation FAILED!")
            return False

        print("  Export validation PASSED")

    return True


def export_encodec(sample_rate: int, output_dir: Path):
    """Export Encodec encoder for a specific sample rate.

    Args:
        sample_rate: One of 24000 or 48000
        output_dir: Directory to save exported model
    """
    print(f"\n{'='*60}")
    print(f"Exporting Encodec encoder for {sample_rate}Hz")
    print(f"{'='*60}")

    # Load original model
    model = load_original_encodec(sample_rate)

    # Extract encoder
    encoder = model.encoder
    encoder.eval()

    # Validate encoder
    if not validate_encoder(encoder, sample_rate):
        print(f"Encoder validation failed for {sample_rate}Hz. Aborting.")
        return False

    # Export
    sr_suffix = f"{sample_rate // 1000}k"
    output_path = output_dir / f"encodec_{sr_suffix}_exported.pt2"

    if not export_encoder(encoder, output_path, sample_rate):
        print(f"Export failed for {sample_rate}Hz.")
        return False

    print(f"\nExport complete for {sample_rate}Hz!")
    print(f"Exported model saved to: {output_path}")

    return True


def main():
    sample_rate, export_all, output_dir = _PARSED_ARGS

    sample_rates = [24000, 48000] if export_all else [sample_rate]

    results = []
    for sr in sample_rates:
        success = export_encodec(sr, output_dir)
        results.append((sr, success))

    # Summary
    print(f"\n{'='*60}")
    print("Export Summary")
    print(f"{'='*60}")
    for sr, success in results:
        status = "SUCCESS" if success else "FAILED"
        print(f"  {sr}Hz: {status}")

    if all(s for _, s in results):
        print("\nAll exports completed successfully!")
        sys.exit(0)
    else:
        print("\nSome exports failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
