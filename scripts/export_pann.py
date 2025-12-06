#!/usr/bin/env python
"""Export PANN CNN14 model using torch.export for dependency-free inference.

IMPORTANT: This script requires `frechet-audio-distance` to be installed:
    pip install frechet-audio-distance

This script:
1. Loads the original PANN model from checkpoint
2. Creates a PANNCore instance with matching architecture
3. Transfers weights from original to PANNCore
4. Exports using torch.export.export() with dynamic batch size
5. Saves the exported model as a .pt2 file

Usage:
    python scripts/export_pann.py --sample-rate 16000
    python scripts/export_pann.py --sample-rate 8000
    python scripts/export_pann.py --sample-rate 32000
    python scripts/export_pann.py --all  # Export all variants
"""

import os
import sys
from pathlib import Path

# Parse args BEFORE importing anything that might use argparse (like laion_clap)
def parse_args():
    """Parse arguments manually to avoid conflicts with laion_clap's argparse."""
    sample_rate = None
    export_all = False
    output_dir = Path(__file__).parent.parent
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
            export_all = True
            i += 1
        elif arg in ("--output-dir", "-o") and i + 1 < len(sys.argv):
            output_dir = Path(sys.argv[i + 1])
            i += 2
        elif arg == "--ckpt-dir" and i + 1 < len(sys.argv):
            ckpt_dir = sys.argv[i + 1]
            i += 2
        elif arg in ("-h", "--help"):
            print(__doc__)
            sys.exit(0)
        else:
            i += 1

    if not export_all and sample_rate is None:
        print("Error: Either --sample-rate or --all must be specified")
        print("Usage: python export_pann.py --sample-rate 16000")
        print("       python export_pann.py --all")
        sys.exit(1)

    # Clear sys.argv to prevent laion_clap from parsing it
    sys.argv = [sys.argv[0]]

    return sample_rate, export_all, output_dir, ckpt_dir


# Parse args before any imports
_PARSED_ARGS = parse_args()

import numpy as np
import torch
from torch.export import export, Dim

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from frechet_audio_distance_exported.models.pann import PANNCore, waveform_to_logmel, PANN_CONFIGS


# PANN checkpoint URLs from Zenodo
PANN_CHECKPOINT_URLS = {
    8000: "https://zenodo.org/record/3987831/files/Cnn14_8k_mAP%3D0.416.pth",
    16000: "https://zenodo.org/record/3987831/files/Cnn14_16k_mAP%3D0.438.pth",
    32000: "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth",
}

PANN_CHECKPOINT_NAMES = {
    8000: "Cnn14_8k_mAP=0.416.pth",
    16000: "Cnn14_16k_mAP=0.438.pth",
    32000: "Cnn14_mAP=0.431.pth",
}


def load_original_pann(sample_rate: int, ckpt_dir: str = None):
    """Load the original PANN model from checkpoint.

    Args:
        sample_rate: One of 8000, 16000, or 32000
        ckpt_dir: Directory to store/load checkpoints

    Returns:
        Original PANN model
    """
    from frechet_audio_distance.models.pann import Cnn14, Cnn14_8k, Cnn14_16k

    config = PANN_CONFIGS[sample_rate]

    # Create model
    if sample_rate == 8000:
        model = Cnn14_8k(
            sample_rate=8000,
            window_size=256,
            hop_size=80,
            mel_bins=64,
            fmin=50,
            fmax=4000,
            classes_num=527
        )
    elif sample_rate == 16000:
        model = Cnn14_16k(
            sample_rate=16000,
            window_size=512,
            hop_size=160,
            mel_bins=64,
            fmin=50,
            fmax=8000,
            classes_num=527
        )
    elif sample_rate == 32000:
        model = Cnn14(
            sample_rate=32000,
            window_size=1024,
            hop_size=320,
            mel_bins=64,
            fmin=50,
            fmax=14000,
            classes_num=527
        )
    else:
        raise ValueError(f"Unsupported sample rate: {sample_rate}")

    # Download checkpoint if needed
    if ckpt_dir is None:
        ckpt_dir = os.path.join(torch.hub.get_dir(), "pann_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_name = PANN_CHECKPOINT_NAMES[sample_rate]
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)

    if not os.path.exists(ckpt_path):
        print(f"Downloading PANN checkpoint for {sample_rate}Hz...")
        url = PANN_CHECKPOINT_URLS[sample_rate]
        torch.hub.download_url_to_file(url, ckpt_path, progress=True)

    # Load checkpoint
    print(f"Loading checkpoint from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    return model


def transfer_weights(original, target):
    """Transfer weights from original PANN to PANNCore.

    The original model has:
    - spectrogram_extractor (not needed)
    - logmel_extractor (not needed)
    - spec_augmenter (not needed)
    - bn0, conv_block1-6, fc1 (need to transfer)
    - fc_audioset (not needed)

    Args:
        original: Original PANN model
        target: PANNCore model
    """
    print("Transferring weights...")

    # Transfer bn0
    target.bn0.load_state_dict(original.bn0.state_dict())
    print("  bn0: copied")

    # Transfer conv blocks
    for i in range(1, 7):
        orig_block = getattr(original, f'conv_block{i}')
        tgt_block = getattr(target, f'conv_block{i}')
        tgt_block.load_state_dict(orig_block.state_dict())
        print(f"  conv_block{i}: copied")

    # Transfer fc1
    target.fc1.load_state_dict(original.fc1.state_dict())
    print("  fc1: copied")


def validate_weights(original, target, sample_rate: int):
    """Validate that weights were transferred correctly.

    We generate test audio, run it through the original model's preprocessing,
    then compare the CNN outputs.

    Args:
        original: Original PANN model
        target: PANNCore model
        sample_rate: Sample rate for generating test audio

    Returns:
        True if validation passes
    """
    print("Validating weight transfer...")

    # Generate test audio (2 seconds)
    duration = 2.0
    audio = np.sin(2 * np.pi * 440 * np.arange(int(sample_rate * duration)) / sample_rate).astype(np.float32)
    audio_tensor = torch.from_numpy(audio).unsqueeze(0)  # [1, samples]

    with torch.no_grad():
        # Get logmel from original model's preprocessing
        x = original.spectrogram_extractor(audio_tensor)  # [1, 1, time, freq]
        x = original.logmel_extractor(x)  # [1, 1, time, mel_bins]

        # Run through original CNN layers
        orig_x = x.transpose(1, 3)
        orig_x = original.bn0(orig_x)
        orig_x = orig_x.transpose(1, 3)

        orig_x = original.conv_block1(orig_x, pool_size=(2, 2), pool_type='avg')
        orig_x = original.conv_block2(orig_x, pool_size=(2, 2), pool_type='avg')
        orig_x = original.conv_block3(orig_x, pool_size=(2, 2), pool_type='avg')
        orig_x = original.conv_block4(orig_x, pool_size=(2, 2), pool_type='avg')
        orig_x = original.conv_block5(orig_x, pool_size=(2, 2), pool_type='avg')
        orig_x = original.conv_block6(orig_x, pool_size=(1, 1), pool_type='avg')

        orig_x = torch.mean(orig_x, dim=3)
        x1, _ = torch.max(orig_x, dim=2)
        x2 = torch.mean(orig_x, dim=2)
        orig_x = x1 + x2
        orig_output = torch.relu(original.fc1(orig_x))

        # Run through our preprocessing + PANNCore
        our_logmel = waveform_to_logmel(audio, sample_rate, target_sample_rate=sample_rate)
        target_output = target(our_logmel)

    # First check: logmel difference (preprocessing)
    logmel_diff = (x - our_logmel).abs().max().item()
    print(f"  Logmel preprocessing max diff: {logmel_diff:.2e}")

    # Second check: final output difference
    max_diff = (orig_output - target_output).abs().max().item()
    mean_diff = (orig_output - target_output).abs().mean().item()
    print(f"  Embedding max diff: {max_diff:.2e}")
    print(f"  Embedding mean diff: {mean_diff:.2e}")

    # Note: We allow up to 1e-2 difference because librosa and torchlibrosa have
    # slightly different STFT implementations. The mean diff is typically ~1e-4.
    if max_diff < 1e-2:
        print("  Validation PASSED")
        return True
    else:
        print("  Validation FAILED!")
        print(f"  Original output range: [{orig_output.min():.4f}, {orig_output.max():.4f}]")
        print(f"  Target output range: [{target_output.min():.4f}, {target_output.max():.4f}]")
        return False


def export_model(model, output_path, sample_rate: int, validate: bool = True):
    """Export the model using torch.export.export().

    Args:
        model: PANNCore model with loaded weights
        output_path: Path to save the exported .pt2 file
        sample_rate: Sample rate (for validation)
        validate: Whether to validate the export
    """
    print(f"Exporting model to {output_path}...")

    model.eval()

    # Define dynamic shapes for batch and time dimensions.
    # The time dimension has constraints due to pooling layers (5 layers with pool_size=2).
    # torch.export suggests: time = 32*_time - 24 for valid time values.
    # Valid time values: 8, 40, 72, 104, 136, 168, 200, 232, ...
    batch_dim = Dim('batch', min=1, max=1024)
    _time = Dim('_time', min=1, max=313)  # max=313 gives time up to ~10000
    time_dim = 32 * _time - 24  # Derived dimension
    dynamic_shapes = {'x': {0: batch_dim, 2: time_dim}}

    # Example input for tracing (200 = 32*7 - 24, so _time=7)
    example_input = torch.randn(2, 1, 200, 64)

    # Export
    print("  Running torch.export.export()...")
    exported = export(model, (example_input,), dynamic_shapes=dynamic_shapes)

    # Save
    print("  Saving...")
    torch.export.save(exported, str(output_path))

    print(f"  Saved to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / (1024*1024):.1f} MB")

    if validate:
        print("  Validating exported model...")
        loaded = torch.export.load(str(output_path))
        loaded_model = loaded.module()

        # Test with various input sizes (must satisfy time = 32*_time - 24)
        # Valid values: 8, 40, 72, 104, 136, 168, 200, 232, 264, ..., 520, ...
        for time_steps in [104, 520]:  # _time=4 and _time=17
            test_input = torch.randn(3, 1, time_steps, 64)
            with torch.no_grad():
                orig_out = model(test_input)
                loaded_out = loaded_model(test_input)

            max_diff = (orig_out - loaded_out).abs().max().item()
            print(f"    time_steps={time_steps}: max diff = {max_diff:.2e}")

            if max_diff > 1e-5:
                print("    Export validation FAILED!")
                return False

        print("  Export validation PASSED")

    return True


def export_pann(sample_rate: int, output_dir: Path, ckpt_dir: str = None):
    """Export PANN model for a specific sample rate.

    Args:
        sample_rate: One of 8000, 16000, or 32000
        output_dir: Directory to save exported model
        ckpt_dir: Directory for PANN checkpoints
    """
    print(f"\n{'='*60}")
    print(f"Exporting PANN for {sample_rate}Hz")
    print(f"{'='*60}")

    # Load original model
    original = load_original_pann(sample_rate, ckpt_dir)

    # Create target model
    print("Creating PANNCore model...")
    target = PANNCore()
    target.eval()

    # Transfer weights
    transfer_weights(original, target)

    # Validate transfer
    if not validate_weights(original, target, sample_rate):
        print(f"Weight transfer validation failed for {sample_rate}Hz. Aborting.")
        return False

    # Export
    sr_suffix = f"{sample_rate // 1000}k"
    output_path = output_dir / f"pann_cnn14_{sr_suffix}_exported.pt2"
    if not export_model(target, output_path, sample_rate):
        print(f"Export failed for {sample_rate}Hz.")
        return False

    print(f"\nExport complete for {sample_rate}Hz!")
    print(f"Exported model saved to: {output_path}")

    return True


def main():
    # Use pre-parsed args (parsed before imports to avoid laion_clap conflicts)
    sample_rate, export_all, output_dir, ckpt_dir = _PARSED_ARGS

    sample_rates = [8000, 16000, 32000] if export_all else [sample_rate]

    results = []
    for sr in sample_rates:
        success = export_pann(sr, output_dir, ckpt_dir)
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
