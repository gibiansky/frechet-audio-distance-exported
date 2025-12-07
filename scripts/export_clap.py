#!/usr/bin/env python
"""Export CLAP audio encoder model using torch.export for dependency-free inference.

IMPORTANT: This script requires `laion_clap` to be installed:
    pip install laion_clap

This script:
1. Downloads the CLAP 630k-audioset checkpoint (HTSAT-tiny audio encoder)
2. Creates a wrapper model that takes mel-spectrogram input
3. Validates against original laion_clap output
4. Exports using torch.export.export()
5. Saves the exported model as a .pt2 file

The exported model takes mel-spectrogram input [B, 1, 1001, 64] and outputs
L2-normalized 512-dim embeddings [B, 512].

Note: The time dimension is fixed to 1001 frames (10 seconds at 48kHz with hop=480).
Shorter audio should be padded with zeros.

Usage:
    python scripts/export_clap.py
    python scripts/export_clap.py --output clap_exported.pt2
"""

import os
import sys
from pathlib import Path

# Parse args BEFORE importing anything that might use argparse (like laion_clap)
def parse_args():
    """Parse arguments manually to avoid conflicts with laion_clap's argparse."""
    output_path = None
    output_dir = Path(__file__).parent.parent

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg in ("--output", "-o") and i + 1 < len(sys.argv):
            output_path = sys.argv[i + 1]
            i += 2
        elif arg == "--output-dir" and i + 1 < len(sys.argv):
            output_dir = Path(sys.argv[i + 1])
            i += 2
        elif arg in ("-h", "--help"):
            print(__doc__)
            sys.exit(0)
        else:
            i += 1

    # Clear sys.argv to prevent laion_clap from parsing it
    sys.argv = [sys.argv[0]]

    return output_path, output_dir


# Parse args before any imports
_PARSED_ARGS = parse_args()

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.export import export

# Fix for PyTorch 2.6+ compatibility with laion_clap checkpoints
_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from frechet_audio_distance_exported.models.clap import CLAP_SAMPLE_RATE


# Time dimension for 10 seconds at 48kHz with hop_size=480
CLAP_TIME_FRAMES = 1001


class CLAPAudioEmbedder(nn.Module):
    """Wrapper for CLAP audio encoder that takes mel-spectrogram input.

    This wrapper:
    1. Applies bn0 normalization to the mel-spectrogram
    2. Reshapes the mel-spectrogram to HTSAT's expected 256x256 image format
    3. Runs through HTSAT's forward_features
    4. Projects to 512-dim embedding
    5. L2 normalizes the output

    Input: [B, 1, T, 64] log-mel spectrogram (before bn0 normalization)
    Output: [B, 512] L2-normalized embeddings
    """

    def __init__(self, audio_branch, audio_projection):
        super().__init__()
        self.audio_branch = audio_branch
        self.audio_projection = audio_projection

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            mel_spec: [B, 1, T, 64] log-mel spectrogram

        Returns:
            [B, 512] L2-normalized embeddings
        """
        B, C, T, freq = mel_spec.shape

        # HTSAT config
        freq_ratio = 4
        spec_size = 256
        target_T = spec_size * freq_ratio  # 1024
        target_F = spec_size // freq_ratio  # 64

        # Interpolate time dimension if needed
        if T < target_T:
            mel_spec = F.interpolate(mel_spec, (target_T, freq), mode='bicubic', align_corners=True)
        if freq < target_F:
            mel_spec = F.interpolate(mel_spec, (mel_spec.shape[2], target_F), mode='bicubic', align_corners=True)

        # Apply bn0 normalization (B, 1, T, 64 -> B, 64, T, 1 -> bn0 -> B, 1, T, 64)
        x = mel_spec.transpose(1, 3)
        x = self.audio_branch.bn0(x)
        x = x.transpose(1, 3)

        # Reshape to HTSAT image format (same as reshape_wav2img)
        B, C, T, freq = x.shape
        x = x.permute(0, 1, 3, 2)  # B C F T
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], freq_ratio, x.shape[3] // freq_ratio)
        x = x.permute(0, 1, 3, 2, 4)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3], x.shape[4])

        # Forward through HTSAT
        output = self.audio_branch.forward_features(x)
        embedding = output['embedding']

        # Project and normalize
        embedding = self.audio_projection(embedding)
        embedding = F.normalize(embedding, dim=-1)

        return embedding


def load_original_clap():
    """Load the original CLAP model from laion_clap.

    Returns:
        Tuple of (model, model_cfg)
    """
    from laion_clap.clap_module import create_model
    from laion_clap.clap_module.factory import load_state_dict
    import laion_clap

    print("Creating CLAP model (630k-audioset, HTSAT-tiny)...")

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
        print("\nDownload complete!")

    print(f"Loading checkpoint from {ckpt_path}...")
    ckpt = load_state_dict(ckpt_path, skip_params=True)

    # Load with strict=False to handle version mismatches
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")

    model.eval()
    return model, model_cfg


def validate_weights(original_model, wrapper):
    """Validate that the wrapper produces equivalent output to the original.

    Args:
        original_model: Original CLAP model from laion_clap
        wrapper: CLAPAudioEmbedder wrapper

    Returns:
        True if validation passes
    """
    from laion_clap.training.data import get_audio_features
    from dataclasses import asdict

    print("Validating wrapper against original model...")

    # Generate test audio (10 seconds at 48kHz - max CLAP length)
    duration = 10.0
    sample_rate = CLAP_SAMPLE_RATE
    audio = np.sin(2 * np.pi * 440 * np.arange(int(sample_rate * duration)) / sample_rate).astype(np.float32)

    # Quantize like CLAP does
    audio_quantized = (audio * 32767.0).astype(np.int16).astype(np.float32) / 32767.0
    audio_tensor = torch.from_numpy(audio_quantized).float()

    # Get original embedding
    with torch.no_grad():
        audio_cfg_dict = asdict(original_model.audio_cfg)
        temp_dict = {}
        temp_dict = get_audio_features(
            temp_dict, audio_tensor, 480000,
            data_truncating='fusion',
            data_filling='repeatpad',
            audio_cfg=audio_cfg_dict
        )

        input_dict = {}
        for k, v in temp_dict.items():
            input_dict[k] = v.unsqueeze(0)

        orig_output = original_model.encode_audio(input_dict, device='cpu')
        orig_embedding = orig_output['embedding']
        orig_projected = original_model.audio_projection(orig_embedding)
        orig_final = F.normalize(orig_projected, dim=-1)

    # Get mel-spectrogram using HTSAT's extractors
    with torch.no_grad():
        waveform = input_dict['waveform']  # [1, 480000]
        x = original_model.audio_branch.spectrogram_extractor(waveform)
        mel = original_model.audio_branch.logmel_extractor(x)

        # Feed to wrapper
        wrapper_final = wrapper(mel)

    # Compare
    max_diff = (orig_final - wrapper_final).abs().max().item()
    mean_diff = (orig_final - wrapper_final).abs().mean().item()
    cosine_sim = F.cosine_similarity(orig_final, wrapper_final).item()

    print(f"  Original embedding range: [{orig_final.min():.4f}, {orig_final.max():.4f}]")
    print(f"  Wrapper embedding range: [{wrapper_final.min():.4f}, {wrapper_final.max():.4f}]")
    print(f"  Max difference: {max_diff:.2e}")
    print(f"  Mean difference: {mean_diff:.2e}")
    print(f"  Cosine similarity: {cosine_sim:.6f}")

    if cosine_sim > 0.999:
        print("  Validation PASSED")
        return True, mel
    else:
        print("  Validation FAILED!")
        return False, None


def export_model(wrapper, mel_example, output_path, validate: bool = True):
    """Export the model using torch.export.export().

    Args:
        wrapper: CLAPAudioEmbedder model
        mel_example: Example mel-spectrogram for export
        output_path: Path to save the exported .pt2 file
        validate: Whether to validate the export
    """
    print(f"Exporting model to {output_path}...")

    wrapper.eval()

    # Export with the actual mel-spectrogram dimensions
    print(f"  Input shape: {mel_example.shape}")
    print("  Running torch.export.export()...")

    # Create example input with correct shape
    example_input = torch.randn_like(mel_example)
    exported = export(wrapper, (example_input,))

    # Save
    print("  Saving...")
    output_path = Path(output_path)
    torch.export.save(exported, str(output_path))

    print(f"  Saved to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / (1024*1024):.1f} MB")

    if validate:
        print("  Validating exported model...")
        loaded = torch.export.load(str(output_path))
        loaded_model = loaded.module()

        with torch.no_grad():
            orig_out = wrapper(mel_example)
            loaded_out = loaded_model(mel_example)

        max_diff = (orig_out - loaded_out).abs().max().item()
        print(f"  Max diff after reload: {max_diff:.2e}")

        if max_diff > 1e-5:
            print("  Export validation FAILED!")
            return False

        print("  Export validation PASSED")

    return True


def main():
    output_path, output_dir = _PARSED_ARGS

    print(f"\n{'='*60}")
    print("Exporting CLAP (HTSAT-tiny 630k-audioset)")
    print(f"{'='*60}")

    # Load original model
    original_model, model_cfg = load_original_clap()

    # Create wrapper
    print("Creating CLAPAudioEmbedder wrapper...")
    wrapper = CLAPAudioEmbedder(
        original_model.audio_branch,
        original_model.audio_projection
    )
    wrapper.eval()

    # Validate wrapper against original
    passed, mel_example = validate_weights(original_model, wrapper)
    if not passed:
        print("Wrapper validation failed. Aborting.")
        sys.exit(1)

    # Export
    if output_path is None:
        output_path = output_dir / "clap_exported.pt2"
    else:
        output_path = Path(output_path)

    if not export_model(wrapper, mel_example, output_path):
        print("Export failed.")
        sys.exit(1)

    print(f"\nExport complete!")
    print(f"Exported model saved to: {output_path}")
    print(f"\nNote: The exported model expects input shape [B, 1, {CLAP_TIME_FRAMES}, 64]")


if __name__ == "__main__":
    main()
