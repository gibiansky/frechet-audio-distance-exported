#!/usr/bin/env python
"""Export VGGish model using torch.export for dependency-free inference.

IMPORTANT: This script requires `frechet-audio-distance` to be installed:
    pip install frechet-audio-distance

This script:
1. Loads the original VGGish model from torch.hub
2. Creates a VGGishCore instance with matching architecture
3. Transfers weights from original to VGGishCore
4. Exports using torch.export.export() with dynamic batch size
5. Saves the exported model as a .pt2 file
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.export import export, Dim

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from frechet_audio_distance_exported.models.vggish import VGGishCore


def load_original_vggish():
    """Load the original VGGish model from torch.hub."""
    print("Loading original VGGish model from torch.hub...")
    model = torch.hub.load('harritaylor/torchvggish', model='vggish')
    model.eval()
    return model


def transfer_weights(original, target):
    """Transfer weights from original VGGish to VGGishCore.

    The original model has the same architecture, so we can copy weights directly.
    We only copy the features and embeddings (without final ReLU).
    """
    print("Transferring weights...")

    # Copy feature layers (convolutional layers)
    target.features.load_state_dict(original.features.state_dict())

    # Copy embedding layers
    # Original embeddings: Linear -> ReLU -> Linear -> ReLU -> Linear -> ReLU
    # Target embeddings: Linear -> ReLU -> Linear -> ReLU -> Linear (no final ReLU)
    # The weights are the same, just different number of modules

    orig_embeddings = original.embeddings
    target_embeddings = target.embeddings

    # Original has 6 modules: [Linear, ReLU, Linear, ReLU, Linear, ReLU]
    # Target has 5 modules: [Linear, ReLU, Linear, ReLU, Linear]

    # Copy Linear layers (indices 0, 2, 4 in original -> 0, 2, 4 in target)
    target_embeddings[0].load_state_dict(orig_embeddings[0].state_dict())
    target_embeddings[2].load_state_dict(orig_embeddings[2].state_dict())
    target_embeddings[4].load_state_dict(orig_embeddings[4].state_dict())

    print("  Features: copied")
    print("  Embeddings: copied (without final ReLU)")


def validate_weights(original, target):
    """Validate that weights were transferred correctly."""
    print("Validating weight transfer...")

    # Create test input
    test_input = torch.randn(5, 1, 96, 64)

    with torch.no_grad():
        # Run through original (bypass preprocessing, manual forward)
        x = original.features(test_input)
        x = torch.transpose(x, 1, 3)
        x = torch.transpose(x, 1, 2)
        x = x.contiguous()
        x = x.view(x.size(0), -1)

        # Apply embeddings without final ReLU
        orig_emb = original.embeddings
        x = orig_emb[0](x)  # Linear
        x = orig_emb[1](x)  # ReLU
        x = orig_emb[2](x)  # Linear
        x = orig_emb[3](x)  # ReLU
        x = orig_emb[4](x)  # Linear (no final ReLU)
        orig_output = x

        # Run through target
        target_output = target(test_input)

    max_diff = (orig_output - target_output).abs().max().item()
    print(f"  Max difference: {max_diff:.2e}")

    if max_diff < 1e-6:
        print("  Validation PASSED")
        return True
    else:
        print("  Validation FAILED!")
        return False


def export_model(model, output_path, validate=True):
    """Export the model using torch.export.export().

    Args:
        model: VGGishCore model with loaded weights
        output_path: Path to save the exported .pt2 file
        validate: Whether to validate the export
    """
    print(f"Exporting model to {output_path}...")

    model.eval()

    # Define dynamic shapes for batch dimension
    # Use Dim.AUTO to let PyTorch infer the dynamic dimension
    batch_dim = Dim('batch', min=1, max=1024)
    dynamic_shapes = {'x': {0: batch_dim}}

    # Example input for tracing - use batch > 1 to avoid specialization
    example_input = torch.randn(2, 1, 96, 64)

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

        test_input = torch.randn(10, 1, 96, 64)
        with torch.no_grad():
            orig_out = model(test_input)
            loaded_out = loaded_model(test_input)

        max_diff = (orig_out - loaded_out).abs().max().item()
        print(f"  Export validation max diff: {max_diff:.2e}")

        if max_diff < 1e-6:
            print("  Export validation PASSED")
        else:
            print("  Export validation FAILED!")


def main():
    parser = argparse.ArgumentParser(
        description="Export VGGish model using torch.export")
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("vggish_exported.pt2"),
        help="Output path for exported model (default: vggish_exported.pt2)")
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation steps")

    args = parser.parse_args()

    # Load original model
    original = load_original_vggish()

    # Create target model
    print("Creating VGGishCore model...")
    target = VGGishCore()
    target.eval()

    # Transfer weights
    transfer_weights(original, target)

    # Validate transfer
    if not args.skip_validation:
        if not validate_weights(original, target):
            print("Weight transfer validation failed. Aborting.")
            sys.exit(1)

    # Export
    export_model(target, args.output, validate=not args.skip_validation)

    print("\nExport complete!")
    print(f"Exported model saved to: {args.output}")


if __name__ == "__main__":
    main()
