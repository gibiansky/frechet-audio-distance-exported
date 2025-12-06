# frechet-audio-distance-exported

> **Warning:** This package is 100% AI-written with only minimal human supervision; use at your own risk.

A lightweight implementation of Frechet Audio Distance (FAD) using exported PyTorch models.

## Why This Package?

The original [`frechet-audio-distance`](https://github.com/gudgud96/frechet-audio-distance) package provides excellent FAD calculation but requires heavy dependencies:

- **transformers** (~500MB) - for CLAP model
- **laion_clap** - CLAP audio-text model
- **encodec** - Meta's audio codec
- **fairseq** (optional) - for some models
- **torchlibrosa** - for PANN model

This package provides the **same functionality** with **minimal dependencies** by using `torch.export` to serialize neural network models into self-contained `.pt2` files.

## How It Works

This package uses PyTorch's `torch.export` API to serialize trained neural network models into standalone files. The key insight is:

1. **Neural networks** can be exported with `torch.export.export()` into `.pt2` files
2. **Preprocessing code** (mel-spectrogram computation) is pure NumPy/Python and is included directly
3. The exported model contains all weights and computation graph - no external downloads needed

The result is a package that can compute FAD scores without needing torch.hub, transformers, or any model repositories.

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/gibiansky/frechet-audio-distance-exported.git
```

Or for development:

```bash
git clone https://github.com/gibiansky/frechet-audio-distance-exported.git
cd frechet-audio-distance-exported
pip install -e ".[dev]"
```

## Usage

```python
from frechet_audio_distance_exported import FrechetAudioDistance

# Initialize - model is automatically downloaded on first use
fad = FrechetAudioDistance(model_name="vggish")

# Compute FAD between two directories of audio files
score = fad.score("path/to/background/", "path/to/eval/")
print(f"FAD Score: {score}")
```

The exported model (~275 MB) is automatically downloaded to the torch hub cache directory on first use. You can specify a custom cache directory with `ckpt_dir`:

```python
fad = FrechetAudioDistance(model_name="vggish", ckpt_dir="/path/to/cache")
```

### API Compatibility

This package provides the same interface as the original `frechet_audio_distance.FrechetAudioDistance`:

```python
# Original package
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance(model_name="vggish", use_pca=False, use_activation=False)
score = fad.score(bg_dir, eval_dir)

# This package (equivalent)
from frechet_audio_distance_exported import FrechetAudioDistance
fad = FrechetAudioDistance(model_name="vggish", ckpt_dir="/path/to/models")
score = fad.score(bg_dir, eval_dir)
```

## Supported Models

| Model | Status | Original Dependency |
|-------|--------|---------------------|
| VGGish | Supported | torch.hub (harritaylor/torchvggish) |
| PANN | Planned | torchlibrosa |
| CLAP | Planned | laion_clap, transformers |
| EnCodec | Planned | encodec |

## Dependency Comparison

| Dependency | Original | Exported |
|------------|----------|----------|
| torch | Required | Required |
| numpy | Required | Required |
| scipy | Required | Required |
| resampy | Required | Required |
| soundfile | Required | Required |
| tqdm | Required | Required |
| transformers | Required (~500MB) | Not needed |
| laion_clap | Required | Not needed |
| encodec | Required | Not needed |
| torchlibrosa | Required | Not needed |
| fairseq | Optional | Not needed |

## Generating Exported Models

The exported `.pt2` model files need to be generated once using the original `frechet-audio-distance` package. The `scripts/` directory contains tools for this:

### Prerequisites

```bash
pip install frechet-audio-distance
```

### Export VGGish

```bash
python scripts/export_vggish.py --output vggish_exported.pt2
```

This will:
1. Download the original VGGish model from torch.hub
2. Transfer weights to a clean PyTorch module
3. Export using `torch.export.export()` with dynamic batch size
4. Save as a `.pt2` file (~275 MB)

### Verify Equivalence

To verify that the exported model produces identical results:

```bash
python scripts/verify_export.py --ckpt-dir /path/to/exported/models
```

This runs several tests:
- **Preprocessing**: Verifies mel-spectrogram computation matches original
- **Embeddings**: Compares embedding vectors from both implementations
- **FAD Score**: Computes FAD with both and verifies they match

## Technical Details

### VGGish Model

- **Input**: 16kHz mono audio
- **Preprocessing**: Log-mel spectrogram (96 frames x 64 mel bands)
  - Window: 25ms, Hop: 10ms
  - Mel range: 125 Hz - 7500 Hz
- **Architecture**: VGG-style CNN + FC layers
- **Output**: 128-dimensional embeddings per 0.96s audio chunk
- **Export format**: `torch.export` (.pt2)

### FAD Computation

The Frechet Audio Distance is computed as:

```
FAD = ||mu_bg - mu_eval||^2 + Tr(Sigma_bg + Sigma_eval - 2*sqrt(Sigma_bg * Sigma_eval))
```

Where:
- `mu_bg`, `mu_eval` are mean embedding vectors
- `Sigma_bg`, `Sigma_eval` are covariance matrices
- Lower FAD indicates more similar audio distributions

## Project Structure

```
frechet-audio-distance-exported/
├── pyproject.toml                    # Package configuration
├── README.md                         # This file
├── LICENSE                           # MIT license
│
├── frechet_audio_distance_exported/  # Main package
│   ├── __init__.py                   # Exports FrechetAudioDistance
│   ├── fad.py                        # Main FAD class
│   └── models/
│       ├── __init__.py
│       └── vggish.py                 # VGGishCore + preprocessing
│
├── scripts/                          # Dev scripts (require frechet_audio_distance)
│   ├── export_vggish.py              # Creates vggish_exported.pt2
│   └── verify_export.py              # Verifies equivalence with original
│
└── tests/                            # Tests (do NOT require frechet_audio_distance)
    └── test_basic.py                 # Basic functionality tests
```

## License

MIT License (same as original frechet-audio-distance)

## Credits

- Original [frechet-audio-distance](https://github.com/gudgud96/frechet-audio-distance) by gudgud96
- VGGish model from [torchvggish](https://github.com/harritaylor/torchvggish) by harritaylor
- Based on [VGGish](https://github.com/tensorflow/models/tree/master/research/audioset/vggish) from Google/AudioSet
