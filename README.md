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

### VGGish (Default)

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

### PANN (CNN14)

PANN (Pretrained Audio Neural Networks) models are also supported with three sample rate variants:

```python
from frechet_audio_distance_exported import FrechetAudioDistance

# PANN at 16kHz (most common)
fad = FrechetAudioDistance(model_name="pann-16k")
score = fad.score("path/to/background/", "path/to/eval/")

# PANN at 8kHz (for low-quality or telephony audio)
fad_8k = FrechetAudioDistance(model_name="pann-8k")

# PANN at 32kHz (for high-fidelity audio)
fad_32k = FrechetAudioDistance(model_name="pann-32k")
```

**Choosing the right sample rate:**
- **8kHz**: Telephony, low-bitrate audio, or bandwidth-limited applications
- **16kHz**: General purpose, speech, most generated audio
- **32kHz**: High-fidelity audio, music with rich high frequencies

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

| Model | Status | Sample Rates | Embedding Dim | Original Dependency |
|-------|--------|--------------|---------------|---------------------|
| VGGish | Supported | 16kHz | 128 | torch.hub (harritaylor/torchvggish) |
| PANN CNN14 | Supported | 8kHz, 16kHz, 32kHz | 2048 | torchlibrosa |
| CLAP | Planned | - | - | laion_clap, transformers |
| EnCodec | Planned | - | - | encodec |

### PANN Model Details

PANN (Pretrained Audio Neural Networks) is a family of convolutional neural networks trained on AudioSet. The CNN14 architecture provides high-quality audio embeddings:

| Variant | Model Name | Sample Rate | FFT Window | Hop Size | Mel Range |
|---------|------------|-------------|------------|----------|-----------|
| 8kHz | `pann-8k` | 8000 Hz | 256 | 80 | 50-4000 Hz |
| 16kHz | `pann-16k` | 16000 Hz | 512 | 160 | 50-8000 Hz |
| 32kHz | `pann-32k` | 32000 Hz | 1024 | 320 | 50-14000 Hz |

**When to use PANN vs VGGish:**
- **PANN**: Higher-dimensional embeddings (2048 vs 128), better for capturing fine-grained audio differences
- **VGGish**: More widely used, smaller embeddings, faster computation

## Dependency Comparison

| Dependency | Original | Exported |
|------------|----------|----------|
| torch | Required | Required |
| numpy | Required | Required |
| scipy | Required | Required |
| resampy | Required | Required |
| soundfile | Required | Required |
| tqdm | Required | Required |
| librosa | Required (for PANN) | Required (for PANN) |
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

### Export PANN

```bash
# Export all PANN variants (8k, 16k, 32k)
python scripts/export_pann.py --all

# Or export a specific sample rate
python scripts/export_pann.py --sample-rate 16000
python scripts/export_pann.py --sample-rate 8000
python scripts/export_pann.py --sample-rate 32000
```

This will:
1. Download the PANN CNN14 checkpoint from Zenodo (~342 MB each)
2. Transfer weights to a clean PANNCore module
3. Validate weight transfer (embedding diff < 1e-4)
4. Export using `torch.export.export()` with dynamic batch/time
5. Save as `.pt2` files (e.g., `pann_cnn14_16k_exported.pt2`)

### Verify Equivalence

To verify that the exported models produce identical results:

```bash
# Verify VGGish
python scripts/verify_export.py --ckpt-dir /path/to/exported/models

# Verify PANN (all variants)
python scripts/verify_pann.py --all --ckpt-dir /path/to/exported/models

# Verify specific PANN variant
python scripts/verify_pann.py --sample-rate 16000
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

### PANN CNN14 Model

- **Input**: 8kHz, 16kHz, or 32kHz mono audio (depending on variant)
- **Preprocessing**: Log-mel spectrogram (64 mel bands)
  - Uses librosa for STFT and mel filterbank
  - Parameters vary by sample rate (see table above)
- **Architecture**: 6 convolutional blocks (1→64→128→256→512→1024→2048), global mean+max pooling, FC layer
- **Output**: 2048-dimensional embedding per audio file
- **Export format**: `torch.export` (.pt2)
- **Reference**: Kong et al., "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition" (2020)

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
│   ├── fad.py                        # Main FAD class (multi-model support)
│   └── models/
│       ├── __init__.py
│       ├── vggish.py                 # VGGishCore + preprocessing
│       └── pann.py                   # PANNCore + preprocessing (librosa-based)
│
├── scripts/                          # Dev scripts (require frechet_audio_distance)
│   ├── export_vggish.py              # Creates vggish_exported.pt2
│   ├── export_pann.py                # Creates pann_cnn14_*_exported.pt2
│   ├── verify_export.py              # Verifies VGGish equivalence
│   └── verify_pann.py                # Verifies PANN equivalence
│
└── tests/                            # Tests (do NOT require frechet_audio_distance)
    ├── test_basic.py                 # VGGish functionality tests
    └── test_pann.py                  # PANN functionality tests
```

## License

MIT License (same as original frechet-audio-distance)

## Credits

- Original [frechet-audio-distance](https://github.com/gudgud96/frechet-audio-distance) by gudgud96
- VGGish model from [torchvggish](https://github.com/harritaylor/torchvggish) by harritaylor
- Based on [VGGish](https://github.com/tensorflow/models/tree/master/research/audioset/vggish) from Google/AudioSet
- PANN models from [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn) by Kong et al.
