"""
Lightweight FAD implementation using exported models.

This module provides a minimal-dependency implementation of Frechet Audio Distance
calculation using exported PyTorch models instead of torch.hub.

Dependencies: torch, numpy, scipy, resampy, soundfile, tqdm
"""

import os
from multiprocessing.dummy import Pool as ThreadPool
from typing import List, Optional

import numpy as np
import resampy
import soundfile as sf
import torch
from scipy import linalg
from tqdm import tqdm

from .models.vggish import waveform_to_examples, SAMPLE_RATE as VGGISH_SAMPLE_RATE
from .models.pann import waveform_to_logmel, PANN_CONFIGS, EMBEDDING_SIZE as PANN_EMBEDDING_SIZE


def _pad_to_valid_pann_time(x: torch.Tensor) -> torch.Tensor:
    """Pad input to a valid time dimension for exported PANN model.

    The exported PANN model requires time = 32*k - 24 for some integer k >= 1.
    Valid values: 8, 40, 72, 104, 136, 168, 200, 232, 264, ...

    Args:
        x: Input tensor of shape [batch, 1, time, 64]

    Returns:
        Padded tensor with valid time dimension
    """
    time = x.shape[2]
    # Find the smallest valid time >= current time
    # time = 32*k - 24, so k = (time + 24) / 32
    k = (time + 24 + 31) // 32  # Round up
    valid_time = 32 * k - 24
    if valid_time < time:
        valid_time += 32  # Safety check

    if valid_time > time:
        # Pad with zeros on the time dimension
        pad_amount = valid_time - time
        x = torch.nn.functional.pad(x, (0, 0, 0, pad_amount))

    return x


# URLs for downloading exported models
EXPORTED_MODEL_URLS = {
    "vggish": "https://github.com/gibiansky/frechet-audio-distance-exported/releases/download/v0.1/vggish_exported.pt2",
    # PANN CNN14 models for different sample rates
    "pann-8k": "https://github.com/gibiansky/frechet-audio-distance-exported/releases/download/v0.2/pann_cnn14_8k_exported.pt2",
    "pann-16k": "https://github.com/gibiansky/frechet-audio-distance-exported/releases/download/v0.2/pann_cnn14_16k_exported.pt2",
    "pann-32k": "https://github.com/gibiansky/frechet-audio-distance-exported/releases/download/v0.2/pann_cnn14_32k_exported.pt2",
}

# Valid model names and their configurations
VALID_MODELS = {
    "vggish": {"sample_rate": 16000, "embedding_dim": 128},
    "pann-8k": {"sample_rate": 8000, "embedding_dim": 2048},
    "pann-16k": {"sample_rate": 16000, "embedding_dim": 2048},
    "pann-32k": {"sample_rate": 32000, "embedding_dim": 2048},
}

# Map PANN model names to their sample rates
PANN_SAMPLE_RATES = {
    "pann-8k": 8000,
    "pann-16k": 16000,
    "pann-32k": 32000,
}


def load_audio(fname: str, sample_rate: int, channels: int, dtype: str = "float32") -> np.ndarray:
    """Load and preprocess audio file.

    Args:
        fname: Path to audio file
        sample_rate: Target sample rate
        channels: Number of channels (1 for mono)
        dtype: Data type for loading

    Returns:
        Audio as numpy array
    """
    wav_data, sr = sf.read(fname, dtype=dtype)

    # Normalize integer audio to [-1.0, +1.0]
    if dtype == 'int16':
        wav_data = wav_data / 32768.0
    elif dtype == 'int32':
        wav_data = wav_data / float(2**31)

    # Convert to mono if needed
    if len(wav_data.shape) > channels:
        wav_data = np.mean(wav_data, axis=1)

    # Resample if needed
    if sr != sample_rate:
        wav_data = resampy.resample(wav_data, sr, sample_rate)

    return wav_data


class FrechetAudioDistance:
    """API-compatible FAD calculator using exported models.

    This class provides the same interface as frechet_audio_distance.FrechetAudioDistance
    but uses exported PyTorch models instead of torch.hub dependencies.

    Minimal dependencies: torch, numpy, scipy, resampy, soundfile, tqdm

    Example:
        >>> fad = FrechetAudioDistance(model_name="vggish")
        >>> score = fad.score("background_audio/", "eval_audio/")
        >>> print(f"FAD Score: {score}")
    """

    def __init__(
        self,
        ckpt_dir: Optional[str] = None,
        model_name: str = "vggish",
        sample_rate: Optional[int] = None,
        channels: int = 1,
        verbose: bool = False,
        audio_load_worker: int = 8,
    ):
        """
        Initialize FAD calculator.

        Args:
            ckpt_dir: Folder where exported models are stored/cached. If None,
                uses torch.hub cache directory. Models are automatically downloaded
                on first use if not present.
            model_name: Model to use. Options:
                - "vggish": VGGish model (128-dim embeddings, 16kHz)
                - "pann-8k": PANN CNN14 at 8kHz (2048-dim embeddings)
                - "pann-16k": PANN CNN14 at 16kHz (2048-dim embeddings)
                - "pann-32k": PANN CNN14 at 32kHz (2048-dim embeddings)
            sample_rate: Sample rate for audio. If None, uses model's default.
                Must match model's expected sample rate.
            channels: Number of channels (1 for mono)
            verbose: Whether to print progress information
            audio_load_worker: Number of threads for audio loading
        """
        if model_name not in VALID_MODELS:
            raise ValueError(
                f"Unknown model: {model_name}. Valid options: {list(VALID_MODELS.keys())}"
            )

        model_config = VALID_MODELS[model_name]
        expected_sr = model_config["sample_rate"]

        # Set sample rate to model default if not specified
        if sample_rate is None:
            sample_rate = expected_sr
        elif sample_rate != expected_sr:
            raise ValueError(
                f"Model '{model_name}' requires sample_rate={expected_sr}, got {sample_rate}"
            )

        self.model_name = model_name
        self.sample_rate = sample_rate
        self.channels = channels
        self.verbose = verbose
        self.audio_load_worker = audio_load_worker

        # Set device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        if self.verbose:
            print(f"[Exported FAD] Using device: {self.device}")

        # Set checkpoint directory
        if ckpt_dir is not None:
            os.makedirs(ckpt_dir, exist_ok=True)
            self.ckpt_dir = ckpt_dir
        else:
            self.ckpt_dir = os.path.join(torch.hub.get_dir(), "exported_fad")
            os.makedirs(self.ckpt_dir, exist_ok=True)

        # Load model
        self._load_model()

    def _load_model(self):
        """Load the exported model, downloading if necessary."""
        # Map model name to exported file name
        if self.model_name == "vggish":
            model_filename = "vggish_exported.pt2"
        elif self.model_name in PANN_SAMPLE_RATES:
            # e.g., "pann-16k" -> "pann_cnn14_16k_exported.pt2"
            sr_suffix = self.model_name.split("-")[1]  # "16k"
            model_filename = f"pann_cnn14_{sr_suffix}_exported.pt2"
        else:
            model_filename = f"{self.model_name}_exported.pt2"

        model_path = os.path.join(self.ckpt_dir, model_filename)

        # Check if model exists locally
        if not os.path.exists(model_path):
            # Try to download from URL
            url = EXPORTED_MODEL_URLS.get(self.model_name)
            if url:
                print(f"[Exported FAD] Downloading {self.model_name} model to {self.ckpt_dir}...")
                torch.hub.download_url_to_file(url, model_path, progress=True)
                print(f"[Exported FAD] Download complete.")
            else:
                raise FileNotFoundError(
                    f"Exported model not found at {model_path} and no download URL available. "
                    f"Please run scripts/export_vggish.py first or provide a valid ckpt_dir."
                )

        if self.verbose:
            print(f"[Exported FAD] Loading model from {model_path}...")

        # Load exported model
        exported = torch.export.load(model_path)
        self.model = exported.module()
        self.model.to(self.device)
        # Note: eval() is not needed/supported for exported models

    def get_embeddings(self, x: List[np.ndarray], sr: int) -> np.ndarray:
        """Get embeddings for a list of audio arrays.

        Args:
            x: List of numpy arrays containing audio samples
            sr: Sample rate of audio

        Returns:
            Concatenated embeddings as numpy array
        """
        embd_lst = []
        is_pann = self.model_name.startswith("pann-")

        for audio in tqdm(x, disable=(not self.verbose)):
            try:
                if is_pann:
                    # PANN preprocessing: waveform -> log-mel spectrogram
                    target_sr = PANN_SAMPLE_RATES[self.model_name]
                    preprocessed = waveform_to_logmel(audio, sr, target_sample_rate=target_sr, return_tensor=True)
                    # Pad to valid time dimension for exported model
                    preprocessed = _pad_to_valid_pann_time(preprocessed)
                    preprocessed = preprocessed.to(self.device)

                    # Run model
                    with torch.no_grad():
                        embd = self.model(preprocessed)

                    # PANN returns one embedding per audio file
                    embd = embd.cpu().numpy()
                else:
                    # VGGish preprocessing: waveform -> mel-spectrogram patches
                    patches = waveform_to_examples(audio, sr, return_tensor=True)
                    patches = patches.to(self.device)

                    # Run model
                    with torch.no_grad():
                        embd = self.model(patches)

                    # Convert to numpy
                    embd = embd.cpu().numpy()

                embd_lst.append(embd)

            except Exception as e:
                if self.verbose:
                    print(f"[Exported FAD] Error processing audio: {e}")
                continue

        if not embd_lst:
            return np.array([])

        return np.concatenate(embd_lst, axis=0)

    def _get_embedding_for_audio(self, audio: np.ndarray) -> np.ndarray:
        """Get embedding for a single audio array (for testing).

        Args:
            audio: Audio as numpy array

        Returns:
            Embeddings as numpy array
        """
        is_pann = self.model_name.startswith("pann-")

        if is_pann:
            target_sr = PANN_SAMPLE_RATES[self.model_name]
            preprocessed = waveform_to_logmel(audio, self.sample_rate, target_sample_rate=target_sr, return_tensor=True)
            # Pad to valid time dimension for exported model
            preprocessed = _pad_to_valid_pann_time(preprocessed)
            preprocessed = preprocessed.to(self.device)

            with torch.no_grad():
                embd = self.model(preprocessed)
        else:
            patches = waveform_to_examples(audio, self.sample_rate, return_tensor=True)
            patches = patches.to(self.device)

            with torch.no_grad():
                embd = self.model(patches)

        return embd.cpu().numpy()

    def calculate_embd_statistics(self, embd_lst: np.ndarray):
        """Calculate mean and covariance of embeddings.

        Args:
            embd_lst: Embeddings array of shape (n_samples, embedding_dim)

        Returns:
            Tuple of (mean, covariance)
        """
        if isinstance(embd_lst, list):
            embd_lst = np.array(embd_lst)
        mu = np.mean(embd_lst, axis=0)
        sigma = np.cov(embd_lst, rowvar=False)
        return mu, sigma

    def calculate_frechet_distance(
        self,
        mu1: np.ndarray,
        sigma1: np.ndarray,
        mu2: np.ndarray,
        sigma2: np.ndarray,
        eps: float = 1e-6
    ) -> float:
        """
        Calculate Frechet Distance between two multivariate Gaussians.

        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is:
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2))

        Adapted from: https://github.com/mseitzer/pytorch-fid

        Args:
            mu1: Mean of first distribution
            sigma1: Covariance matrix of first distribution
            mu2: Mean of second distribution
            sigma2: Covariance matrix of second distribution
            eps: Small value for numerical stability

        Returns:
            Frechet distance
        """
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2).astype(complex), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('FID calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset).astype(complex))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError(f'Imaginary component {m}')
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

    def _load_audio_files(self, dir: str, dtype: str = "float32") -> List[np.ndarray]:
        """Load all audio files from a directory.

        Args:
            dir: Directory containing audio files
            dtype: Data type for loading audio

        Returns:
            List of audio arrays
        """
        task_results = []

        pool = ThreadPool(self.audio_load_worker)
        files = [f for f in os.listdir(dir) if not f.startswith('.')]
        pbar = tqdm(total=len(files), disable=(not self.verbose))

        def update(*a):
            pbar.update()

        if self.verbose:
            print(f"[Exported FAD] Loading audio from {dir}...")

        for fname in files:
            res = pool.apply_async(
                load_audio,
                args=(os.path.join(dir, fname), self.sample_rate, self.channels, dtype),
                callback=update,
            )
            task_results.append(res)

        pool.close()
        pool.join()
        pbar.close()

        return [k.get() for k in task_results]

    def score(
        self,
        background_dir: str,
        eval_dir: str,
        background_embds_path: Optional[str] = None,
        eval_embds_path: Optional[str] = None,
        dtype: str = "float32"
    ) -> float:
        """
        Compute Frechet Audio Distance between two directories of audio files.

        Args:
            background_dir: Path to directory containing background audio files
            eval_dir: Path to directory containing evaluation audio files
            background_embds_path: Path to save/load background embeddings (.npy)
            eval_embds_path: Path to save/load evaluation embeddings (.npy)
            dtype: Data type for loading audio

        Returns:
            FAD score, or -1 if an error occurred
        """
        try:
            # Load or compute background embeddings
            if background_embds_path and os.path.exists(background_embds_path):
                if self.verbose:
                    print(f"[Exported FAD] Loading embeddings from {background_embds_path}...")
                embds_background = np.load(background_embds_path)
            else:
                audio_background = self._load_audio_files(background_dir, dtype=dtype)
                embds_background = self.get_embeddings(audio_background, sr=self.sample_rate)
                if background_embds_path:
                    os.makedirs(os.path.dirname(background_embds_path), exist_ok=True)
                    np.save(background_embds_path, embds_background)

            # Load or compute eval embeddings
            if eval_embds_path and os.path.exists(eval_embds_path):
                if self.verbose:
                    print(f"[Exported FAD] Loading embeddings from {eval_embds_path}...")
                embds_eval = np.load(eval_embds_path)
            else:
                audio_eval = self._load_audio_files(eval_dir, dtype=dtype)
                embds_eval = self.get_embeddings(audio_eval, sr=self.sample_rate)
                if eval_embds_path:
                    os.makedirs(os.path.dirname(eval_embds_path), exist_ok=True)
                    np.save(eval_embds_path, embds_eval)

            # Check if embeddings are empty
            if len(embds_background) == 0:
                print("[Exported FAD] Background set dir is empty, exiting...")
                return -1
            if len(embds_eval) == 0:
                print("[Exported FAD] Eval set dir is empty, exiting...")
                return -1

            # Compute statistics and FAD score
            mu_background, sigma_background = self.calculate_embd_statistics(embds_background)
            mu_eval, sigma_eval = self.calculate_embd_statistics(embds_eval)

            fad_score = self.calculate_frechet_distance(
                mu_background,
                sigma_background,
                mu_eval,
                sigma_eval
            )

            return fad_score

        except Exception as e:
            print(f"[Exported FAD] An error occurred: {e}")
            return -1
