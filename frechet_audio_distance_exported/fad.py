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


# URLs for downloading exported models
EXPORTED_MODEL_URLS = {
    "vggish": None,  # Will be set when hosted
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
        sample_rate: int = 16000,
        channels: int = 1,
        verbose: bool = False,
        audio_load_worker: int = 8,
    ):
        """
        Initialize FAD calculator.

        Args:
            ckpt_dir: Folder where exported models are stored
            model_name: Model to use (currently only "vggish" supported)
            sample_rate: Sample rate for audio (must be 16000 for vggish)
            channels: Number of channels (1 for mono)
            verbose: Whether to print progress information
            audio_load_worker: Number of threads for audio loading
        """
        assert model_name == "vggish", "Only 'vggish' model is currently supported"
        assert sample_rate == 16000, "Sample rate must be 16000 for VGGish"

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
        """Load the exported model."""
        model_path = os.path.join(self.ckpt_dir, f"{self.model_name}_exported.pt2")

        # Check if model exists locally
        if not os.path.exists(model_path):
            # Try to download from URL
            url = EXPORTED_MODEL_URLS.get(self.model_name)
            if url:
                if self.verbose:
                    print(f"[Exported FAD] Downloading {self.model_name} model...")
                torch.hub.download_url_to_file(url, model_path)
            else:
                raise FileNotFoundError(
                    f"Exported model not found at {model_path}. "
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

        for audio in tqdm(x, disable=(not self.verbose)):
            try:
                # Preprocess audio to mel-spectrogram patches
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
