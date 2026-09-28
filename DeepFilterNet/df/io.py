import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torchaudio as ta
from loguru import logger
from numpy import ndarray
import numpy as np
from torch import Tensor

from df.logger import warn_once
from df.utils import download_file, get_cache_dir, get_git_root

# --- soundfile import (independent of AudioMetaData chain) ---
try:
    import soundfile as sf
except ImportError:
    sf = None

# --- AudioMetaData import chain and resample constants ---
# Three-tier import for AudioMetaData, with constants determined by the tier.

# Tier 1: Try new torchaudio where AudioMetaData is at top level
try:
    from torchaudio import AudioMetaData

    # Modern torchaudio >= 2.9 uses these sinc constants
    SINC = "sinc_interp_hann"
    KAISER = "sinc_interp_kaiser"

# Tier 2: Try legacy torchaudio where AudioMetaData is under backend.common
except ImportError:
    try:
        from torchaudio.backend.common import AudioMetaData  # type: ignore

        # Legacy torchaudio < 2.9 uses these constants
        SINC = "sinc_interpolation"
        KAISER = "kaiser_window"

    # Tier 3: Define minimal AudioMetaData dataclass fallback
    # This tier is reached when neither import works (e.g., torchaudio 2.11+)
    # removed both the top-level and backend.common AudioMetaData
    except ImportError:

        @dataclass
        class AudioMetaData:  # type: ignore
            sample_rate: int
            num_frames: int = 0
            num_channels: int = 0
            bits_per_sample: int = 0
            encoding: str = ""

        # torchaudio 2.11 uses modern constants; keep them consistent
        SINC = "sinc_interp_hann"
        KAISER = "sinc_interp_kaiser"


def load_audio(
    file: str, sr: Optional[int] = None, verbose=True, **kwargs
) -> Tuple[Tensor, AudioMetaData]:
    """Loads an audio file using soundfile (preferred) or torchaudio (fallback).

    Args:
        file (str): Path to an audio file.
        sr (int): Optionally resample audio to specified target sampling rate.
        **kwargs: Passed to the underlying loader. The resample method
            may be set via `method` which is passed to `resample()`.

    Returns:
        audio (Tensor): Audio tensor of shape [C, T], if channels_first=True (default).
        info (AudioMetaData): Meta data of the original audio file. Contains the original sr.
    """
    ikwargs = {}
    if "format" in kwargs:
        ikwargs["format"] = kwargs["format"]
    rkwargs = {}
    if "method" in kwargs:
        rkwargs["method"] = kwargs.pop("method")

    # Prefer soundfile: supports more formats and is independent of torchaudio backend changes
    if sf is not None:
        try:
            info_sf = sf.info(file)
            orig_sr = info_sf.samplerate
            info = AudioMetaData(
                sample_rate=orig_sr,
                num_frames=info_sf.frames,
                num_channels=info_sf.channels,
            )

            num_frames = kwargs.get("num_frames", -1)
            frame_offset = kwargs.get("frame_offset", 0)
            if "num_frames" in kwargs and sr is not None:
                num_frames = num_frames * orig_sr // sr

            data, orig_sr = sf.read(file, start=frame_offset, frames=num_frames, dtype="float32")
            audio = torch.from_numpy(data)
            if audio.ndim == 1:
                audio = audio.unsqueeze(0)
            else:
                audio = audio.t()

            if sr is not None and orig_sr != sr:
                if verbose:
                    warn_once(
                        f"Audio sampling rate does not match model sampling rate ({orig_sr}, {sr}). "
                        "Resampling..."
                    )
                audio = resample(audio, orig_sr, sr, **rkwargs)
            return audio.contiguous(), info
        except Exception as e:
            if verbose:
                logger.warning(f"Soundfile failed to load audio ({e}). Falling back to torchaudio.")

    # Fallback to torchaudio if soundfile is unavailable or failed
    try:
        info = ta.info(file, **ikwargs)
    except (AttributeError, ImportError):
        # torchaudio.info may not be available in some builds; load a 1-frame stub to get sr
        try:
            temp_audio, orig_sr = ta.load(file, num_frames=1)
            info = AudioMetaData(sample_rate=orig_sr)
        except Exception:
            # Last resort: create minimal metadata
            info = AudioMetaData(sample_rate=48000)

    if "num_frames" in kwargs and sr is not None:
        kwargs["num_frames"] *= info.sample_rate // sr
    audio, orig_sr = ta.load(file, **kwargs)
    if sr is not None and orig_sr != sr:
        if verbose:
            warn_once(
                f"Audio sampling rate does not match model sampling rate ({orig_sr}, {sr}). "
                "Resampling..."
            )
        audio = resample(audio, orig_sr, sr, **rkwargs)
    return audio.contiguous(), info


def save_audio(
    file: str,
    audio: Union[Tensor, ndarray],
    sr: int,
    output_dir: Optional[str] = None,
    suffix: Optional[str] = None,
    log: bool = False,
    dtype=torch.int16,
):
    outpath = file
    if suffix is not None:
        file, ext = os.path.splitext(file)
        outpath = file + f"_{suffix}" + ext
    if output_dir is not None:
        outpath = os.path.join(output_dir, os.path.basename(outpath))
    if log:
        logger.info(f"Saving audio file '{outpath}'")

    # Prefer soundfile for saving: avoids torchaudio dtype/backend quirks
    if sf is not None:
        try:
            audio_np = torch.as_tensor(audio).cpu().numpy()
            if audio_np.ndim == 2:
                audio_np = audio_np.T
            if dtype == torch.int16:
                if audio_np.dtype == np.int16:
                    pcm16 = audio_np
                else:
                    # Scale float audio in [-1, 1] to PCM_16.
                    pcm16 = (np.clip(audio_np, -1.0, 1.0) * 32767.0).astype(np.int16)
                sf.write(outpath, pcm16, sr, subtype="PCM_16")
            else:
                if audio_np.dtype != np.float32:
                    audio_np = audio_np.astype(np.float32)
                sf.write(outpath, audio_np, sr, subtype="FLOAT")
            return
        except Exception as e:
            logger.warning(f"Soundfile failed to save audio ({e}). Falling back to torchaudio.")

    # Fallback to torchaudio
    audio = torch.as_tensor(audio)
    if audio.ndim == 1:
        audio.unsqueeze_(0)

    # torchaudio.save >= 2.9 (torchcodec backend) only supports float32 in range [-1.0, 1.0]
    if audio.dtype != torch.float32:
        if audio.dtype == torch.int16:
            audio = audio.to(torch.float32) / 32768.0
        elif audio.dtype == torch.int32:
            audio = audio.to(torch.float32) / 2147483648.0
        else:
            audio = audio.to(torch.float32)

    # Clamp to [-1.0, 1.0] to avoid clipping artifacts
    max_val = audio.abs().max()
    if max_val > 1.0:
        audio = audio / max_val

    ta.save(outpath, audio, sr)


try:
    from torchaudio.functional import resample as ta_resample
except ImportError:
    from torchaudio.compliance.kaldi import resample_waveform as ta_resample  # type: ignore


def get_resample_params(method: str) -> Dict[str, Any]:
    params = {
        "sinc_fast": {"resampling_method": SINC, "lowpass_filter_width": 16},
        "sinc_best": {"resampling_method": SINC, "lowpass_filter_width": 64},
        "kaiser_fast": {
            "resampling_method": KAISER,
            "lowpass_filter_width": 16,
            "rolloff": 0.85,
            "beta": 8.555504641634386,
        },
        "kaiser_best": {
            "resampling_method": KAISER,
            "lowpass_filter_width": 16,
            "rolloff": 0.9475937167399596,
            "beta": 14.769656459379492,
        },
    }
    assert method in params.keys(), f"method must be one of {list(params.keys())}"
    return params[method]


def resample(audio: Tensor, orig_sr: int, new_sr: int, method="sinc_fast"):
    params = get_resample_params(method)
    return ta_resample(audio, orig_sr, new_sr, **params)


def get_test_sample(sr: int = 48000) -> Tensor:
    dir = get_git_root()
    file_path = os.path.join("assets", "clean_freesound_33711.wav")
    if dir is None:
        url = "https://github.com/Rikorose/DeepFilterNet/raw/main/" + file_path
        save_dir = get_cache_dir()
        path = download_file(url, save_dir)
    else:
        path = os.path.join(dir, file_path)
    sample, _ = load_audio(path, sr=sr)
    return sample
