"""End-to-end enhance pipeline test using the real DeepFilterNet model on CPU.

The suite's only ``e2e``-marked test: CI deselects it with ``-m "not e2e"``.
"""

import time
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from gui.core.backend import InProcessBackend, JobConfig
from gui.core.events import EventBus
from gui.core.jobs import JobQueue, JobState

SR = 48000
REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
MODEL_NAME = "DeepFilterNet3"  # JobConfig's default model


@lru_cache(maxsize=1)
def model_available() -> bool:
    """True when the repo ships the model zip or the model can be downloaded.

    Checked once per process: the local ``models/`` zip first (offline-capable),
    then a single bounded probe of the upstream model URL.
    """
    if (MODELS_DIR / f"{MODEL_NAME}.zip").is_file():
        return True
    import urllib.request

    url = f"https://github.com/Rikorose/DeepFilterNet/raw/main/models/{MODEL_NAME}.zip"
    try:
        with urllib.request.urlopen(url, timeout=5):
            return True
    except OSError:
        return False


def _seed_model_cache() -> None:
    """Extract the repo's model zip into the df cache when the cache is empty.

    ``df.enhance.maybe_download_model`` only consults the user cache and the
    network. Seeding the cache from ``models/`` makes the e2e run reproducible
    offline; when the zip is absent the download path stays intact.
    """
    from df.utils import get_cache_dir

    cache_dir = Path(get_cache_dir())
    model_dir = cache_dir / MODEL_NAME
    if (model_dir / "config.ini").is_file() or (model_dir / "checkpoints").is_dir():
        return  # already cached
    zip_path = MODELS_DIR / f"{MODEL_NAME}.zip"
    if not zip_path.is_file():
        return  # no local zip: maybe_download_model will fetch it
    cache_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(cache_dir)


@pytest.mark.e2e
@pytest.mark.skipif(not model_available(), reason="model not available offline")
def test_real_enhance_roundtrip(tmp_path):
    """1.5s noise+tone -> real model on CPU -> DONE, same frames, changed samples."""
    from df.io import load_audio, save_audio

    _seed_model_cache()

    seconds = 1.5
    n = int(seconds * SR)
    t = np.linspace(0, seconds, n, endpoint=False, dtype=np.float64)
    rng = np.random.default_rng(0)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.2 * rng.standard_normal(n)
    src = tmp_path / "noise_tone.wav"
    save_audio(str(src), audio.astype(np.float32)[np.newaxis, :], sr=SR)

    out_dir = tmp_path / "out"
    cfg = JobConfig(device="CPU", chunk_mode="preset", chunk_size_s=1, output_dir=str(out_dir))
    queue = JobQueue(InProcessBackend(), EventBus())
    queue.start()

    started = time.monotonic()
    job = queue.submit([src], cfg)
    # Real model load plus CPU inference needs far more headroom than the 30s
    # default; the bound only guards against a hang, not slow inference.
    assert queue.wait_idle(300.0), "queue should go idle after the real-model run"
    queue.shutdown()
    print(f"e2e queue roundtrip took {time.monotonic() - started:.1f}s")

    assert (
        job.state == JobState.DONE
    ), f"expected DONE, got {job.state} (error={job.error!r}, results={job.file_results})"
    out_file = out_dir / "noise_tone-deep-filtered.wav"
    assert out_file.exists(), f"output file missing: {list(out_dir.glob('*'))}"

    out, out_info = load_audio(str(out_file), verbose=False)
    inp, inp_info = load_audio(str(src), verbose=False)
    out_np, inp_np = out.numpy(), inp.numpy()
    assert (
        out_info.num_frames == inp_info.num_frames == n
    ), f"frames out={out_info.num_frames} in={inp_info.num_frames} expected={n}"
    assert np.isfinite(out_np).all(), "output contains non-finite samples"
    assert not np.allclose(out_np, inp_np), "output must differ from input (model did not run)"
