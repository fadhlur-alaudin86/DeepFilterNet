"""Model registry for DeepFilterNet GUI."""

from pathlib import Path

PRESETS: tuple[str, ...] = ("DeepFilterNet", "DeepFilterNet2", "DeepFilterNet3")


class ModelError(Exception):
    """Raised when a model operation fails."""


def resolve_model_dir(model: str) -> str:
    """Resolve a model name to its directory path.

    If *model* is a known preset, lazy-download it via ``maybe_download_model``.
    Otherwise treat *model* as a custom path and verify it contains ``config.ini``.
    """
    from importlib import import_module

    _df_enhance = import_module("df.enhance")  # lazy import, no eager df import
    maybe_download_model = _df_enhance.maybe_download_model

    if model in PRESETS:
        return maybe_download_model(model)

    # Custom path: must contain config.ini
    p = Path(model)
    if not p.is_dir() or not (p / "config.ini").is_file():
        raise ModelError(
            f"Custom path {model!r} is not a valid model directory (missing config.ini)"
        )
    return str(p)


def ensure_model(model: str, attempts: int = 3) -> str:
    """Retry wrapper for model resolution.

    Attempts up to *attempts* times, catching ``ModelError``,
    ``SystemExit`` and ``OSError`` with 1s/2s backoff.
    Re-raises ``ModelError`` after all attempts are exhausted.
    """
    import time

    last_exception: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return resolve_model_dir(model)
        except (ModelError, SystemExit, OSError) as exc:
            last_exception = exc
            backoff = 1 if attempt < attempts else 0
            time.sleep(backoff)
    raise ModelError(
        f"Failed to resolve model {model!r} after {attempts} attempts"
    ) from last_exception


def model_choices() -> list[str]:
    """Return the list of model choices for the GUI."""
    return list(PRESETS) + ["Custom path"]
