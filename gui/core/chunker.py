"""In-memory audio chunk planner — pure numpy/stdlib, no df/torch imports."""

import numpy as np

AUTO_THRESHOLD_S = 120
AUTO_CHUNK_S = 60
MIN_CHUNK_S = 5
MAX_CHUNK_S = 600


def validate_chunk_size(seconds: int) -> int:
    """Validate chunk size in seconds. Raises ValueError if outside 5..600."""
    if seconds < MIN_CHUNK_S or seconds > MAX_CHUNK_S:
        raise ValueError(
            f"Chunk size must be between {MIN_CHUNK_S} and {MAX_CHUNK_S} seconds, got {seconds}"
        )
    return seconds


def plan_chunks(
    n_samples: int, sr: int, mode: str, size_s: int = AUTO_CHUNK_S
) -> list[tuple[int, int]]:
    """Plan chunk boundaries for audio segmentation.

    Returns list of (start, end) sample index tuples.
    """
    if mode not in ("auto", "preset"):
        raise ValueError(f"Invalid mode '{mode}'. Use 'auto' or 'preset'.")

    if mode == "auto":
        duration_s = n_samples / sr
        if duration_s <= AUTO_THRESHOLD_S:
            return [(0, n_samples)]
        step = AUTO_CHUNK_S * sr
    else:  # preset
        step = size_s * sr

    chunks: list[tuple[int, int]] = []
    cursor = 0
    while cursor < n_samples:
        end = min(cursor + step, n_samples)
        chunks.append((cursor, end))
        cursor = end

    return chunks


def concat_chunks(parts: list[np.ndarray]) -> np.ndarray:
    """Concatenate a list of audio chunk arrays along the last axis."""
    return np.concatenate(parts, axis=-1)
