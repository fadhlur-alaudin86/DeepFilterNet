"""ConfigStore for DeepFilterNet GUI — atomic persistence with defaults."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import platformdirs

DEFAULTS: dict[str, Any] = {
    "model": "DeepFilterNet3",
    "epoch": "best",
    "post_filter": True,
    "atten_lim_db": None,
    "device": "Auto",
    "no_df_stage": False,
    "delay_compensation": True,
    "chunk_mode": "auto",
    "chunk_size_s": 60,
    "output_dir": "./out",
    "output_format": "wav",
    "suffix_enabled": True,
    "log_level": "INFO",
    "log_view_level": "INFO",
    "theme_mode": "SYSTEM",
    "window_width": 1100,
    "window_height": 760,
    "last_view": "enhance",
}


class ConfigStore:
    def __init__(self, path: Path | None = None) -> None:
        if path is None:
            path = Path(platformdirs.user_config_dir("deepfilternet-gui")) / "config.json"
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> dict:
        if not self._path.exists():
            return dict(DEFAULTS)
        try:
            data = json.loads(self._path.read_text())
        except json.JSONDecodeError:
            # Back up original bytes and return defaults
            backup = self._path.with_suffix(".bak")
            backup.write_bytes(self._path.read_bytes())
            return dict(DEFAULTS)
        # Drop unknown keys, fill missing defaults
        result: dict[str, Any] = {}
        for key in DEFAULTS:
            if key in data:
                result[key] = data[key]
            else:
                result[key] = DEFAULTS[key]
        return result

    def get(self, key: str) -> Any:
        data = self.load()
        return data.get(key)

    def set(self, key: str, value: Any) -> None:
        data = self.load()
        data[key] = value
        self._save_data(data)

    def save(self) -> None:
        data = self.load()
        self._save_data(data)

    def _save_data(self, data: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(str(tmp), str(self._path))
