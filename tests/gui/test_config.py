"""ConfigStore tests for DeepFilterNet GUI."""

from __future__ import annotations

import json
from pathlib import Path

from gui.core.config import DEFAULTS, ConfigStore


def test_roundtrip_preserves_values(tmp_path: Path) -> None:
    """set 3 keys, save, new store load, values equal."""
    cs = ConfigStore(tmp_path / "config.json")
    cs.set("model", "DeepFilterNet3")
    cs.set("epoch", "best")
    cs.set("post_filter", True)
    cs.save()
    cs2 = ConfigStore(tmp_path / "config.json")
    loaded = cs2.load()
    assert loaded["model"] == "DeepFilterNet3"
    assert loaded["epoch"] == "best"
    assert loaded["post_filter"] is True


def test_unknown_keys_dropped_on_load(tmp_path: Path) -> None:
    """file with {"bogus": 1} -> load() has no "bogus."""
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({"bogus": 1}))
    cs = ConfigStore(path=cfg_file)
    loaded = cs.load()
    assert "bogus" not in loaded
    assert loaded["model"] == "DeepFilterNet3"


def test_missing_keys_defaulted_on_load(tmp_path: Path) -> None:
    """file {} -> load() == DEFAULTS subset."""
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({}))
    cs = ConfigStore(path=cfg_file)
    loaded = cs.load()
    for key in DEFAULTS:
        assert loaded[key] == DEFAULTS[key], f"Missing default for {key}"


def test_corrupt_config_backed_up_and_defaults_returned(tmp_path: Path) -> None:
    """file "not json" -> load()==defaults, path.with_suffix(".bak") exists with original bytes."""
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text("not json")
    bak = cfg_file.with_suffix(".bak")
    cs = ConfigStore(path=cfg_file)
    loaded = cs.load()
    assert loaded == dict(DEFAULTS)
    assert bak.exists()
    assert bak.read_bytes() == b"not json"


def test_save_leaves_valid_json(tmp_path: Path) -> None:
    """after set+save, json.loads(path.read_text()) works."""
    cfg_file = tmp_path / "config.json"
    cs = ConfigStore(path=cfg_file)
    cs.set("model", "DeepFilterNet3")
    cs.set("epoch", "best")
    cs.save()
    text = cfg_file.read_text()
    parsed = json.loads(text)
    assert parsed["model"] == "DeepFilterNet3"
    assert parsed["epoch"] == "best"


def test_default_path_under_platformdirs() -> None:
    """ConfigStore().path name == "config.json", parent endswith "deepfilternet-gui."""
    cs = ConfigStore()
    p = cs.path
    assert p.name == "config.json"
    assert p.parent.name.endswith("deepfilternet-gui")
