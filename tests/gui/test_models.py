"""Tests for the DeepFilterNet model registry."""

import pytest
from pathlib import Path
from importlib import import_module

from gui.core.models import ModelError, PRESETS, resolve_model_dir, ensure_model, model_choices


def _get_enhance_module():
    return import_module("df.enhance")


def test_preset_delegates_to_download(monkeypatch):
    # patch df.enhance.maybe_download_model -> "/x"
    enhance_mod = _get_enhance_module()
    monkeypatch.setattr(enhance_mod, "maybe_download_model", lambda name: "/x")
    result = resolve_model_dir("DeepFilterNet3")
    assert result == "/x"


def test_custom_path_requires_config_ini(tmp_path):
    # dir without config.ini -> ModelError
    bad_dir = tmp_path / "no_config"
    bad_dir.mkdir()
    with pytest.raises(ModelError):
        resolve_model_dir(str(bad_dir))

    # with config.ini -> returns str(dir)
    good_dir = tmp_path / "good_config"
    good_dir.mkdir()
    (good_dir / "config.ini").touch()
    assert resolve_model_dir(str(good_dir)) == str(good_dir)


def test_unknown_preset_like_name_is_path_error(tmp_path):
    # "DeepFilterNet4" (not in PRESETS) -> ModelError (missing dir)
    with pytest.raises(ModelError):
        resolve_model_dir("DeepFilterNet4")


def test_ensure_model_retries_then_succeeds(monkeypatch):
    # side_effect [SystemExit, SystemExit, "/ok"] -> "/ok", called 3 times
    enhance_mod = _get_enhance_module()

    call_count = 0

    def side_effect(name):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise SystemExit(1)
        return "/ok"

    monkeypatch.setattr(enhance_mod, "maybe_download_model", side_effect)
    result = ensure_model("DeepFilterNet", attempts=3)
    assert result == "/ok"


def test_ensure_model_exhausts_attempts(monkeypatch):
    # side_effect always SystemExit -> ModelError, called exactly `attempts` times
    enhance_mod = _get_enhance_module()

    monkeypatch.setattr(enhance_mod, "maybe_download_model", lambda name: (_ for _ in ()).throw(SystemExit(1)))

    with pytest.raises(ModelError):
        ensure_model("DeepFilterNet", attempts=3)