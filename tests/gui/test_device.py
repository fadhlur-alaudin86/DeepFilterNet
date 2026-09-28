import os

import pytest

from gui.core.device import apply_device, available_devices, resolve_device_env


def test_resolve_auto_returns_none():
    assert resolve_device_env("Auto") is None


def test_resolve_cpu_and_cuda():
    assert resolve_device_env("CPU") == "cpu"
    assert resolve_device_env("CUDA:2") == "cuda:2"


def test_resolve_invalid_raises():
    with pytest.raises(ValueError):
        resolve_device_env("cuda:99")
    with pytest.raises(ValueError):
        resolve_device_env("GPU")
    with pytest.raises(ValueError):
        resolve_device_env("")


def test_apply_device_auto_clears_stale_env(monkeypatch):
    monkeypatch.setenv("DEVICE", "cuda:0")
    apply_device("Auto")
    assert "DEVICE" not in os.environ


def test_apply_device_sets_env(monkeypatch):
    apply_device("CUDA:1")
    assert os.environ["DEVICE"] == "cuda:1"


def test_available_devices_without_torch(monkeypatch):
    import sys

    monkeypatch.setitem(
        sys.modules,
        "torch",
        type(
            "Mod",
            (),
            {
                "cuda": type(
                    "Mod",
                    (),
                    {
                        "is_available": lambda self: False,
                        "device_count": lambda self: 0,
                    },
                )()
            },
        ),
    )
    result = available_devices()
    assert result == ["Auto", "CPU"]
