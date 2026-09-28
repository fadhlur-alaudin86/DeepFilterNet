"""Tests for the managed runtime dependency bootstrap."""

import os

from gui.core import dependency as dep


def test_current_env_ready_true_in_repo_venv():
    # in .venv, find_spec real -> True
    assert dep.current_env_ready() is True


def test_current_env_ready_false_when_torch_missing(monkeypatch):
    # patch find_spec -> None for torch -> False
    real_find_spec = dep.importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "torch":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(dep.importlib.util, "find_spec", fake_find_spec)
    assert dep.current_env_ready() is False


def test_detect_cuda_tag_parses_nvidia_smi(monkeypatch):
    # fake stdout "CUDA Version: 12.8" -> "cu128"; FileNotFoundError -> None

    class FakeCompleted:
        def __init__(self, stdout):
            self.stdout = stdout

    monkeypatch.setattr(
        dep.subprocess,
        "run",
        lambda *args, **kwargs: FakeCompleted("| NVIDIA-SMI 550.54.14 | CUDA Version: 12.8 |\n"),
    )
    assert dep.detect_cuda_tag() == "cu128"

    def raise_not_found(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi not found")

    monkeypatch.setattr(dep.subprocess, "run", raise_not_found)
    assert dep.detect_cuda_tag() is None

    # unparseable output -> None (CPU wheel, no --index-url)
    monkeypatch.setattr(dep.subprocess, "run", lambda *args, **kwargs: FakeCompleted("no version"))
    assert dep.detect_cuda_tag() is None


def test_ensure_runtime_installs_in_order(monkeypatch, tmp_path):
    # recorded pip calls: [torch --index-url .../cu128, ./DeepFilterNet, flet platformdirs];
    # progress called with phase labels "venv", "torch", "deps"
    created = []
    pip_calls = []
    progress_calls = []

    monkeypatch.setattr(dep, "runtime_dir", lambda: tmp_path)
    monkeypatch.setattr(
        dep.venv, "create", lambda path, with_pip=True: created.append((path, with_pip))
    )
    monkeypatch.setattr(dep, "detect_cuda_tag", lambda: "cu128")
    monkeypatch.setattr(
        dep.subprocess, "run", lambda args, **kwargs: pip_calls.append((list(args), kwargs))
    )

    info = dep.ensure_runtime(lambda phase, frac: progress_calls.append((phase, frac)))

    assert created == [(tmp_path / "venv", True)]
    assert [phase for phase, _ in progress_calls] == ["venv", "torch", "deps"]
    assert len(pip_calls) == 3

    torch_cmd = " ".join(pip_calls[0][0])
    assert "install torch" in torch_cmd
    assert "--index-url https://download.pytorch.org/whl/cu128" in torch_cmd

    assert "./DeepFilterNet" in pip_calls[1][0]

    deps_cmd = " ".join(pip_calls[2][0])
    assert "flet" in deps_cmd
    assert "platformdirs" in deps_cmd

    for _, kwargs in pip_calls:
        assert kwargs.get("check") is True
        assert kwargs.get("cwd") == dep.REPO_ROOT

    assert info == dep.RuntimeInfo(mode="venv", python=dep.venv_python(tmp_path / "venv"))


def test_ensure_runtime_cpu_fallback(monkeypatch, tmp_path):
    # detect_cuda_tag None -> pip install torch (default index), no --index-url arg
    pip_calls = []

    monkeypatch.setattr(dep, "runtime_dir", lambda: tmp_path)
    monkeypatch.setattr(dep.venv, "create", lambda path, with_pip=True: None)
    monkeypatch.setattr(dep, "detect_cuda_tag", lambda: None)
    monkeypatch.setattr(dep.subprocess, "run", lambda args, **kwargs: pip_calls.append(list(args)))

    dep.ensure_runtime(lambda *_: None)

    assert "install" in pip_calls[0] and "torch" in pip_calls[0]
    assert "--index-url" not in pip_calls[0]


def test_ensure_gui_process_noop_when_ready(monkeypatch):
    # current_env_ready True -> returns False, no subprocess spawned
    spawned = []
    runtime_calls = []

    monkeypatch.setattr(dep, "current_env_ready", lambda: True)
    monkeypatch.setattr(dep, "ensure_runtime", lambda progress: runtime_calls.append(progress))
    monkeypatch.setattr(
        dep.subprocess, "Popen", lambda *args, **kwargs: spawned.append((args, kwargs))
    )

    assert dep.ensure_gui_process() is False
    assert runtime_calls == []
    assert spawned == []


def test_ensure_gui_process_relaunches_when_not_ready(monkeypatch):
    # spawn call args contain [runtime python, "-m", "gui", "--runtime-child"]
    # and env PYTHONPATH includes repo root; returns True
    runtime_calls = []
    spawned = []

    monkeypatch.setattr(dep, "current_env_ready", lambda: False)
    monkeypatch.setattr(dep, "ensure_runtime", lambda progress: runtime_calls.append(progress))
    monkeypatch.setattr(
        dep.subprocess, "Popen", lambda args, **kwargs: spawned.append((list(args), kwargs))
    )

    assert dep.ensure_gui_process() is True
    assert len(runtime_calls) == 1

    args, kwargs = spawned[0]
    assert args[0] == str(dep.venv_python(dep.runtime_dir() / "venv"))
    assert args[1:] == ["-m", "gui", "--runtime-child"]
    assert str(dep.REPO_ROOT) in kwargs["env"]["PYTHONPATH"].split(os.pathsep)
