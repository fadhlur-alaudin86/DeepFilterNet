"""Frozen-packaging robustness tests (Task 15 follow-up, R16/R17).

Covers the interpreter-resolution seam used by the frozen bootstrap, the
bundle data-source seams, the torch-less save-format probe, and the
venv-creation split in ``ensure_runtime``. No test creates a real venv or
subprocess: ``venv.create`` and ``subprocess.run`` are always patched.
"""

import sys
from pathlib import Path

import pytest

from gui.core import dependency as dep
from gui.core import jobs


def _clear_frozen(monkeypatch) -> None:
    """Drop both frozen markers so the code under test runs in source mode."""
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)


def test_probe_returns_empty_when_df_unimportable(monkeypatch):
    # block "df" (sys.modules["df"] = None, cached df.* submodules purged)
    # -> supported_formats() == [] (no raise), cache reset around the call
    for name in [n for n in sys.modules if n.startswith("df.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "df", None)
    monkeypatch.setattr(jobs, "_FORMATS_CACHE", None)
    assert jobs.supported_formats() == []


def test_resolve_python_prefers_realtime_when_not_frozen(monkeypatch):
    # sys.frozen absent -> Path(sys.executable); shutil.which never consulted
    _clear_frozen(monkeypatch)
    consulted = []
    monkeypatch.setattr(dep.shutil, "which", lambda name: consulted.append(name))
    assert dep.resolve_python() == Path(sys.executable)
    assert consulted == []


def test_resolve_python_frozen_finds_python3(monkeypatch):
    # frozen True, which("python3") -> Path; which("python") must NOT be
    # consulted (python3 wins)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    consulted = []

    def fake_which(name):
        consulted.append(name)
        if name == "python3":
            return "/usr/bin/python3"
        return "/usr/bin/python"

    monkeypatch.setattr(dep.shutil, "which", fake_which)
    assert dep.resolve_python() == Path("/usr/bin/python3")
    assert "python" not in consulted


def test_resolve_python_frozen_falls_back_to_python(monkeypatch):
    # python3 None, python -> Path
    monkeypatch.setattr(sys, "frozen", True, raising=False)

    def fake_which(name):
        return None if name == "python3" else "/usr/bin/python"

    monkeypatch.setattr(dep.shutil, "which", fake_which)
    assert dep.resolve_python() == Path("/usr/bin/python")


def test_resolve_python_frozen_none_raises(monkeypatch):
    # both None -> RuntimeError naming Python 3.11
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(dep.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="Python 3.11"):
        dep.resolve_python()


def test_package_source_frozen_uses_bundle_data(monkeypatch):
    # _MEIPASS set -> _MEIPASS/"DeepFilterNet"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", "/opt/frozen/_internal", raising=False)
    assert dep.package_source() == Path("/opt/frozen/_internal") / "DeepFilterNet"


def test_package_source_source_mode_uses_repo(monkeypatch):
    # no frozen flags -> REPO_ROOT/"DeepFilterNet"
    _clear_frozen(monkeypatch)
    assert dep.package_source() == dep.REPO_ROOT / "DeepFilterNet"


def test_ensure_runtime_frozen_venv_via_subprocess(monkeypatch, tmp_path):
    # frozen: venv via [resolved, "-m", "venv", venv_dir] subprocess.run;
    # stdlib venv.create never called; pip always under the venv python
    frozen_python = tmp_path / "python3"
    venv_calls = []
    pip_calls = []
    stdlib_calls = []

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(dep, "runtime_dir", lambda: tmp_path)
    monkeypatch.setattr(dep, "resolve_python", lambda: frozen_python)
    monkeypatch.setattr(dep, "detect_cuda_tag", lambda: None)
    monkeypatch.setattr(dep.venv, "create", lambda *a, **k: stdlib_calls.append(a))

    def fake_run(args, **kwargs):
        args = list(args)
        if "venv" in args:
            venv_calls.append(args)
        else:
            pip_calls.append(args)

    monkeypatch.setattr(dep.subprocess, "run", fake_run)

    dep.ensure_runtime(lambda *_: None)

    assert venv_calls == [[str(frozen_python), "-m", "venv", str(tmp_path / "venv")]]
    assert stdlib_calls == []
    assert len(pip_calls) == 3
    venv_python = str(dep.venv_python(tmp_path / "venv"))
    for args in pip_calls:
        assert args[0] == venv_python


def test_ensure_runtime_source_keeps_stdlib_venv_create(monkeypatch, tmp_path):
    # source mode: stdlib venv.create kept; subprocess.run only for pip,
    # and the df install target is package_source()
    _clear_frozen(monkeypatch)
    created = []
    pip_calls = []

    monkeypatch.setattr(dep, "runtime_dir", lambda: tmp_path)
    monkeypatch.setattr(
        dep.venv, "create", lambda path, with_pip=True: created.append((path, with_pip))
    )
    monkeypatch.setattr(dep, "detect_cuda_tag", lambda: None)
    monkeypatch.setattr(dep.subprocess, "run", lambda args, **kwargs: pip_calls.append(list(args)))

    dep.ensure_runtime(lambda *_: None)

    assert created == [(tmp_path / "venv", True)]
    assert len(pip_calls) == 3
    assert str(dep.package_source()) in pip_calls[1]
    venv_python = str(dep.venv_python(tmp_path / "venv"))
    for args in pip_calls:
        assert args[0] == venv_python
