"""Managed runtime bootstrap for the DeepFilterNet GUI.

Creates and maintains a dedicated virtualenv with the GUI runtime
dependencies (torch, DeepFilterNet, flet, platformdirs) under
``platformdirs.user_data_dir("deepfilternet-gui") / "runtime"``, and
re-launches the GUI process into that runtime when the current
environment is not ready.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
import venv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import platformdirs

# Repository root (this file lives at <root>/gui/core/dependency.py); source
# mode installs REPO_ROOT/"DeepFilterNet" and puts REPO_ROOT on the child
# process PYTHONPATH (frozen mode uses package_source()/gui_source_parent()).
REPO_ROOT = Path(__file__).resolve().parents[2]

# Phase labels reported to progress callbacks, in install order.
PHASE_VENV = "venv"
PHASE_TORCH = "torch"
PHASE_DEPS = "deps"


@dataclass(frozen=True)
class RuntimeInfo:
    """Description of the managed runtime produced by ``ensure_runtime``."""

    mode: str
    python: Path


def runtime_dir() -> Path:
    """Return the managed runtime root; the venv lives at ``runtime_dir() / "venv"``."""
    return Path(platformdirs.user_data_dir("deepfilternet-gui")) / "runtime"


def venv_python(venv_dir: Path) -> Path:
    """Return the Python interpreter path inside *venv_dir* (platform-aware)."""
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def is_frozen() -> bool:
    """True inside a frozen bundle (PyInstaller ``sys.frozen``/``sys._MEIPASS``)."""
    return bool(getattr(sys, "frozen", False)) or hasattr(sys, "_MEIPASS")


def resolve_python() -> Path:
    """Return the interpreter used to create the managed venv.

    Source runs use the current interpreter (``sys.executable``). A frozen
    process cannot: ``sys.executable`` is the GUI binary itself, and running
    ``venv.create`` on it would re-launch the app through ``ensurepip``. So
    frozen runs resolve a real interpreter from ``PATH`` (``python3`` first,
    then ``python``) and raise RuntimeError naming Python 3.11 when neither
    is found.
    """
    if not is_frozen():
        return Path(sys.executable)
    for name in ("python3", "python"):
        found = shutil.which(name)
        if found:
            return Path(found)
    raise RuntimeError(
        "No Python interpreter found on PATH. Install Python 3.11 or newer "
        "and ensure it is on PATH so the app can create its managed runtime."
    )


def _bundle_data_root() -> Path:
    """Root of the frozen bundle's data directory (PyInstaller ``_MEIPASS``)."""
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass)
    return Path(sys.executable).parent


def package_source() -> Path:
    """Directory holding the DeepFilterNet (``df``) sources for the pip install.

    Frozen bundles ship the ``DeepFilterNet/`` tree as data (see
    ``deepfilter-gui.spec``); source runs install the checkout at
    ``REPO_ROOT/"DeepFilterNet"``.
    """
    if not is_frozen():
        return REPO_ROOT / "DeepFilterNet"
    return _bundle_data_root() / "DeepFilterNet"


def gui_source_parent() -> Path:
    """Directory containing the ``gui`` package, for the child PYTHONPATH.

    Frozen bundles ship the ``gui/`` tree as data next to ``DeepFilterNet/``;
    source runs use the repository root.
    """
    if not is_frozen():
        return REPO_ROOT
    return _bundle_data_root()


def current_env_ready() -> bool:
    """True when torch, soundfile, flet and df are all importable in this process."""
    return all(
        importlib.util.find_spec(name) is not None for name in ("torch", "soundfile", "flet", "df")
    )


def detect_cuda_tag() -> str | None:
    """Return a ``"cu128"``-style wheel index tag read from ``nvidia-smi``, or None.

    Parses the ``CUDA Version: X.Y`` header line of ``nvidia-smi`` output, e.g.
    ``12.8`` becomes ``"cu128"``. Returns None when nvidia-smi is absent
    (no NVIDIA driver, FileNotFoundError) or the output is unparseable, which
    selects the CPU wheel (no ``--index-url``).
    """
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
    except OSError:
        return None
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", result.stdout or "")
    if match is None:
        return None
    major, minor = match.group(1), match.group(2)
    return f"cu{major}{minor}"


def _pip_install(python: Path, *specs: str) -> None:
    """Run ``python -m pip install *specs`` (cwd = repo root for local paths).

    Raises RuntimeError with the pip stderr tail on failure so the setup
    dialog can display the underlying cause.
    """
    args = [str(python), "-m", "pip", "install", *specs]
    try:
        subprocess.run(args, check=True, capture_output=True, text=True, cwd=REPO_ROOT)
    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or "").strip()[-2000:]
        raise RuntimeError(f"pip install failed ({' '.join(specs)}): {tail}") from exc
    except OSError as exc:
        raise RuntimeError(f"pip install failed ({' '.join(specs)}): {exc}") from exc


def _create_venv_frozen(venv_dir: Path) -> None:
    """Create the venv with an external interpreter (frozen processes only).

    ``venv.create`` cannot be used frozen: it runs ``sys.executable`` (the GUI
    binary itself) through ``ensurepip``. Failures raise RuntimeError with the
    stderr tail, matching the ``_pip_install`` contract the setup dialog shows.
    """
    cmd = [str(resolve_python()), "-m", "venv", str(venv_dir)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or "").strip()[-2000:]
        raise RuntimeError(f"venv creation failed: {tail}") from exc
    except OSError as exc:
        raise RuntimeError(f"venv creation failed: {exc}") from exc


def ensure_runtime(progress: Callable[[str, float], None]) -> RuntimeInfo:
    """Create the managed venv if needed and install the GUI runtime stack.

    Install order, reported to *progress* as phase labels: ``"venv"``
    (create venv with pip; source runs use stdlib ``venv.create``, frozen
    runs delegate to ``resolve_python() -m venv``), ``"torch"`` (CUDA-aware
    wheel index when ``detect_cuda_tag()`` finds a driver, CPU wheel
    otherwise), ``"deps"`` (the ``package_source()`` DeepFilterNet tree,
    then flet + platformdirs). ``requirements-gui-runtime.txt`` is the
    human-readable manifest of the dependency set. pip always runs under
    the managed venv's own python.
    """
    venv_dir = runtime_dir() / "venv"
    progress(PHASE_VENV, 0.1)
    if not venv_dir.exists():
        if is_frozen():
            _create_venv_frozen(venv_dir)
        else:
            venv.create(venv_dir, with_pip=True)
    python = venv_python(venv_dir)

    progress(PHASE_TORCH, 0.3)
    torch_specs = ["torch"]
    tag = detect_cuda_tag()
    if tag is not None:
        torch_specs += ["--index-url", f"https://download.pytorch.org/whl/{tag}"]
    _pip_install(python, *torch_specs)

    progress(PHASE_DEPS, 0.7)
    _pip_install(python, str(package_source()))
    _pip_install(python, "flet>=1.0,<2.0", "platformdirs")
    return RuntimeInfo(mode="venv", python=python)


def ensure_gui_process(selftest: bool = False) -> bool:
    """Decide where the GUI process should run.

    Returns False to continue in-process (current environment already
    ready); otherwise ensures the managed runtime, spawns the relaunched
    child ``<runtime python> -m gui --runtime-child`` with PYTHONPATH set
    to ``gui_source_parent()`` (the repository root in source mode, the
    bundled data directory when frozen), and returns True (caller must exit).
    *selftest* forwards ``--selftest`` to the child so a headless check
    stays headless after the relaunch instead of opening the GUI.
    """
    if current_env_ready():
        return False
    ensure_runtime(progress=lambda *_: None)
    python = venv_python(runtime_dir() / "venv")
    env = {**os.environ, "PYTHONPATH": str(gui_source_parent())}
    args = [str(python), "-m", "gui", "--runtime-child"]
    if selftest:
        args.append("--selftest")
    subprocess.Popen(args, env=env)
    return True
