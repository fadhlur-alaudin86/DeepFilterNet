# DeepFilterNet GUI - Setup Guide

This guide covers running the Flet-based DeepFilterNet desktop GUI from
source and producing a packaged (PyInstaller) executable.

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10 - 3.12 (3.11 recommended) | Used for both source runs and packaging |
| pip | >= 23 | `python -m pip install --upgrade pip` |
| Git | any | `requirements.txt` installs DeepFilterLib/DeepFilterNet from Git checkouts |
| Rust toolchain (rustc + cargo) | >= 1.70 | Needed to build the `pyDF` (DeepFilterLib) Python extension |
| PyInstaller | >= 6 | Required by `flet pack` (see below) |
| ffmpeg | >= 4 | Optional; only used by the `deepFilter` CLI |

Operating systems: Linux and Windows are primary targets; macOS is
best-effort (see Platform Notes).

---

## Running from Source

```bash
git clone https://github.com/SuperBypassUdinnn/DeepFilterNet.git
cd DeepFilterNet

python -m venv .venv
source .venv/bin/activate        # Linux/macOS
```

Linux/macOS:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt -r requirements-gui.txt

python -m gui
```

Windows: `requirements.txt` is a Linux-only CUDA freeze (its
`nvidia-nccl-cu13`, `nvidia-nvshmem-cu13` and `triton` pins have no
`win_amd64` wheels), so install the CPU-only stack used by the Windows CI
leg instead:

```powershell
.venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-gui.txt
python -m pip install soundfile loguru deepfilterlib pytest

python -m gui
```

Sanity check (builds every view against a temporary config without opening
a window; requires a complete runtime environment):

```bash
python -m gui --selftest         # prints "selftest ok", exits 0
```

Note for working on the local `DeepFilterNet/` checkout instead of the
pinned Git install pulled in by `requirements.txt`:

```bash
# Linux/macOS
PYTHONPATH="$PWD/DeepFilterNet" python -m gui
# Windows PowerShell
$env:PYTHONPATH="$PWD\DeepFilterNet"; python -m gui
```

---

## First-Run Behavior

On startup the GUI checks whether the current process can import `torch`,
`soundfile`, `flet` and `df`:

- **Complete environment** (e.g. an activated `.venv` with
  `requirements.txt` installed): the main window opens immediately.
- **Incomplete environment**: an interactive setup dialog offers to create
  the managed runtime. The app then:
  1. Creates a virtualenv at `<user data dir>/deepfilternet-gui/runtime/venv`
     (`~/.local/share/deepfilternet-gui/runtime/venv` on Linux,
     `%LOCALAPPDATA%\deepfilternet-gui\runtime\venv` on Windows).
  2. Installs `torch` (CUDA wheel index when `nvidia-smi` reports a CUDA
     version, CPU wheel otherwise), the local `./DeepFilterNet` package,
     then `flet` and `platformdirs` (see
     `requirements-gui-runtime.txt` for the manifest).
  3. Relaunches the GUI process inside that managed runtime.
- **First enhancement** downloads the DeepFilterNet model weights into the
  model cache; this is the only other step that needs the network.
- Choosing **Skip** (or a failed setup) continues in degraded mode: the
  window opens with Enhance disabled and a banner until the runtime is
  repaired from Settings.

---

## Building a Packaged Executable

### Prerequisites

Besides `requirements-gui.txt`, PyInstaller must be installed in the
environment used for packaging:

```bash
pip install -r requirements-gui.txt
pip install pyinstaller
```

`flet pack` requires PyInstaller but does not install it itself; without it
the command fails with:

```text
Please install PyInstaller module to use flet pack command: No module named 'PyInstaller'
```

### Primary build path (verified on Linux)

```bash
flet pack gui/main.py --name DeepFilterNet-GUI \
  --pyinstaller-build-args=--exclude-module=torch \
  --pyinstaller-build-args=--exclude-module=tensorflow
```

Output on Linux: `dist/DeepFilterNet-GUI` (single-file executable, about
66 MB) plus `dist/DeepFilterNet-GUI.desktop`.

`torch` and `tensorflow` must never be bundled (the managed runtime
provides them later). Without the excludes the bundle grows to about 2.6 GB
because deferred `import torch` statements inside `gui/` are still visible
to static analysis. Verify the exclusion after a build:

```bash
grep -c "('torch'," build/DeepFilterNet-GUI/Analysis-00.toc   # must print 0
grep "excluded module named torch" build/DeepFilterNet-GUI/warn-DeepFilterNet-GUI.txt
```

### Fallback build path (direct PyInstaller, one-folder)

```bash
pyinstaller deepfilter-gui.spec --noconfirm --clean
```

Output: `dist/DeepFilterNet-GUI/` containing the `DeepFilterNet-GUI`
executable. The spec entry point is `gui/main.py`; the same
`torch`/`tensorflow` excludes are baked in, and `gui`/`flet` submodules are
collected as hidden imports.

### Verification

- Source environment: `python -m gui --selftest` must print `selftest ok`
  and exit 0.
- Packaged binary: `./dist/DeepFilterNet-GUI --selftest` currently exits
  non-zero; see Known Limitations.

---

## Known Limitations

- **CI Windows leg uses a CPU-only dependency set.** `requirements.txt` is a
  pip freeze from a Linux machine: its `nvidia-nccl-cu13`, `nvidia-nvshmem-cu13`
  and `triton` pins have no `win_amd64` wheels, so `pip install -r
  requirements.txt` on `windows-latest` fails during dependency resolution.
  The Windows job therefore installs `torch`/`torchaudio` from the CPU wheel
  index (`https://download.pytorch.org/whl/cpu`) together with `soundfile`,
  `loguru`, `deepfilterlib`, `pytest` and `pyinstaller` (plus
  `requirements-gui.txt`, whose `flet` and `platformdirs` pins are
  Windows-safe), and runs the tests with `PYTHONPATH=DeepFilterNet`. The
  Linux job keeps installing `requirements.txt` unchanged; both legs run the
  same test command and the same `flet pack` build.
- **Windows source runs need `PYTHONPATH`.** Unlike the CI leg (which sets
  `PYTHONPATH=DeepFilterNet` in the workflow), the Windows install sequence
  above does not pip-install the `DeepFilterNet` package, so use the
  `PYTHONPATH` note under Running from Source or the app falls back to the
  first-run managed runtime instead of the `.venv` you just created.
- **Packaged `--selftest` fails (exit 1).** Two independent causes, both
  outside the packaging configuration:
  1. With `torch` excluded, startup treats the environment as incomplete
     and enters the first-run runtime bootstrap, which cannot create a
     virtualenv inside a frozen process (`sys.executable` is the packaged
     binary itself, so the `ensurepip` step fails with exit status 2).
  2. Skipping the bootstrap (`--runtime-child --selftest`) fails later:
     `EnhanceView` probes `df.io.save_audio` while building, and the
     bundled `df` import requires `torch`, which is deliberately not
     bundled.
  Use the source selftest to verify a packaging environment.

---

## CI Workflow

`.github/workflows/build-gui.yml` runs on version tags (`v*`) and on
manual dispatch. It builds on `ubuntu-latest` and `windows-latest` with
Python 3.11, runs `python -m pytest tests/gui -m "not e2e"` (CPU only, no
model download) on both legs, packages with the `flet pack` command above
and uploads `dist/` as a build artifact. Dependency installs are OS-gated
because the two legs use different torch stacks:

- **Linux:** `pip install -r requirements.txt -r requirements-gui.txt`
  (plus `pytest`, `pyinstaller`) - the pinned environment used for local
  development, installed verbatim.
- **Windows:** `requirements.txt` cannot be installed there (it is a Linux
  pip freeze whose `nvidia-nccl-cu13`, `nvidia-nvshmem-cu13` and `triton`
  pins have no `win_amd64` wheels), so the leg installs a CPU-only stack
  instead: `torch`/`torchaudio` from `https://download.pytorch.org/whl/cpu`,
  `-r requirements-gui.txt`, plus `soundfile`, `loguru`, `deepfilterlib`
  (provides the `libdf` extension), `pytest` and `pyinstaller`. Its test
  step sets `PYTHONPATH=DeepFilterNet` so tests run against the checked-out
  `df` sources, since `DeepFilterNet` itself is not pip-installed on that
  leg.

---

## Platform Notes

- **Linux (primary, verified):** `flet pack` produces a single-file
  executable and a `.desktop` entry. Runtime system libraries:
  `libglib2.0-0`, `libgl1`; `ffmpeg` is optional and only needed by the
  CLI.
- **Windows (primary):** source runs use the CPU-only install sequence
  above (do not install `requirements.txt` there); the packaging commands
  are otherwise identical, and the executable is built as
  `dist/DeepFilterNet-GUI.exe` (pass `-D` for a one-folder bundle). The
  managed runtime uses `venv\Scripts\python.exe`, and the packaged app
  runs without a console window. Building `pyDF` from source (instead of
  the prebuilt `deepfilterlib` wheel) requires the Visual Studio Build
  Tools C++ workload (present on GitHub-hosted runners).
- **macOS (best-effort):** `flet pack` can produce an `.app` bundle, but
  macOS packaging is not verified by this repository's CI and no release
  artifacts are published for it.
