# -*- mode: python ; coding: utf-8 -*-
"""
DeepFilterNet GUI - PyInstaller spec file (fallback builder).

Primary build command (preferred, see SETUP.md):
    flet pack gui/main.py --name DeepFilterNet-GUI --pyinstaller-build-args=--exclude-module=torch --pyinstaller-build-args=--exclude-module=tensorflow

Fallback build (direct PyInstaller, onedir):
    pyinstaller deepfilter-gui.spec --noconfirm --clean

The output goes to dist/DeepFilterNet-GUI/ (one-folder bundle).
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

REPO_ROOT = Path(".")

# Data files: the Flet GUI has no static assets of its own (gui/resources is
# empty); flet's own PyInstaller hooks ship the runtime assets.
datas = []

# Hidden imports: gui submodules are reached through deferred imports inside
# view builders and gui/main.py, flet is imported lazily at startup.
hiddenimports = (
    collect_submodules("gui") + collect_submodules("flet") + ["platformdirs", "loguru"]
)

binaries = []

a = Analysis(
    ["gui/main.py"],
    pathex=[str(REPO_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # torch must NEVER be bundled (design spec section 7): the packaged
        # binary delegates heavy dependencies to the managed runtime venv.
        "torch",
        "tensorflow",
        # Test/tooling packages are never needed at runtime.
        "pytest",
        "unittest",
        "doctest",
        "tkinter",
        "matplotlib",
        "PIL",
        "IPython",
        "notebook",
    ],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="DeepFilterNet-GUI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # No terminal window on Windows
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="DeepFilterNet-GUI",
)
