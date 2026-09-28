# DeepFilterNet Desktop GUI — Design Spec

Date: 2026-09-28
Status: Approved (design sections 1–5 reviewed in conversation)
Path: Architectural (brainstorming skill)

## 1. Purpose and Success Criteria

Build a new desktop GUI for DeepFilterNet that exposes the project's
user-facing enhancement functions, inspired by UVR
(`ultimatevocalremovergui`) but built on a modern framework. The previous
PyQt6 GUI's source was lost (only `__pycache__` remains); its surviving
history lives on branch `gui-legacy` (tip `58b3a10`) and stash
`stash@{0}` (`f88a838`) as a design reference only — not code to import
wholesale.

Success criteria:

- A user can select audio files, pick a model and parameters, press
  Enhance, and receive enhanced output files with live progress —
  without touching a terminal.
- All inference-only functions of the Python `df` package are reachable
  from the UI (see §6 non-goals for deferred items).
- Runs on Windows and Linux as mandatory targets, macOS best-effort.
- Distributable both from source (`python -m gui`) and as a packaged
  binary (PyInstaller via `flet pack`, adapted from the existing
  `deepfilter-gui.spec`).

## 2. Decisions (grilling rounds Q1–Q15)

| # | Topic | Decision |
|---|-------|----------|
| Q1 | Framework | Flet (1.0.x, Apache-2.0), Material 3 theming |
| Q2 | Scope | Inference only for v1; training/eval/ONNX export deferred, architecture allows later addition |
| Q3 | Platforms | Windows + Linux mandatory, macOS best-effort |
| Q4 | Distribution | Both source-run and packaged binary |
| Q5 | Legacy GUI history | Restore dangling commits to reference branch `gui-legacy` (done: `git branch gui-legacy 58b3a10`); stash kept as-is; never run `git gc`/`git stash drop` |
| Q6 | UI language | English |
| Q7 | Backend integration | (a) In-process Python API (`init_df()`/`enhance()`) |
| Q8 | Processing model | In-memory chunking (Auto / preset / Custom) |
| Q9 | Parameter exposure | Standard panel + collapsible Advanced |
| Q10 | Output | Global output dir (default `./out`), format override, suffix default ON |
| Q11 | Packaged dependencies | Managed runtime venv, downloaded on first run (CUDA-aware wheel selection) |
| Q12 | Device UI | Dropdown `Auto` / `CPU` / `CUDA:n` in Standard options, hidden when no CUDA |
| Q13 | Onboarding | Combination: SetupDialog for runtime bootstrap; silent model download on first Enhance |
| Q14 | Model scope | Presets DF1/2/3 plus custom model path; no ONNX |
| Q15 | Settings persistence | JSON config file in platform user config dir |

## 3. Architecture

Selected approach: **thin backend interface, in-process implementation
for v1** (approach 3 of 3 considered).

- Approach 1: single process, worker thread + event bus — simplest, best
  pause/resume ergonomics; risk: native crash kills the app.
- Approach 2: UI process + inference subprocess — crash isolation, but
  IPC/serialization complexity and the same pain points as the legacy
  ffmpeg subprocess pipeline.
- Approach 3 (chosen): approach 1 behind an `EnhancementBackend` ABC so a
  `SubprocessBackend` can be added later without touching UI, queue, or
  chunker.

### 3.1 Package layout

```
gui/
  __init__.py
  main.py                  # entrypoint: bootstrap, setup dialog, run Flet app
  app.py                   # Flet App class: views, theme, wiring
  core/
    backend.py             # EnhancementBackend (ABC) + InProcessBackend
    jobs.py                # JobQueue: task queue, worker thread, state machine
    chunker.py             # in-memory chunking (Auto/preset/Custom) + concat
    config.py              # ConfigStore: load/save JSON in user config dir
    models.py              # model registry: DF1/2/3 presets + custom path
    device.py              # CUDA detection, Auto/CPU/CUDA:n resolution
    dependency.py          # managed runtime venv bootstrap/repair
    events.py              # event bus: worker -> UI via run_coroutine_threadsafe
  ui/
    pages/                 # main views (Enhance, Queue, Settings, Log)
    widgets/               # reusable components (file list, progress, console)
  resources/               # icons, splash
```

### 3.2 Backend interface

```python
class EnhancementBackend(ABC):
    def enhance_chunk(self, audio: np.ndarray, cfg: JobConfig) -> np.ndarray: ...
    def cancel(self) -> None: ...
    def shutdown(self) -> None: ...
```

`InProcessBackend` is the only v1 implementation.

### 3.3 Data flow

1. UI collects files (FilePicker) and parameters; user clicks Enhance.
2. `JobQueue` creates a `Job` with a **config snapshot** (like UVR's
   `QueueTask` — no live references to widgets) and enqueues it.
3. A single worker thread pulls jobs, splits audio in memory via
   `Chunker`, calls `enhance()` per chunk through `InProcessBackend`,
   and advances the job state machine:
   `queued -> running <-> paused -> done | failed | cancelled`.
4. Every progress tick, chunk completion, and log line is published
   through `events.py`, which hops to the Flet loop with
   `asyncio.run_coroutine_threadsafe(coro, page.loop)`; only the Flet
   loop mutates UI. **No direct `page.update()` from worker threads**
   (known race condition, flet issue #3571).
5. Pause takes effect at chunk boundaries (in-memory chunking makes this
   a natural sync point; no `psutil` process suspension). Cancel stops
   after the current chunk and discards buffered output.

### 3.4 Flet-specific constraints (verified 2026-09)

- Flet 1.0.x, Apache-2.0; Material 3, `color_scheme_seed`,
  `ThemeMode.SYSTEM` for dark mode; no QSS.
- No native OS file drag-and-drop (issue #5990); `FilePicker` is the
  primary input path. `flet-desktop-drop` evaluated post-v1.
- Long work runs via `page.run_thread` / `asyncio.to_thread`; UI updates
  only via the event bus described above.
- Packaging: `flet pack` (PyInstaller) chosen over `flet build`
  (serious_python) — adapts the existing `deepfilter-gui.spec`, faster
  boot; `flet build` remains a future option. Packaged startup is slow;
  the setup dialog/splash covers it.

## 4. UI Design

Single window, sidebar navigation via `NavigationRail` (design variant
2a, chosen over UVR-style horizontal tabs):

```
+------+--------------------------------------+
| rail | ENHANCE view                         |
| Enh. | File list (FilePicker add)           |
| Queue| Output dir + Enhance All             |
| Log  | Standard options (always visible)    |
| Set. | Advanced options (collapsible)       |
|      | Progress (per-file + global)         |
+------+--------------------------------------+
status bar: device | queue count | log snippet
```

- **Enhance** (default view): file list, output controls, Standard
  options, Advanced (collapsed), progress.
- **Queue**: job list with states, reorder, retry, per-job log excerpt
  (UVR pattern).
- **Log**: full console with level filter and copy/save.
- **Settings**: default output dir, model dir, runtime/dependency
  management, theme (System/Light/Dark), config file location.

### 4.1 Parameter mapping (Standard)

| UI control | `enhance()` / backend target | Default |
|---|---|---|
| Model preset or custom path | `model_base_dir` | DF3 |
| Atten limit slider (0–60 dB, 0 = off) | `atten_lim_db` | off (`None`, matches CLI default) |
| Postfilter toggle | `pf` | ON |
| Device dropdown Auto/CPU/CUDA:n | resolved at `init_df()` (see §5) | Auto |
| Output dir | output path | `./out` |
| Output format WAV/FLAC/MP3 | `save_audio` format | WAV (MP3 disabled when soundfile lacks an MP3 encoder) |
| Suffix toggle | `<stem>-deep-filtered` naming | ON |

The suffix toggle lives with the other output controls in the Standard
view (Q9 placed it in Advanced; Q10's output grouping supersedes it).

Advanced (collapsible): epoch, delay compensation
(`no_delay_compensation`), log level, `no_df_stage`, chunk mode/size.

## 5. Enhancement Pipeline

### 5.1 Device selection

The Python CLI has no device flag; `df.utils.get_device()` reads
`train.DEVICE` from config. `InProcessBackend` resolves the UI choice
before `init_df()` by overriding the df config device value (exact
mechanism — environment variable vs `df.config` write — confirmed
against `DeepFilterNet/df/config.py` during implementation). Device
list for the dropdown comes from `torch.cuda.device_count()` at
startup.

### 5.2 Model lifecycle

`init_df()` results are cached by `(model_base_dir, epoch)`; re-init
only when Model/Epoch changes or device changes. `model.reset_h0()` per
file (same as `enhance()` CLI behavior).

### 5.3 Chunking (in-memory; replaces legacy ffmpeg file pipeline)

- **Auto** (default): single chunk when duration ≤ threshold (default
  120 s); otherwise split at `ChunkSize` (default 60 s).
- **Preset**: 30 / 60 / 120 / 300 s. **Custom**: 5–600 s, validated.
- Split on sample boundaries: `audio[i*sr*size : (i+1)*sr*size]`, no
  overlap by default (per-chunk `enhance()` already resets state);
  concatenate results. Chunk-boundary artifacts are equivalent to the
  legacy GUI's file-splitting behavior — accepted for v1, documented
  here as a known limitation; an overlap mode is an Advanced-option
  candidate if audible.
- Decode via `df.io.load_audio` (soundfile) — **no ffmpeg dependency**.
  Unsupported formats surface the native soundfile error in the UI.

### 5.4 Output

Global output dir; naming `<stem>-deep-filtered.<ext>`; collisions get
a numeric suffix. Rust-only SNR flags (`--min-db-thresh`,
`--max-db-erb-thresh`, `--max-db-df-thresh`, `--reduce-mask`) are not
exposed by the Python API — recorded as a future `df` enhancement, out
of scope for v1.

## 6. Non-Goals (v1)

- Training, evaluation, ONNX export UI (future: new views/job types
  without changing the backend interface).
- SNR threshold flags (not in Python API).
- ONNX runtime backend, multi-GPU batching, per-file parameter
  overrides, plugin system.
- Native OS drag-and-drop (Flet limitation; post-v1 evaluation of
  `flet-desktop-drop`).

## 7. Dependency & Packaging Strategy

- **Managed runtime venv** (adapted from legacy `dependency_manager`):
  `~/.local/share/deepfilternet-gui/runtime/` on Linux (platformdirs
  equivalent on Windows).
  - Source-run: if `torch` + `soundfile` import cleanly in the active
    env, use it; otherwise bootstrap the managed venv.
  - Packaged: always the managed venv (torch never bundled; keeps the
    binary 50–180 MB).
  - Wheel selection: `nvidia-smi` -> CUDA tag -> matching torch
    index URL; CPU wheel fallback when no GPU.
- **Packaging**: `flet pack` with adapted `deepfilter-gui.spec`;
  legacy `.github/workflows/build-gui.yml` revived from `gui-legacy`
  and adjusted.
- **Onboarding (Q13c)**:
  1. Startup: ConfigStore + runtime check.
  2. Runtime missing/broken -> SetupDialog with progress bar
     (bootstrap venv, install wheels, Retry/Skip; Skip disables
     Enhance until repaired).
  3. Runtime OK -> main window immediately.
  4. Model download (`maybe_download_model`) silent on first Enhance
     with log progress + "Preparing model…" status indicator
     (~40 MB, not dialog-worthy).
  5. Static Flet splash masks packaged-app startup latency.

## 8. Error Handling

All failures flow through the event bus; the UI never crashes.

| Condition | Response |
|---|---|
| Runtime missing/partial | Toast + "Repair runtime" -> SetupDialog |
| GPU OOM during enhance | Job -> failed; log suggests smaller chunk or CPU; queue continues |
| Unsupported file format | That file -> failed with native soundfile message; others continue |
| Model download failure | 3 automatic retries, then toast; Enhance retryable |
| Cancel / pause | Deterministic at chunk boundaries; resume processes remaining chunks |
| Native/CUDA crash | The GUI process dies with it; diagnosis via `enhance.log` (persistent). An in-process crash cannot show a dialog — a restart/supervisor dialog is deferred together with `SubprocessBackend` (accepted limitation of approach 1; the reason the backend interface retains value) |

## 9. Configuration Persistence

- Location via `platformdirs`: Linux `~/.config/deepfilternet-gui/`,
  Windows `%APPDATA%\deepfilternet-gui\`.
- **Persisted**: all Enhance options (model, atten limit, pf, device,
  epoch, chunk mode/size, advanced flags), output dir, format, suffix
  toggle, theme, log level, window size/position, last view.
- **Session-only**: file list, queue contents, per-job state.
- **Robustness**: schema-validated load (unknown keys dropped, missing
  keys defaulted, `.bak` backup before rewrite), atomic write
  (tmp + rename). Save on window close + debounced 1 s after changes.

## 10. Logging

Standard `logging` with rotating file handler
(`platformdirs.user_log_dir`/`enhance.log`) plus a stream to the Log
view via the event bus. UI-selected level (INFO default, DEBUG for
troubleshooting). File list and chunk boundaries logged at DEBUG.

## 11. Testing Strategy

- **Unit (pytest, no Flet)**: chunker (boundaries, concat, roundtrip),
  job state machine (pause/cancel/retry), ConfigStore (schema, backward
  compat, atomic write), device resolver, backend argument mapping
  (mocked `enhance()`).
- **Smoke**: app boots headless, all four views render.
- **Pipeline**: real `enhance()` on a 1–2 s WAV fixture, CPU-only CI.

## 12. References

- DeepFilterNet inference API: `DeepFilterNet/df/enhance.py`
  (`init_df`, `enhance`, `run`, `setup_df_argument_parser`,
  `maybe_download_model`), I/O: `DeepFilterNet/df/io.py`.
- Legacy GUI (reference only): branch `gui-legacy`, commits `7d066a2`
  (PyQt6 GUI + CI workflow) and `58b3a10` (native dialogs); stash
  `stash@{0}` (`f88a838`).
- Existing packaging: `deepfilter-gui.spec` (PyInstaller, onedir).
- UVR reference project: `~/.clone/Github/ultimatevocalremovergui/`.
- Flet docs/issues: threading cookbook, issue #3571 (update race),
  issue #5990 (no native DnD), `flet pack` docs.
