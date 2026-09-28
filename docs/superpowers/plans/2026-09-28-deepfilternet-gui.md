# DeepFilterNet Desktop GUI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Flet-based desktop GUI that exposes DeepFilterNet's inference functions (file list -> chunked in-process enhancement -> output files) with queue, progress, logging, and first-run dependency bootstrap.

**Architecture:** Single-process Flet app; a `JobQueue` worker thread processes config-snapshot jobs through a thin `EnhancementBackend` interface (v1: `InProcessBackend` calling `df.enhance.init_df`/`enhance`). All UI updates flow through an `EventBus` bound to the Flet event loop. Dependencies (torch etc.) live either in the active env or in a managed venv created on first run; torch is never bundled.

**Tech Stack:** Python 3.11, Flet >=1.0,<2.0, torch/torchaudio (via `df` package), numpy, platformdirs, pytest, PyInstaller/`flet pack`.

**Spec:** `docs/superpowers/specs/2026-09-28-deepfilternet-gui-design.md` — this plan argues from it; executors read both.

## Global Constraints

- UI strings, code comments, docs, and commit messages: English only. No emojis or decorative icons anywhere (files or commits).
- Commit messages: Conventional Commits, short plain text (e.g. `feat(gui): add config store`), no Markdown syntax.
- Formatting: `black` line-length 100, `isort` profile black (repo `pyproject.toml`); must pass `black --check gui tests` and `flake8 gui tests`.
- Python 3.11 in existing `.venv`; Flet `>=1.0,<2.0` (Apache-2.0).
- torch is NEVER bundled into the binary; packaged mode uses a managed venv (spec §7).
- Platforms: Windows + Linux mandatory, macOS best-effort.
- No ffmpeg; all audio I/O through `df.io.load_audio` / `df.io.save_audio` / `df.io.resample`.
- Do NOT modify `DeepFilterNet/df/` public API. Device selection = env var `DEVICE` (`config.py` env override checked before `.ini`; `get_device()` reads `train.DEVICE`): `None`/unset = auto (cuda:0 if available), `"cpu"`, `"cuda:n"`.
- Never call `page.update()` from worker threads; all UI mutation goes through `EventBus.publish` -> `asyncio.run_coroutine_threadsafe(coro, page.loop)` (flet issue #3571).
- Do not touch branch `gui-legacy` or `stash@{0}` (legacy reference only).
- Test command from repo root: `.venv/bin/python -m pytest tests/gui -v`.
- Heavy imports (`torch`, `df.*`) are lazy (inside functions) in `gui/core` modules so the UI can start degraded.

## Review Focus

Failure modes from the spec that no single task covers on its own; each line's pinning test lives in the owning task.

1. **Device selection silently ignored** — `DEVICE` must be set before `init_df()` runs, and `Auto` must clear a stale user-set `DEVICE` — pinned in Task 2 (`test_apply_device_auto_clears_stale_env`) and Task 5 (`test_device_applied_before_init_df`).
2. **Concat length mismatch after chunking** — `enhance(pad=True)` returns the input length; a remainder chunk must not change total length — pinned in Task 4 (`test_concat_chunks_roundtrip_length`) and Task 7 (`test_full_file_output_length_matches_input`).
3. **Pause/cancel racing the worker** — no double-processed chunks, resume continues at the next chunk, cancel discards partial file output — pinned in Task 7 (`test_pause_resumes_at_next_chunk`, `test_cancel_discards_partial_output`).
4. **Config loss on crash or corrupt file** — atomic write, `.bak` backup, defaults on parse failure — pinned in Task 1 (`test_corrupt_config_backed_up_and_defaults_returned`, `test_save_leaves_valid_json`).
5. **Unsupported output format discovered only at save time** (after enhancement ran) — format support must be probed before a job runs — pinned in Task 7 (`test_unsupported_format_rejected_at_submit`).

---

### Task 1: Scaffolding + ConfigStore

**Files:**
- Create: `gui/__init__.py`, `gui/core/__init__.py`, `gui/core/config.py`, `requirements-gui.txt`
- Test: `tests/gui/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `ConfigStore(path: Path | None = None)`, methods `load() -> dict`, `get(key: str) -> Any`, `set(key: str, value: Any) -> None`, `save() -> None`, `path` property; module-level `DEFAULTS: dict[str, Any]`. Later tasks import `ConfigStore` for persistence and read defaults keys: `model, epoch, post_filter, atten_lim_db, device, no_df_stage, delay_compensation, chunk_mode, chunk_size_s, output_dir, output_format, suffix_enabled, log_level, theme_mode, window_width, window_height, last_view`.

- [ ] **Step 1: Install GUI dev dependencies**

Run: `.venv/bin/pip install "flet>=1.0,<2.0" platformdirs pytest black isort flake8` and add `flet>=1.0,<2.0` / `platformdirs` to `requirements-gui.txt` (one per line).
Expected: all import (`.venv/bin/python -c "import flet, platformdirs, pytest"`).

- [ ] **Step 2: Write the failing tests**

`tests/gui/test_config.py` — key cases:

```python
def test_roundtrip_preserves_values(tmp_path):        # set 3 keys, save, new store load, values equal
def test_unknown_keys_dropped_on_load(tmp_path)       # file with {"bogus": 1} -> load() has no "bogus"
def test_missing_keys_defaulted_on_load(tmp_path)     # file {} -> load() == DEFAULTS subset
def test_corrupt_config_backed_up_and_defaults_returned(tmp_path)  # file "not json" -> load()==defaults, path.with_suffix(".bak") exists with original bytes
def test_save_leaves_valid_json(tmp_path)             # after set+save, json.loads(path.read_text()) works
def test_default_path_under_platformdirs()            # ConfigStore().path name == "config.json", parent endswith "deepfilternet-gui"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/gui/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gui.core.config'`.

- [ ] **Step 4: Implement `gui/core/config.py`**

`DEFAULTS` dict with the 17 keys above (values per spec: `model="DeepFilterNet3"`, `epoch="best"`, `post_filter=True`, `atten_lim_db=None`, `device="Auto"`, `no_df_stage=False`, `delay_compensation=True`, `chunk_mode="auto"`, `chunk_size_s=60`, `output_dir="./out"`, `output_format="wav"`, `suffix_enabled=True`, `log_level="INFO"`, `theme_mode="SYSTEM"`, `window_width=1100`, `window_height=760`, `last_view="enhance"`). Default path via `platformdirs.user_config_dir("deepfilternet-gui")`. `load()`: parse JSON; on `JSONDecodeError` copy bytes to `.bak` and return defaults; drop unknown keys; fill missing. `save()`: write to `path.with_suffix(".tmp")` then `os.replace`. Keep black-formatted.

- [ ] **Step 5: Run tests, format, commit**

Run: `.venv/bin/python -m pytest tests/gui/test_config.py -v` then `black gui tests && isort gui tests && flake8 gui tests`
Expected: PASS, lint clean.

```bash
git add requirements-gui.txt tests/gui/test_config.py gui/__init__.py gui/core/__init__.py gui/core/config.py
git commit -m "feat(gui): add config store with atomic persistence"
```

---

### Task 2: Device Resolver

**Files:**
- Create: `gui/core/device.py`
- Test: `tests/gui/test_device.py`

**Interfaces:**
- Consumes: env-override behavior of `df.config` (env `DEVICE` beats `.ini`).
- Produces: `available_devices() -> list[str]` (first entry `"Auto"`), `resolve_device_env(choice: str) -> str | None` (pure), `apply_device(choice: str) -> None` (raises `ValueError` on invalid). Values: `"Auto"->None`, `"CPU"->"cpu"`, `"CUDA:n"->"cuda:n"`.

- [ ] **Step 1: Write the failing tests**

```python
def test_resolve_auto_returns_none():            assert resolve_device_env("Auto") is None
def test_resolve_cpu_and_cuda():                 assert resolve_device_env("CPU") == "cpu"; resolve_device_env("CUDA:2") == "cuda:2"
def test_resolve_invalid_raises():               pytest.raises(ValueError) for "cuda:99", "GPU", ""
def test_apply_device_auto_clears_stale_env(monkeypatch):  monkeypatch.setenv("DEVICE", "cuda:0"); apply_device("Auto"); "DEVICE" not in os.environ
def test_apply_device_sets_env(monkeypatch):     apply_device("CUDA:1"); os.environ["DEVICE"] == "cuda:1"
def test_available_devices_without_torch(monkeypatch):  # patch torch.cuda.is_available->False (and import failure): == ["Auto", "CPU"]
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/gui/test_device.py -v` — expected FAIL (module missing).

- [ ] **Step 3: Implement `gui/core/device.py`**

`resolve_device_env` validates with a regex `^(Auto|CPU|CUDA:\d+)$` (case-sensitive per spec strings) and maps to the env value. `available_devices()` lazily imports torch inside `try/except ImportError`; on success and `torch.cuda.is_available()`, append `CUDA:0..torch.cuda.device_count()-1`.

- [ ] **Step 4: Run tests + lint + commit**

Run: tests + `black/isort/flake8` — expected PASS/clean.

```bash
git add gui/core/device.py tests/gui/test_device.py
git commit -m "feat(gui): add device resolver with DEVICE env override"
```

---

### Task 3: Model Registry

**Files:**
- Create: `gui/core/models.py`
- Test: `tests/gui/test_models.py`

**Interfaces:**
- Consumes: `df.enhance.maybe_download_model(name: str) -> str` (extracts model zip; may `exit(1)` on network failure — treat `SystemExit` as failure).
- Produces: `PRESETS: tuple[str, ...] = ("DeepFilterNet", "DeepFilterNet2", "DeepFilterNet3")`; `class ModelError(Exception)`; `resolve_model_dir(model: str) -> str` (preset name -> download; otherwise validate directory contains `config.ini`); `ensure_model(model: str, attempts: int = 3) -> str` (retry wrapper, re-raises `ModelError` after `attempts`); `model_choices() -> list[str]`.

- [ ] **Step 1: Write the failing tests**

```python
def test_preset_delegates_to_download(monkeypatch):   # patch df.enhance.maybe_download_model -> "/x"; resolve_model_dir("DeepFilterNet3") == "/x", called once
def test_custom_path_requires_config_ini(tmp_path):   # dir without config.ini -> ModelError; with config.ini -> returns str(dir)
def test_unknown_preset_like_name_is_path_error(tmp_path):  # "DeepFilterNet4" (not in PRESETS) -> ModelError (missing dir)
def test_ensure_model_retries_then_succeeds(monkeypatch):   # side_effect [SystemExit, SystemExit, "/ok"] -> "/ok", called 3 times
def test_ensure_model_exhausts_attempts(monkeypatch): # side_effect always SystemExit -> ModelError, called exactly `attempts` times
```

- [ ] **Step 2: Run to verify failure** — same pattern as Task 1.

- [ ] **Step 3: Implement `gui/core/models.py`**

Lazy-import `maybe_download_model` inside `resolve_model_dir` (global constraints: no eager `df` import). `ensure_model` catches `(ModelError, SystemExit, OSError)` per attempt with a 1s/2s backoff via `time.sleep`, then re-raises `ModelError` chained from the last error. `model_choices()` returns `list(PRESETS) + ["Custom…"]` (ASCII ellipsis policy: use `"Custom path"` instead to avoid non-ASCII — name it exactly `"Custom path"`).

- [ ] **Step 4: Run tests + lint + commit**

```bash
git add gui/core/models.py tests/gui/test_models.py
git commit -m "feat(gui): add model registry with retrying download"
```

---

### Task 4: Chunker

**Files:**
- Create: `gui/core/chunker.py`
- Test: `tests/gui/test_chunker.py`

**Interfaces:**
- Consumes: nothing (pure numpy/stdlib).
- Produces: `AUTO_THRESHOLD_S = 120`, `AUTO_CHUNK_S = 60`, `MIN_CHUNK_S = 5`, `MAX_CHUNK_S = 600`; `validate_chunk_size(seconds: int) -> int` (raises `ValueError` outside 5..600); `plan_chunks(n_samples: int, sr: int, mode: str, size_s: int = AUTO_CHUNK_S) -> list[tuple[int, int]]`; `concat_chunks(parts: list[np.ndarray]) -> np.ndarray`.

- [ ] **Step 1: Write the failing tests** (spec §5.3 exact values)

```python
SR = 48000
def test_auto_short_file_single_chunk():    # 60s -> [(0, 60*SR)] regardless of size_s
def test_auto_long_file_splits_at_60s():    # 130s, mode="auto" -> [(0,3600*..) chunks of 60s] -> 3 chunks, last (120s,130s)
def test_preset_size_used():                # mode="preset", size_s=30, 70s -> 3 chunks (30,30,10)
def test_custom_size_validation():          # validate_chunk_size(4) and (601) raise ValueError; 5 and 600 pass
def test_invalid_mode_raises():             # mode="bogus" -> ValueError
def test_concat_chunks_roundtrip_length():  # random arrays split by plan_chunks, each enhanced identity -> concat total == n_samples
def test_empty_audio_single_empty_chunk():  # n_samples=0 -> [(0, 0)]
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement `gui/core/chunker.py`**

`plan_chunks`: duration = `n_samples / sr`; auto -> one chunk when `duration <= AUTO_THRESHOLD_S`, else step `AUTO_CHUNK_S * sr`; preset/custom -> step `size_s * sr`; always append a final `(cursor, n_samples)` if `cursor < n_samples`. `concat_chunks` = `np.concatenate(parts, axis=-1)`.

- [ ] **Step 4: Run tests + lint + commit.**

```bash
git add gui/core/chunker.py tests/gui/test_chunker.py
git commit -m "feat(gui): add in-memory audio chunk planner"
```

---

### Task 5: JobConfig + EnhancementBackend

**Files:**
- Create: `gui/core/backend.py`
- Test: `tests/gui/test_backend.py`

**Interfaces:**
- Consumes: `df.enhance.init_df(model_base_dir=None, post_filter=False, log_level="INFO", log_file=None, epoch="best", mask_only=False) -> (model, df_state, suffix, epoch)`; `df.enhance.enhance(model, df_state, audio: Tensor[C,T], pad=True, atten_lim_db=None) -> Tensor`; Task 2 `apply_device`; Task 3 `resolve_model_dir`.
- Produces (Task 7 relies on these exact names):

```python
@dataclass(frozen=True)
class JobConfig:
    model: str = "DeepFilterNet3"; epoch: str = "best"; post_filter: bool = True
    atten_lim_db: int | None = None          # None/0 = disabled (CLI default)
    device: str = "Auto"; no_df_stage: bool = False; delay_compensation: bool = True
    chunk_mode: str = "auto"; chunk_size_s: int = 60
    output_dir: str = "./out"; output_format: str = "wav"; suffix_enabled: bool = True
    log_level: str = "INFO"

class EnhancementBackend(ABC):
    def enhance_chunk(self, audio: np.ndarray, cfg: JobConfig) -> np.ndarray: ...
    def cancel(self) -> None: ...
    def shutdown(self) -> None: ...

class InProcessBackend(EnhancementBackend): ...
```

- [ ] **Step 1: Write the failing tests** (patch `df.enhance.init_df` and `df.enhance.enhance` — the source modules via sys.modules, NOT `gui.core.backend.*` per R5: lazy function-level imports re-resolve each call, and `df/__init__.py` re-exports shadow the submodule attribute so dotted-string patching cannot work)

```python
def test_first_enhance_initializes_once():      # 2 chunks same cfg -> init_df called 1x, enhance 2x
def test_cache_invalidated_on_param_change():   # change epoch / post_filter / device / model -> new init_df call
def test_device_applied_before_init_df():       # cfg.device="CUDA:1" -> inside fake init_df spy, os.environ["DEVICE"]=="cuda:1"
def test_auto_device_clears_env():              # cfg.device="Auto" -> spy sees "DEVICE" absent
def test_pad_and_atten_mapped():                # cfg.delay_compensation=False -> enhance called with pad=False; atten_lim_db=12 -> atten_lim_db=12
def test_no_df_stage_maps_to_mask_only():       # cfg.no_df_stage=True -> init_df kwarg mask_only=True
def test_enhance_chunk_shape_preserved():       # input (2, 48000) float32 -> output same shape, float32, finite
def test_cancel_is_noop_v1():                   # cancel() then enhance_chunk still works (boundary cancel owned by JobQueue, spec §3.3)
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement `gui/core/backend.py`**

`InProcessBackend._cache: dict[tuple, tuple]` keyed by `(model, epoch, post_filter, no_df_stage, device)`. `_ensure_model(cfg)`: `apply_device(cfg.device)`, then on miss call lazy-imported `init_df(model_base_dir=cfg.model, post_filter=cfg.post_filter, log_level=cfg.log_level, log_file=None, epoch=cfg.epoch, mask_only=cfg.no_df_stage)` and store. `enhance_chunk`: `torch.from_numpy(audio)` -> `enhance(...)` -> `.cpu().numpy().astype(np.float32)`; `np.ascontiguousarray` before returning. `shutdown()` clears cache.

- [ ] **Step 4: Run tests + lint + commit.**

```bash
git add gui/core/backend.py tests/gui/test_backend.py
git commit -m "feat(gui): add in-process enhancement backend with model cache"
```

---

### Task 6: EventBus + Logging Bridge

**Files:**
- Create: `gui/core/events.py`
- Test: `tests/gui/test_events.py`

**Interfaces:**
- Consumes: `loguru.logger` (df dependency) for the bridge.
- Produces:

```python
@dataclass(frozen=True)
class AppEvent:
    type: str            # "job_state" | "job_progress" | "log" | "runtime"
    payload: dict

class EventBus:
    def subscribe(self, fn: Callable[[AppEvent], None]) -> None: ...
    def publish(self, event: AppEvent) -> None: ...        # thread-safe
    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None: ...

def setup_logging(bus: EventBus, level: str, log_dir: Path) -> Path: ...  # returns log file path
def setup_df_log_bridge(bus: EventBus, level: str) -> None: ...           # loguru sink -> bus "log" events
```

- [ ] **Step 1: Write the failing tests**

```python
def test_publish_dispatches_to_subscribers():     # 2 subscribers each get the event
def test_publish_without_loop_is_synchronous():   # no bind_loop -> subscriber called before publish returns
def test_publish_with_loop_schedules_coroutine(): # bind fake loop; publish from another thread -> loop.run_coroutine_threadsafe called once (mock)
def test_publish_from_multiple_threads():        # 8 threads x 50 events -> all delivered, no exception
def test_setup_logging_creates_rotating_file(tmp_path):  # log_dir file "enhance.log" exists after logger.info; handler is RotatingFileHandler
def test_log_bridge_forwards_loguru_to_bus():     # setup_df_log_bridge; logger.info("hello") -> a bus event type "log" containing "hello"
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement `gui/core/events.py`**

`subscribe` stores callbacks in a `list` guarded by `threading.Lock`. `publish`: if loop bound, `asyncio.run_coroutine_threadsafe(self._dispatch(e), loop)` where `_dispatch` awaits each callback in an async wrapper (callbacks still may be sync callables — call them directly inside the coroutine); else dispatch inline. `setup_logging`: `logging.handlers.RotatingFileHandler(log_dir / "enhance.log", maxBytes=1_000_000, backupCount=3)` on root logger. `setup_df_log_bridge`: `loguru.logger.add(lambda m: bus.publish(AppEvent("log", {"level": m.record["level"].name, "text": str(m).rstrip()})), level=level)`; return the sink id from the function so it can be removed later (store as function attribute `setup_df_log_bridge.last_sink_id`).

- [ ] **Step 4: Run tests + lint + commit.**

```bash
git add gui/core/events.py tests/gui/test_events.py
git commit -m "feat(gui): add thread-safe event bus and log bridge"
```

---

### Task 7: JobQueue + Output Pipeline

**Files:**
- Create: `gui/core/jobs.py`
- Test: `tests/gui/test_jobs.py`

**Interfaces:**
- Consumes: Task 3 `JobConfig`, Task 4 `plan_chunks`/`concat_chunks`, Task 5 `EnhancementBackend`, Task 6 `EventBus`/`AppEvent`, `df.io.load_audio/save_audio/resample`, `df.model.ModelParams().sr` (model sample rate, lazy import).
- Produces:

```python
class JobState(str, Enum): QUEUED RUNNING PAUSED DONE FAILED CANCELLED

@dataclass
class Job:
    id: str; files: list[Path]; cfg: JobConfig
    state: JobState = JobState.QUEUED
    file_results: dict[str, str]      # file path str -> "done"|"failed: <msg>"|"pending"|"running"|"cancelled"
    progress: float = 0.0             # 0..1 across all files/chunks
    error: str | None = None          # job-level message when state == FAILED

def build_output_path(src: Path, cfg: JobConfig) -> Path   # cfg.output_dir / "<stem>-deep-filtered-<n>.<fmt>"
def supported_formats() -> list[str]                       # real-writer probe: ["wav"] + flac/mp3 iff df.io.save_audio round-trip succeeds (R13; env-dependent)

class JobQueue:
    def __init__(self, backend: EnhancementBackend, bus: EventBus) -> None
    def submit(self, files: list[Path], cfg: JobConfig) -> Job    # validates formats first
    def pause(self, job_id: str) -> None
    def resume(self, job_id: str) -> None
    def cancel(self, job_id: str) -> None
    def retry(self, job_id: str) -> None          # requeues failed job with same snapshot
    def start(self) -> None                       # spawns worker thread
    def shutdown(self) -> None                    # cancel all, join thread, backend.shutdown()
    def wait_idle(self, timeout: float = 30.0) -> bool   # test helper: True when no RUNNING/QUEUED work
```

- [ ] **Step 1: Write the failing tests** (use a `FakeBackend(EnhancementBackend)` that returns `audio * 0.5`, records call shapes, and can be armed to raise `RuntimeError("CUDA out of memory")`; write tiny WAVs with `df.io.save_audio` into `tmp_path`)

```python
def test_submit_completes_and_writes_output(tmp_path):      # 0.5s wav -> DONE, output exists, name == "x-deep-filtered.wav"
def test_output_collision_gets_numeric_suffix(tmp_path):    # run twice -> second file named "x-deep-filtered-2.wav"
def test_full_file_output_length_matches_input(tmp_path):   # 2.5s file, chunk 1s -> output frames == input frames
def test_pause_resumes_at_next_chunk(tmp_path):             # FakeBackend counts calls; pause during chunk 2 -> after resume total calls == n_chunks (no chunk 2 repeat)
def test_cancel_discards_partial_output(tmp_path):          # cancel mid-file -> state CANCELLED, no output file on disk
def test_failed_file_does_not_block_others(tmp_path):       # [good.wav, bad.txt] -> job FAILED, good.wav result "done", bad.txt starts with "failed:"
def test_oom_error_includes_actionable_hint(tmp_path):      # armed OOM -> file_results msg contains "chunk" and "CPU"
def test_unsupported_format_rejected_at_submit(tmp_path):   # submit(["x.wav"], cfg output_format="mp3") when supported_formats() patched without mp3 -> ValueError before backend called
def test_progress_events_published(tmp_path):               # bus captures "job_progress" events with monotonically nondecreasing payload["progress"]
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement `gui/core/jobs.py`**

Worker loop (single thread, `threading.Event` for wakeups): take first `QUEUED` job (FIFO), for each remaining file (track per-job cursor `file_index`, `chunk_index` so pause/resume continues): lazy `load_audio(str(path), sr=ModelParams().sr)` -> `audio, info = ...`; `orig_sr = info.sample_rate`; plan chunks on `audio.numpy()`; per chunk check `pause_event`/`cancel_event` first (if set: persist cursor, mark `PAUSED` or `CANCELLED` and stop); call `backend.enhance_chunk`; publish `job_progress`. After all chunks: `np.concatenate` -> `resample(out, df_sr, orig_sr)` -> `save_audio(str(build_output_path(...)), out, sr=orig_sr)`. Per-file exceptions: `RuntimeError` with "out of memory" in message -> hint `"Reduce chunk size or switch device to CPU"`; any other -> raw message; mark file failed, continue next file. Job final state: `DONE` if every file done, else `FAILED` with `error="1 of 2 files failed"`. `supported_formats()` probes `df.io.save_audio` of a 1-frame wav/flac/mp3 to a `tempfile` in try/except (real-writer probe per R13; result cached module-level). `submit()` raises `ValueError` if `cfg.output_format not in supported_formats()`. Publish `job_state` events on every transition. All `df`/`torch` imports lazy inside the worker function.

- [ ] **Step 4: Run tests + lint + commit.**

```bash
git add gui/core/jobs.py tests/gui/test_jobs.py
git commit -m "feat(gui): add job queue with chunked pipeline and cancellation"
```

---

### Task 8: Managed Runtime + Process Bootstrap

**Files:**
- Create: `gui/core/dependency.py`, `gui/main.py` (bootstrap half only; UI wiring in Task 9), `requirements-gui-runtime.txt`
- Test: `tests/gui/test_dependency.py`

**Interfaces:**
- Consumes: `platformdirs.user_data_dir`, `pip` behavior, Task 3 `ensure_model`.
- Produces:

```python
def runtime_dir() -> Path                       # user_data_dir("deepfilternet-gui") / "runtime"
def current_env_ready() -> bool                 # find_spec for torch, soundfile, flet, df all present
def detect_cuda_tag() -> str | None             # "cu128"-style tag from `nvidia-smi --query-gpu=driver_version...`; None if absent
def ensure_runtime(progress: Callable[[str, float], None]) -> RuntimeInfo   # creates venv, installs torch(+CUDA index), then ./DeepFilterNet, flet, platformdirs
def ensure_gui_process() -> bool                # False = continue here; True = relaunched elsewhere (caller must exit)
@dataclass(frozen=True) class RuntimeInfo: mode: str; python: Path
```

- [ ] **Step 1: Write the failing tests** (mock `subprocess.run`, `venv` creation, and `importlib.util.find_spec`)

```python
def test_current_env_ready_true_in_repo_venv():          # in .venv, find_spec real -> True
def test_current_env_ready_false_when_torch_missing(monkeypatch):  # patch find_spec -> None for torch -> False
def test_detect_cuda_tag_parses_nvidia_smi(monkeypatch): # fake stdout "CUDA Version: 12.8" -> "cu128"; FileNotFoundError -> None
def test_ensure_runtime_installs_in_order(monkeypatch):  # recorded pip calls: [torch --index-url .../cu128, ./DeepFilterNet, flet platformdirs]; progress called with phase labels "venv", "torch", "deps"
def test_ensure_runtime_cpu_fallback(monkeypatch):       # detect_cuda_tag None -> pip install torch (default index), no --index-url arg
def test_ensure_gui_process_noop_when_ready(monkeypatch):    # current_env_ready True -> returns False, no subprocess spawned
def test_ensure_gui_process_relaunches_when_not_ready(monkeypatch):  # ensure_runtime mocked; spawn call args contain [runtime python, "-m", "gui", "--runtime-child"] and env PYTHONPATH includes repo root; returns True
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement `gui/core/dependency.py` + bootstrap in `gui/main.py`**

`requirements-gui-runtime.txt`: `flet>=1.0,<2.0`, `platformdirs`, `./DeepFilterNet` (local path installs `df` + torch/torchaudio deps). `ensure_runtime`: `venv.create(runtime_dir()/"venv", with_pip=True)` if absent; pip install sequence per test above with `check=True`; wrap each pip call so failures raise `RuntimeError` with the pip stderr tail (SetupDialog shows it). `ensure_gui_process`: if `current_env_ready()` return `False`; else `ensure_runtime(progress=lambda *_: None)`; then `subprocess.Popen([venv_python, "-m", "gui", "--runtime-child"], env={**os.environ, "PYTHONPATH": repo_root})` and return `True`. `gui/main.py` bootstrap order: parse `--runtime-child` (argparse, before flet import) -> `ensure_gui_process()` (exit if True) -> Task 9 UI startup. If torch is missing but flet present (source run), the relaunch happens BEFORE any UI; the SetupDialog path (Task 12) covers the interactive case by calling `ensure_runtime` with a visible progress callback instead of the silent default — Task 12 replaces the silent lambda, not this logic.

- [ ] **Step 4: Run tests + lint + commit.**

```bash
git add gui/core/dependency.py gui/main.py requirements-gui-runtime.txt tests/gui/test_dependency.py
git commit -m "feat(gui): add managed runtime bootstrap and process relaunch"
```

---

### Task 9: Flet App Skeleton (Navigation, Theme, Selftest)

**Files:**
- Create: `gui/app.py`, `gui/ui/__init__.py`, `gui/ui/pages/__init__.py`, `gui/ui/pages/enhance_view.py`, `gui/ui/pages/queue_view.py`, `gui/ui/pages/log_view.py`, `gui/ui/pages/settings_view.py`, `gui/resources/` (empty dir, placeholder for splash/icon assets)
- Modify: `gui/main.py` (UI startup half)
- Test: `tests/gui/test_app_skeleton.py`

**Interfaces:**
- Consumes: Task 1 `ConfigStore`, Task 6 `EventBus`, Task 7 `JobQueue` (constructed but not started).
- Produces: `class DeepFilterApp` with `build() -> ft.Control` (root column: `NavigationRail` + view container + status bar); view classes `EnhanceView(cfg: ConfigStore, queue: JobQueue, bus: EventBus)`, `QueueView(queue, bus)`, `LogView(bus)`, `SettingsView(cfg: ConfigStore)` each with `build() -> ft.Control` (placeholder content in this task); `main(page: ft.Page)` entry in `gui/app.py`; `gui/main.py` runs `ft.run(gui.app.main)` (flet 1.0 renamed ft.app).

- [ ] **Step 1: Write the failing test**

```python
def test_all_views_build_controls():     # instantiate each view with a real ConfigStore(tmp), JobQueue(FakeBackend, EventBus), EventBus -> .build() returns ft.Control
def test_navigation_has_four_destinations():  # build() -> rail.destinations labels == ["Enhance", "Queue", "Log", "Settings"]
def test_selftest_exits_zero():          # subprocess `.venv/bin/python -m gui --selftest` builds all views w/o opening window, returncode == 0
def test_theme_loaded_from_config(tmp_path):  # cfg theme_mode "DARK" -> app.build() reflects theme_mode on page or theme control
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement skeleton**

`gui/app.py`: `DeepFilterApp.__init__(page, cfg, bus, queue)` — sets `page.theme_mode` from config, `page.theme.color_scheme_seed`, window size from config, binds bus to `page.loop` (`bus.bind_loop(page.loop)`) and subscribes a dispatcher that applies payload updates and calls `page.update()` **only from the loop thread**. Persist `last_view` to `ConfigStore` on rail change and `window_width`/`window_height` on window resize/close (spec §9), debounced 1 s. `main(page)` wires the four views and rail switching (`on_change` swaps the container content). `--selftest` in `gui/main.py`: construct `ConfigStore` in a tmp dir + views, print "selftest ok", `sys.exit(0)` — never calls `ft.app`.

- [ ] **Step 4: Run tests + `--selftest` manually + lint + commit.**

Run: `.venv/bin/python -m pytest tests/gui/test_app_skeleton.py -v && .venv/bin/python -m gui --selftest` — expected PASS + `selftest ok`.

```bash
git add gui/app.py gui/main.py gui/ui tests/gui/test_app_skeleton.py
git commit -m "feat(gui): add flet app skeleton with navigation and theme"
```

---

### Task 10: Enhance View (File List + Standard/Advanced Options + Progress)

**Files:**
- Create: `gui/ui/pages/enhance_view.py` (Task 9 placeholder replaced), `gui/ui/widgets/__init__.py`, `gui/ui/widgets/file_list.py`
- Test: `tests/gui/test_enhance_view.py`

**Interfaces:**
- Consumes: `ConfigStore`, `JobQueue.submit`, `EventBus`, `JobConfig` (Task 5), `available_devices()` (Task 2), `model_choices()` + `resolve_model_dir()`/`ModelError` (Task 3 — custom path validated at snapshot time), `supported_formats()` (Task 7), `validate_chunk_size` (Task 4), `ft.FilePicker`.
- Produces: `EnhanceView.build()` full layout per spec §4; `EnhanceView.snapshot() -> JobConfig` (reads current control values into a `JobConfig`); `FileList` widget with `add_paths(paths: list[str])`, `paths() -> list[Path]`, `clear()`; standard controls: model dropdown (`model_choices()`), atten slider 0–60 with `0 = off` mapping to `atten_lim_db=None`, postfilter `ft.Switch`, device dropdown (`available_devices()`), output dir picker, format dropdown (`supported_formats()`), suffix switch; advanced `ft.Expander` with epoch text field, delay-compensation switch (default on), `no_df_stage` switch, log-level dropdown, chunk mode dropdown (Auto/Preset/Custom) + size number field (validated 5–600, preset sizes 30/60/120/300); `Enhance All` button; two `ft.ProgressBar` (per-file, global); status line.

- [ ] **Step 1: Write the failing tests**

```python
def test_snapshot_maps_controls_to_jobconfig(view):   # set atten 12, model DeepFilterNet2, device CPU -> JobConfig(atten_lim_db=12, model=..., device="CPU")
def test_atten_zero_maps_to_none(view):               # slider 0 -> cfg.atten_lim_db is None
def test_chunk_size_validation_error_shown(view):     # enter 3 -> error text on field, submit blocked
def test_add_files_updates_list_and_persists(view, tmp_path):  # add_paths -> paths() correct; cfg store updated after save
def test_enhance_all_submits_job(view, fake_queue):    # 2 files -> fake_queue.submitted == 1 job with 2 files
def test_device_dropdown_populated(monkeypatch):       # available_devices patched -> control options match
def test_progress_event_updates_bars(view):            # publish job_progress 0.5 -> global progress bar value 0.5
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement the view**

Wire `FilePicker` (one for files, one for directory) to `FileList.add_paths`. `Enhance All`: build `JobConfig` via `snapshot()` (block on validation errors), `queue.start()` if not running, `queue.submit(...)`, persist current control values to `ConfigStore` + `save()`. Bus subscription (registered by the view) updates progress bars/status on `job_progress`/`job_state` events; because the bus is loop-bound by Task 9, all updates are loop-thread safe. Controls that change rarely (device list) populate once at build.

- [ ] **Step 4: Run tests + lint + commit.**

```bash
git add gui/ui tests/gui/test_enhance_view.py
git commit -m "feat(gui): add enhance view with options and progress"
```

---

### Task 11: Queue, Log, and Settings Views

**Files:**
- Create: `gui/ui/widgets/console.py`, replace placeholders in `gui/ui/pages/queue_view.py`, `gui/ui/pages/log_view.py`, `gui/ui/pages/settings_view.py`
- Test: `tests/gui/test_views.py`

**Interfaces:**
- Consumes: `JobQueue.pause/resume/cancel/retry` + `Job.id/state/progress/file_results` (Task 7), `AppEvent` types (Task 6), `ConfigStore` (Task 1), `runtime_dir/current_env_ready` (Task 8).
- Produces: `QueueView.refresh() -> None` (renders job cards with state chip, per-file results, Pause/Resume/Cancel/Retry buttons enabled per state, Up/Down reorder); `Console(level_filter: str = "INFO")` widget with `append(event)` ring buffer (max 2000 lines), level dropdown filter, Clear/Copy/Save buttons; `LogView` = `Console` + level control persisted to config; `SettingsView`: default output dir picker, theme mode dropdown (System/Light/Dark), "Runtime repair" button (calls `ensure_runtime` with progress events), config file path display (read-only, from `cfg.path`).

- [ ] **Step 1: Write the failing tests**

```python
def test_queue_renders_job_and_buttons(queue_view, job):     # after submit: card shows state; cancel enabled while RUNNING/QUEUED
def test_pause_resume_cancel_buttons_call_queue(fake_queue_view):  # click simulation via control on_click callbacks -> fake queue methods called with job.id
def test_retry_requeues_failed_job(fake_queue, failed_job):  # retry -> job state back to QUEUED
def test_console_filters_by_level(console):                  # append INFO+DEBUG lines, filter INFO -> DEBUG line hidden
def test_console_caps_at_2000_lines(console):                # append 2500 -> len == 2000, oldest dropped
def test_settings_theme_change_persists(settings_view, tmp_path):  # set LIGHT, save -> reloaded store theme_mode == "LIGHT"
def test_runtime_repair_publishes_progress(settings_view, fake_bus):  # button -> runtime events with phase labels present
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement the three views + `Console` widget.**

Queue reorder = swap indices in an internal ordered id list used by `refresh()` (job processing order follows `JobQueue` internal list — expose `JobQueue.move_up(job_id)`/`move_down(job_id)` and add tests for them in `tests/gui/test_jobs.py` as an extra step: `test_move_up_reorders_pending_jobs`).

- [ ] **Step 4: Run tests + lint + commit.**

```bash
git add gui/ui tests/gui/test_views.py tests/gui/test_jobs.py
git commit -m "feat(gui): add queue, log, and settings views"
```

---

### Task 12: Setup Dialog + First-Run Flow + Model Download Status

**Files:**
- Create: `gui/ui/widgets/setup_dialog.py`, modify `gui/main.py` and `gui/app.py`
- Test: `tests/gui/test_setup_dialog.py`

**Interfaces:**
- Consumes: `ensure_runtime(progress)` + `current_env_ready` (Task 8), `ensure_model` (Task 3), `EventBus` "runtime" events, `EnhanceView` submit path (Task 10).
- Produces: `SetupDialog(on_retry, on_skip)` with `open(progress_cb)` — phases `venv`/`torch`/`deps` shown as `ft.ProgressBar` + label + stderr tail on failure; Retry/Skip buttons; `main.py` flow: `current_env_ready()` false AND flet importable -> show SetupDialog before main window (run its own minimal `ft.app`), on success relaunch child (`ensure_gui_process`) and exit; on Skip -> continue into main app with `RuntimeInfo(mode="degraded")`; `AppEvent("runtime", ...)` publisher; Enhance view disables Enhance All + shows banner when degraded; first submit with model not yet downloaded publishes `{"phase": "model_download"}` to status bar, handled by a subscriber in `app.py`.

- [ ] **Step 1: Write the failing tests**

```python
def test_setup_dialog_reports_phases(fake_bus):       # simulate progress("venv", .1), ("torch", .5) -> labels updated
def test_setup_dialog_shows_error_and_retry(tmp_path):# ensure_runtime raises RuntimeError("pip failed tail") -> error text visible, retry callback re-invokes ensure_runtime
def test_skip_disables_enhance(degraded_view):        # runtime degraded -> Enhance All button disabled + banner visible
def test_model_download_status_event(app_subscriber): # publish runtime event phase=model_download -> status bar text contains "Preparing model"
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement.** `SetupDialog` runs `ensure_runtime` in a `page.run_thread` worker, hopping progress back via `bus.publish` (loop-bound as usual). Wire `main.py` order: argparse -> `ensure_gui_process()` (silent relaunch only when flet itself would be missing... in practice flet present, torch missing -> interactive dialog path) -> SetupDialog when `not current_env_ready()` -> main `ft.app`.

- [ ] **Step 4: Run tests + lint + commit.**

```bash
git add gui/ui gui/main.py gui/app.py tests/gui/test_setup_dialog.py
git commit -m "feat(gui): add first-run setup dialog and degraded mode"
```

---

### Task 13: End-to-End Pipeline Test (Real Model, CPU)

**Files:**
- Test: `tests/gui/test_e2e_pipeline.py`
- Modify: `tests/gui/conftest.py` (create: shared `FakeBackend`, `make_wav` helpers promoted from Task 7)

**Interfaces:**
- Consumes: everything: `InProcessBackend`, `JobQueue`, `EventBus`, real `df` model from `models/` (repo contains pretrained zips) or download.

- [ ] **Step 1: Write the test**

```python
@pytest.mark.e2e
def test_real_enhance_roundtrip(tmp_path):
    # generate 1.5s of noise+tone at 48kHz via numpy, save with df.io.save_audio
    # cfg = JobConfig(device="CPU", chunk_mode="preset", chunk_size_s=1, output_dir=tmp_path)
    # queue = JobQueue(InProcessBackend(), bus); submit 1 file; wait_idle()
    # assert job.state == DONE
    # out = load_audio(output_path); assert frames == input frames, np.isfinite(out).all()
    # assert output louder file was actually processed (out != input)  -- use allclose False
```

Skip marker: `pytest.mark.skipif(not model_available(), reason="model not available offline")` where `model_available()` checks `models/` zips or network once.

- [ ] **Step 2: Run it.** Run: `.venv/bin/python -m pytest tests/gui/test_e2e_pipeline.py -v` — Expected: PASS on CPU (first run may download model; allow ~3 min timeout).

- [ ] **Step 3: Refactor shared fixtures + run whole suite.**

Run: `.venv/bin/python -m pytest tests/gui -v` — Expected: all green.

```bash
git add tests/gui
git commit -m "test(gui): add end-to-end enhance pipeline test"
```

---

### Task 14: Packaging + CI Workflow

**Files:**
- Modify: `deepfilter-gui.spec` (adapt entry/excludes for Flet)
- Create: `.github/workflows/build-gui.yml` (revived + adapted), `SETUP.md`
- Reference: `git show 7d066a2:.github/workflows/build-gui.yml`, `git show 7d066a2:SETUP.md` (legacy, read-only)

**Interfaces:**
- Consumes: `--selftest` flag (Task 9), `requirements-gui.txt` (Task 1), `gui/main.py` entry.
- Produces: distributable artifact; `flet pack gui/main.py` as primary build command; `deepfilter-gui.spec` kept as fallback for direct `pyinstaller deepfilter-gui.spec`.

- [ ] **Step 1: Local build verification (Linux)**

Run: `.venv/bin/flet pack gui/main.py --name DeepFilterNet-GUI` then `./dist/DeepFilterNet-GUI --selftest` (adjust to actual dist layout `dist/gui/`).
Expected: build succeeds (torch excluded — verify `torch` NOT in `build/` imports list), selftest exits 0.
If `flet pack` fails on hidden imports, fall back: adapt `deepfilter-gui.spec` (entry `gui/main.py`, `hiddenimports` for `flet`, `excludes=["torch", "tensorflow"]`, onedir) and run `.venv/bin/pyinstaller deepfilter-gui.spec` — record which path worked in `SETUP.md`.

- [ ] **Step 2: Write `SETUP.md`** (English, no emojis): source-run steps (clone, `.venv`, `pip install -r requirements.txt -r requirements-gui.txt`, `python -m gui`), first-run behavior (managed venv, model download), packaged-build steps, platform notes (Windows/Linux; macOS best-effort).

- [ ] **Step 3: CI workflow**

Create `.github/workflows/build-gui.yml` modeled on `git show 7d066a2:.github/workflows/build-gui.yml`: matrix `ubuntu-latest` + `windows-latest`; steps: checkout, setup-python 3.11, install `requirements.txt` + `requirements-gui.txt`, run `pytest tests/gui -m "not e2e"` (CPU, no model download), run `flet pack`, upload `dist/` artifact.

- [ ] **Step 4: Lint + verify workflow syntax + commit**

Run: `black --check gui tests && flake8 gui tests && .venv/bin/python -m pytest tests/gui -m "not e2e" -v`
Expected: all green.

```bash
git add deepfilter-gui.spec SETUP.md .github/workflows/build-gui.yml
git commit -m "build(gui): add flet packaging and CI workflow"
```

---

## Execution Notes

- Tasks are strictly ordered: each consumes the previous task's `Interfaces`. A fresh subagent sees only its own task — the Interfaces blocks are the contract; do not rename across tasks.
- Do not import `torch`/`df` at module import time anywhere in `gui/` (degraded mode + packaged startup depend on it).
- Legacy `gui/__pycache__/` contents are reference artifacts; new source files may share paths with them — do not delete the `.pyc` files.
