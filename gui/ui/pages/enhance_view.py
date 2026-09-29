"""Enhance view: file list, standard/advanced options and progress (Task 10).

Layout follows spec 4: file list, output controls + Enhance All, Standard
options (always visible), Advanced options (collapsed) and per-file/global
progress with a status line. Control values are read into a ``JobConfig``
snapshot on submit and persisted to the ``ConfigStore`` on Enhance All;
every option edit also notifies ``on_options_changed`` so the app can
debounce-persist them (spec 9: save on window close + 1 s after changes).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import flet as ft

from gui.core.backend import JobConfig
from gui.core.chunker import validate_chunk_size
from gui.core.config import ConfigStore
from gui.core.device import available_devices
from gui.core.events import AppEvent, EventBus
from gui.core.jobs import JobQueue, supported_formats
from gui.core.models import ModelError, model_choices, resolve_model_dir
from gui.ui.widgets import FileList

# Chunk mode labels. plan_chunks() accepts only "auto"/"preset", so a Custom
# size is submitted as "preset" with a user-chosen length; "custom" is kept
# only for ConfigStore round-tripping of the UI mode.
CHUNK_MODE_AUTO = "Auto"
CHUNK_MODE_PRESET = "Preset"
CHUNK_MODE_CUSTOM = "Custom"
_CHUNK_MODES = (CHUNK_MODE_AUTO, CHUNK_MODE_PRESET, CHUNK_MODE_CUSTOM)
_CHUNK_MODE_TO_JOB = {
    CHUNK_MODE_AUTO: "auto",
    CHUNK_MODE_PRESET: "preset",
    CHUNK_MODE_CUSTOM: "preset",
}
_CHUNK_MODE_TO_CONFIG = {
    CHUNK_MODE_AUTO: "auto",
    CHUNK_MODE_PRESET: "preset",
    CHUNK_MODE_CUSTOM: "custom",
}
_CHUNK_MODE_LABELS = {value: key for key, value in _CHUNK_MODE_TO_CONFIG.items()}
PRESET_CHUNK_SIZES = (30, 60, 120, 300)
DEFAULT_CHUNK_S = 60

CUSTOM_MODEL_LABEL = "Custom path"
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
NO_FORMAT_STATUS = "No writable audio format available"


def _clamp(value: float) -> float:
    """Clamp a progress fraction to 0..1."""
    return min(max(value, 0.0), 1.0)


class EnhanceView:
    """Enhance page: file list, enhancement options and progress.

    Controls are created once in ``__init__``; ``build()`` composes them so
    bus-driven updates always target the controls the last tree contains.
    The view subscribes to the event bus once, at construction.
    """

    def __init__(
        self,
        cfg: ConfigStore,
        queue: JobQueue,
        bus: EventBus,
        on_options_changed: Callable[[], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self.queue = queue
        self.bus = bus
        # Fires on every option edit; the app persists the snapshot with
        # debounce (spec 9). None keeps the view standalone.
        self._on_options_changed = on_options_changed
        initial = cfg.load()

        # Inputs and standard options.
        self.file_list = FileList()
        # Service controls auto-register with the page on construction
        # (Flet 1.0 Service.__post_init__); they must NOT go into the view
        # tree or page.overlay (the client would try to render them as
        # widgets: "Unknown control"). The extra tuple reference matters:
        # after every event Flet drops services whose refcount shows no live
        # owner (session.unregister_services), so a single attribute is not
        # enough to keep the pickers alive.
        self.file_picker = ft.FilePicker()
        self.dir_picker = ft.FilePicker()
        self._pickers = (self.file_picker, self.dir_picker)

        devices = available_devices()
        # Strict: an empty list means the writer cannot produce any format;
        # never offer a fallback that submit() would reject anyway.
        formats = supported_formats()
        stored_format = str(initial.get("output_format") or "")
        format_value = (
            stored_format if stored_format in formats else (formats[0] if formats else None)
        )
        choices = model_choices()
        model_value = str(initial.get("model") or "")
        custom_path = "" if model_value in choices else model_value
        self.model_dropdown = ft.Dropdown(
            label="Model",
            value=model_value if model_value in choices else CUSTOM_MODEL_LABEL,
            options=[ft.DropdownOption(key=choice, text=choice) for choice in choices],
            on_select=self._on_model_change,
        )
        self.custom_model_field = ft.TextField(
            label="Model directory",
            value=custom_path,
            visible=bool(custom_path),
            on_change=self._on_option_edited,
        )
        self.atten_slider = ft.Slider(
            label="{value}",
            min=0,
            max=60,
            divisions=60,
            value=self._initial_atten(initial.get("atten_lim_db")),
            on_change=self._on_option_edited,
        )
        self.postfilter_switch = ft.Switch(
            label="Postfilter",
            value=bool(initial.get("post_filter", True)),
            on_change=self._on_option_edited,
        )
        self.device_dropdown = ft.Dropdown(
            label="Device",
            value=self._initial_choice(str(initial.get("device") or ""), devices, "Auto"),
            options=[ft.DropdownOption(key=device, text=device) for device in devices],
            on_select=self._on_option_edited,
        )

        # Output controls.
        self.output_dir_field = ft.TextField(
            label="Output directory",
            value=str(initial.get("output_dir") or "./out"),
            expand=True,
            on_change=self._on_option_edited,
        )
        self.browse_dir_button = ft.FilledButton("Browse...", on_click=self._on_pick_dir)
        self.format_dropdown = ft.Dropdown(
            label="Format",
            value=format_value,
            options=[ft.DropdownOption(key=fmt, text=fmt.upper()) for fmt in formats],
            on_select=self._on_option_edited,
        )
        self.suffix_switch = ft.Switch(
            label="Suffix -deep-filtered",
            value=bool(initial.get("suffix_enabled", True)),
            on_change=self._on_option_edited,
        )

        # Advanced options (collapsed by default).
        self.epoch_field = ft.TextField(
            label="Epoch",
            value=str(initial.get("epoch") or "best"),
            helper="best, latest or a checkpoint number",
            on_change=self._on_epoch_change,
        )
        self.delay_switch = ft.Switch(
            label="Delay compensation",
            value=bool(initial.get("delay_compensation", True)),
            on_change=self._on_option_edited,
        )
        self.no_df_switch = ft.Switch(
            label="No DF stage",
            value=bool(initial.get("no_df_stage", False)),
            on_change=self._on_option_edited,
        )
        self.log_level_dropdown = ft.Dropdown(
            label="Log level",
            value=self._initial_choice(
                str(initial.get("log_level") or "INFO").upper(), LOG_LEVELS, "INFO"
            ),
            options=[ft.DropdownOption(key=level, text=level) for level in LOG_LEVELS],
            on_select=self._on_option_edited,
        )
        mode_label = _CHUNK_MODE_LABELS.get(
            str(initial.get("chunk_mode") or "auto").lower(), CHUNK_MODE_AUTO
        )
        self.chunk_mode_dropdown = ft.Dropdown(
            label="Chunk mode",
            value=mode_label,
            options=[ft.DropdownOption(key=mode, text=mode) for mode in _CHUNK_MODES],
            on_select=self._on_chunk_mode_change,
        )
        self.chunk_size_field = ft.TextField(
            label="Chunk size (s)",
            value=str(initial.get("chunk_size_s") or DEFAULT_CHUNK_S),
            keyboard_type=ft.KeyboardType.NUMBER,
            helper="Presets: 30, 60, 120, 300. Custom: 5-600 seconds",
            disabled=mode_label == CHUNK_MODE_AUTO,
            on_change=self._on_chunk_size_change,
        )
        self.advanced = ft.ExpansionTile(
            title=ft.Text("Advanced options", weight=ft.FontWeight.W_600),
            controls=[
                ft.Row(
                    [
                        self.epoch_field,
                        self.delay_switch,
                        self.no_df_switch,
                        self.log_level_dropdown,
                    ],
                    spacing=12,
                    wrap=True,
                ),
                ft.Row(
                    [self.chunk_mode_dropdown, self.chunk_size_field],
                    spacing=12,
                    wrap=True,
                ),
            ],
            expanded=False,
        )

        # Progress and status (driven by bus events).
        self.file_progress = ft.ProgressBar(value=0.0, expand=True)
        self.global_progress = ft.ProgressBar(value=0.0, expand=True)
        self.status_text = ft.Text("Ready", size=12)

        # Actions.
        self.add_files_button = ft.FilledButton("Add Files...", on_click=self._on_add_files)
        self.enhance_button = ft.FilledButton("Enhance All", on_click=self._on_enhance)
        if not formats:
            self.enhance_button.disabled = True
            self.status_text.value = NO_FORMAT_STATUS

        # One subscription per view instance; app.py binds the bus to the page
        # loop, so callbacks run on the loop thread.
        bus.subscribe(self._on_event)

    # ----------------------------------------------------------------- layout

    def build(self) -> ft.Control:
        """Compose the full Enhance layout (spec 4)."""
        return ft.Column(
            [
                ft.Row(
                    [
                        ft.Text("Enhance", size=20, weight=ft.FontWeight.W_600),
                        ft.Row(
                            [self.add_files_button, self.enhance_button],
                            spacing=8,
                        ),
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                ),
                self.file_list.build(),
                ft.Row(
                    [
                        self.output_dir_field,
                        self.browse_dir_button,
                        self.format_dropdown,
                        self.suffix_switch,
                    ],
                    spacing=12,
                    wrap=True,
                ),
                ft.Text("Standard options", size=14, weight=ft.FontWeight.W_600),
                ft.Row(
                    [
                        self.model_dropdown,
                        self.custom_model_field,
                        ft.Column(
                            [
                                ft.Text("Atten limit (dB, 0 = off)", size=12),
                                self.atten_slider,
                            ],
                            spacing=2,
                        ),
                        self.postfilter_switch,
                        self.device_dropdown,
                    ],
                    spacing=12,
                    wrap=True,
                ),
                self.advanced,
                ft.Column(
                    [
                        ft.Text("Current file", size=12),
                        ft.Row([self.file_progress], expand=True),
                        ft.Text("Overall", size=12),
                        ft.Row([self.global_progress], expand=True),
                        self.status_text,
                    ],
                    spacing=4,
                ),
            ],
            spacing=12,
            expand=True,
        )

    # --------------------------------------------------------------- snapshot

    def snapshot(self) -> JobConfig:
        """Read current control values into an immutable ``JobConfig``.

        Raises ``ValueError`` for an invalid epoch/chunk size and
        ``ModelError`` when a custom model path fails validation.
        """
        mode = self.chunk_mode_dropdown.value or CHUNK_MODE_AUTO
        return JobConfig(
            model=self._model_value(),
            epoch=self._epoch_value(),
            post_filter=bool(self.postfilter_switch.value),
            atten_lim_db=self._atten_value(),
            device=self.device_dropdown.value or "Auto",
            no_df_stage=bool(self.no_df_switch.value),
            delay_compensation=bool(self.delay_switch.value),
            chunk_mode=_CHUNK_MODE_TO_JOB.get(mode, "auto"),
            chunk_size_s=self._chunk_size_value(),
            output_dir=self._output_dir_value(),
            output_format=self.format_dropdown.value or "",
            suffix_enabled=bool(self.suffix_switch.value),
            log_level=self.log_level_dropdown.value or "INFO",
        )

    def validate(self) -> bool:
        """Set inline error texts; True when every control is valid."""
        ok = True
        self.custom_model_field.error = None
        if self.model_dropdown.value == CUSTOM_MODEL_LABEL:
            try:
                self._model_value()
            except ModelError as exc:
                self.custom_model_field.error = str(exc)
                ok = False
        self.epoch_field.error = None
        try:
            self._epoch_value()
        except ValueError as exc:
            self.epoch_field.error = str(exc)
            ok = False
        self.chunk_size_field.error = None
        if self.chunk_mode_dropdown.value != CHUNK_MODE_AUTO:
            try:
                self._chunk_size_value()
            except ValueError as exc:
                self.chunk_size_field.error = str(exc)
                ok = False
        return ok

    def enhance_all(self) -> bool:
        """Validate, snapshot, submit the file list and persist settings.

        Returns False (and updates the status line) when validation fails,
        no files are queued or the queue rejects the job.
        """
        if not self.format_dropdown.options:
            self._set_status(NO_FORMAT_STATUS)
            return False
        if not self.validate():
            self._set_status("Fix the highlighted options before starting")
            return False
        paths = self.file_list.paths()
        if not paths:
            self._set_status("Add at least one input file first")
            return False
        try:
            cfg = self.snapshot()
        except (ModelError, ValueError) as exc:
            self._set_status(str(exc))
            return False
        try:
            self.queue.start()  # no-op when the worker already runs
            self.queue.submit(paths, cfg)
        except ValueError as exc:  # JobQueue rejects unsupported formats
            self._set_status(str(exc))
            return False
        # Start a fresh run at 0%; live job events take over from there.
        self.file_progress.value = 0.0
        self.global_progress.value = 0.0
        self.persist_settings()
        self._set_status(f"Queued {len(paths)} file(s)")
        return True

    def persist_settings(self) -> None:
        """Write current control values to the ConfigStore and save them."""
        for key, value in self.options_snapshot().items():
            self.cfg.set(key, value)
        self.cfg.save()

    def options_snapshot(self) -> dict:
        """Read persistable control values without touching the ConfigStore.

        Feeds the app-level debounce path (spec 9); never raises on invalid
        input (falls back to the current default instead).
        """
        try:
            chunk_size = self._chunk_size_value()
        except ValueError:
            chunk_size = DEFAULT_CHUNK_S
        mode = self.chunk_mode_dropdown.value or CHUNK_MODE_AUTO
        return {
            "model": self._raw_model(),
            "epoch": (self.epoch_field.value or "best").strip() or "best",
            "post_filter": bool(self.postfilter_switch.value),
            "atten_lim_db": self._atten_value(),
            "device": self.device_dropdown.value or "Auto",
            "no_df_stage": bool(self.no_df_switch.value),
            "delay_compensation": bool(self.delay_switch.value),
            "chunk_mode": _CHUNK_MODE_TO_CONFIG.get(mode, "auto"),
            "chunk_size_s": chunk_size,
            "output_dir": self._output_dir_value(),
            "output_format": self.format_dropdown.value or "",
            "suffix_enabled": bool(self.suffix_switch.value),
            "log_level": self.log_level_dropdown.value or "INFO",
        }

    # ------------------------------------------------------------------ events

    def _on_event(self, event: AppEvent) -> None:
        """Update progress bars and the status line from bus payloads."""
        if event.type == "job_progress":
            try:
                progress = float(event.payload.get("progress") or 0.0)
            except (TypeError, ValueError):
                return
            self.global_progress.value = progress
            self.file_progress.value = self._file_fraction(event.payload.get("job_id"), progress)
            self.status_text.value = f"Enhancing - {progress * 100:.0f}%"
            self._refresh()
        elif event.type == "job_state":
            state = event.payload.get("state")
            error = event.payload.get("error")
            if not state:
                return
            if state == "failed" and error:
                self.status_text.value = f"Failed: {error}"
            else:
                self.status_text.value = str(state).capitalize()
            self._refresh()

    def _file_fraction(self, job_id, progress: float) -> float:
        """Progress inside the current file: global share minus finished files."""
        jobs = getattr(self.queue, "jobs", None) or ()
        job = next((job for job in jobs if job.id == job_id), None)
        if job is None or not job.files:
            return _clamp(progress)
        done = sum(1 for value in job.file_results.values() if value == "done")
        return _clamp(progress * len(job.files) - done)

    def _set_status(self, text: str) -> None:
        self.status_text.value = text
        self._refresh()

    def _refresh(self) -> None:
        """Push mutations to the client when running outside a Flet event.

        Bus callbacks are not control events, so Flet's automatic
        post-event update does not cover them; headless tests have no page.
        """
        try:
            page = self.status_text.page
        except RuntimeError:
            return  # controls not mounted yet (headless build)
        page.update()

    # ------------------------------------------------------- picker callbacks

    async def _on_add_files(self, e: ft.Event) -> None:
        # pick_files() returns the selection directly (a cancelled dialog
        # yields no files); on_result is reserved for client-side actions.
        files = await self.file_picker.pick_files(allow_multiple=True)
        paths = [file.path for file in (files or []) if file.path]
        if paths:
            self.file_list.add_paths(paths)

    async def _on_pick_dir(self, e: ft.Event) -> None:
        # Only hint an existing directory: native dialogs reject missing paths.
        current = Path(self._output_dir_value())
        directory = await self.dir_picker.get_directory_path(
            initial_directory=str(current) if current.is_dir() else None
        )
        if directory:
            self.output_dir_field.value = directory
            self._notify_options_changed()

    # -------------------------------------------------------- control handlers

    def _on_enhance(self, e: ft.Event) -> None:
        self.enhance_all()

    def _on_model_change(self, e: ft.Event) -> None:
        self.custom_model_field.visible = self.model_dropdown.value == CUSTOM_MODEL_LABEL
        self.validate()
        self._notify_options_changed()

    def _on_epoch_change(self, e: ft.Event) -> None:
        self.validate()
        self._notify_options_changed()

    def _on_chunk_mode_change(self, e: ft.Event) -> None:
        self.chunk_size_field.disabled = self.chunk_mode_dropdown.value == CHUNK_MODE_AUTO
        self.validate()
        self._notify_options_changed()

    def _on_chunk_size_change(self, e: ft.Event) -> None:
        self.validate()
        self._notify_options_changed()

    def _on_option_edited(self, e: ft.Event) -> None:
        """Generic option edit: hand the snapshot to the app for debounce-save."""
        self._notify_options_changed()

    def _notify_options_changed(self) -> None:
        if self._on_options_changed is not None:
            self._on_options_changed()

    # ---------------------------------------------------------------- helpers

    def _model_value(self) -> str:
        choice = self.model_dropdown.value or ""
        if choice != CUSTOM_MODEL_LABEL:
            return choice
        path = (self.custom_model_field.value or "").strip()
        resolve_model_dir(path)  # raises ModelError for an invalid custom path
        return path

    def _raw_model(self) -> str:
        """Model value without custom-path validation (for persistence)."""
        if self.model_dropdown.value == CUSTOM_MODEL_LABEL:
            return (self.custom_model_field.value or "").strip()
        return self.model_dropdown.value or ""

    def _epoch_value(self) -> str:
        epoch = (self.epoch_field.value or "").strip()
        if epoch in ("best", "latest") or epoch.isdigit():
            return epoch
        raise ValueError("Epoch must be 'best', 'latest' or a checkpoint number")

    def _atten_value(self) -> int | None:
        db = int(round(float(self.atten_slider.value or 0)))
        return None if db <= 0 else db

    def _chunk_size_value(self) -> int:
        mode = self.chunk_mode_dropdown.value or CHUNK_MODE_AUTO
        raw = (self.chunk_size_field.value or "").strip()
        try:
            seconds = int(raw) if raw else DEFAULT_CHUNK_S
        except ValueError:
            raise ValueError("Chunk size must be a whole number of seconds") from None
        if mode == CHUNK_MODE_AUTO:
            return seconds  # size is ignored by plan_chunks in auto mode
        if mode == CHUNK_MODE_PRESET:
            if seconds not in PRESET_CHUNK_SIZES:
                raise ValueError("Preset chunk sizes: 30, 60, 120 or 300 seconds")
            return seconds
        validate_chunk_size(seconds)  # Custom: 5..600 (Task 4)
        return seconds

    def _output_dir_value(self) -> str:
        return (self.output_dir_field.value or "").strip() or "./out"

    @staticmethod
    def _initial_atten(value) -> float:
        """Clamp a persisted attenuation limit onto the 0..60 slider."""
        if value is None:
            return 0
        try:
            return float(min(max(int(value), 0), 60))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _initial_choice(stored: str, options, fallback: str) -> str:
        """Clamp a persisted choice to the currently available options."""
        if stored in options:
            return stored
        return options[0] if options else fallback
