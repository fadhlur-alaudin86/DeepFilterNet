"""Settings view: defaults, theme and runtime repair (Task 11).

Owns the default output directory picker, the theme mode dropdown
(System/Light/Dark), a read-only config file path display and the
"Runtime repair" action, which runs ``ensure_runtime`` off the UI loop
and forwards its phase progress onto the event bus as ``"runtime"``
events (ledger R6: the constructor takes the bus for exactly this).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import flet as ft

from gui.core.config import ConfigStore
from gui.core.dependency import current_env_ready, ensure_runtime, runtime_dir
from gui.core.events import AppEvent, EventBus

# ConfigStore values are uppercase; dropdown labels are display strings.
_THEME_OPTIONS = (
    ("SYSTEM", "System"),
    ("LIGHT", "Light"),
    ("DARK", "Dark"),
)
_THEME_MODES = {
    "SYSTEM": ft.ThemeMode.SYSTEM,
    "LIGHT": ft.ThemeMode.LIGHT,
    "DARK": ft.ThemeMode.DARK,
}
DEFAULT_OUTPUT_DIR = "./out"


class SettingsView:
    """Settings page: defaults, theme and runtime repair.

    Long repair work runs through ``asyncio.to_thread`` so the UI loop
    never blocks (spec 3.4); progress hops back via ``bus.publish``.
    """

    def __init__(self, cfg: ConfigStore, bus: EventBus) -> None:
        self.cfg = cfg
        self.bus = bus
        initial = cfg.load()

        # One picker, strongly referenced so flet's post-event service GC
        # does not unregister it (same pattern as EnhanceView).
        self.dir_picker = ft.FilePicker()
        self._pickers = (self.dir_picker,)

        self.output_dir_field = ft.TextField(
            label="Default output directory",
            value=str(initial.get("output_dir") or DEFAULT_OUTPUT_DIR),
            expand=True,
        )
        self.browse_button = ft.FilledButton("Browse...", on_click=self._on_pick_dir)
        self.theme_dropdown = ft.Dropdown(
            label="Theme",
            value=self._initial_theme(initial.get("theme_mode")),
            options=[ft.DropdownOption(key=key, text=label) for key, label in _THEME_OPTIONS],
            on_select=self._on_theme_select,
        )
        self.config_path_field = ft.TextField(
            label="Config file",
            value=str(cfg.path),
            read_only=True,
            expand=True,
        )
        self.runtime_dir_field = ft.TextField(
            label="Runtime directory",
            value=str(runtime_dir()),
            read_only=True,
            expand=True,
        )
        self.runtime_status = ft.Text(self._env_status(), size=12)
        self.repair_status = ft.Text("", size=12)
        self.repair_button = ft.FilledButton("Runtime repair", on_click=self._on_repair)
        self.save_button = ft.FilledButton("Save", on_click=self._on_save)

    # ----------------------------------------------------------------- layout

    def build(self) -> ft.Control:
        """Compose the Settings page (spec 4)."""
        return ft.Container(
            content=ft.Column(
                [
                    ft.Row(
                        [
                            ft.Text("Settings", size=20, weight=ft.FontWeight.W_600),
                            self.save_button,
                        ],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    ),
                    ft.Text("Defaults", size=14, weight=ft.FontWeight.W_600),
                    ft.Row(
                        [self.output_dir_field, self.browse_button],
                        spacing=12,
                        wrap=True,
                    ),
                    self.theme_dropdown,
                    ft.Text("Application", size=14, weight=ft.FontWeight.W_600),
                    self.config_path_field,
                    ft.Text("Runtime", size=14, weight=ft.FontWeight.W_600),
                    self.runtime_dir_field,
                    ft.Row(
                        [self.repair_button, self.repair_status],
                        spacing=12,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    self.runtime_status,
                ],
                spacing=8,
                expand=True,
            ),
            expand=True,
        )

    # ------------------------------------------------------------- persistence

    def save(self) -> None:
        """Persist the output directory and theme mode to the ConfigStore."""
        output_dir = (self.output_dir_field.value or "").strip() or DEFAULT_OUTPUT_DIR
        self.cfg.set("output_dir", output_dir)
        self.cfg.set("theme_mode", str(self.theme_dropdown.value or "SYSTEM"))
        self.cfg.save()

    # ------------------------------------------------------- picker callbacks

    async def _on_pick_dir(self, e: ft.Event) -> None:
        # get_directory_path() returns the selection directly (flet 1.0
        # return-value pattern); None means the dialog was cancelled.
        current = Path((self.output_dir_field.value or "").strip() or ".")
        directory = await self.dir_picker.get_directory_path(
            initial_directory=str(current) if current.is_dir() else None
        )
        if directory:
            self.output_dir_field.value = directory
            self.save()

    # ------------------------------------------------------- control handlers

    def _on_theme_select(self, e: ft.Event) -> None:
        # Persist immediately: a theme the user just picked must survive
        # the debounced app-level save window.
        self.save()
        self._apply_theme()

    def _on_save(self, e: ft.Event) -> None:
        self.save()
        self._apply_theme()

    async def _on_repair(self, e: ft.Event) -> None:
        self.repair_button.disabled = True
        self.repair_status.value = "Repairing runtime..."
        self._refresh()
        try:
            # Heavy pip/venv work must not block the UI loop (spec 3.4).
            await asyncio.to_thread(ensure_runtime, self._report_progress)
        except Exception as exc:
            self.bus.publish(AppEvent("runtime", {"phase": "error", "error": str(exc)}))
            self.repair_status.value = f"Repair failed: {exc}"
        else:
            self.repair_status.value = "Runtime repair complete"
        finally:
            self.repair_button.disabled = False
            self._refresh()

    # ---------------------------------------------------------------- helpers

    def _report_progress(self, phase: str, value: float) -> None:
        """ensure_runtime progress callback: forward phases to the bus."""
        self.bus.publish(AppEvent("runtime", {"phase": phase, "progress": value}))

    def _apply_theme(self) -> None:
        """Apply the selected theme to the live page when it is mounted."""
        try:
            page = self.theme_dropdown.page
        except RuntimeError:
            return  # controls not mounted yet (headless build)
        page.theme_mode = _THEME_MODES.get(
            str(self.theme_dropdown.value or ""), ft.ThemeMode.SYSTEM
        )

    def _refresh(self) -> None:
        """Push mutations to the client when running outside a Flet event."""
        try:
            page = self.repair_button.page
        except RuntimeError:
            return  # controls not mounted yet (headless build)
        page.update()

    @staticmethod
    def _initial_theme(stored) -> str:
        """Clamp a persisted theme mode onto the System/Light/Dark choices."""
        value = str(stored or "SYSTEM").upper()
        return value if value in _THEME_MODES else "SYSTEM"

    @staticmethod
    def _env_status() -> str:
        """Human-readable readiness of the current Python environment."""
        if current_env_ready():
            return "Current environment: ready"
        return "Current environment: missing runtime dependencies"
