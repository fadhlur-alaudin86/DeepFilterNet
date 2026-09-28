"""DeepFilterNet Flet application skeleton: navigation, theme, persistence.

`DeepFilterApp` applies persisted theme/window state to the page, binds the
`EventBus` to the page's asyncio loop (UI updates only from that loop), and
composes the root layout: NavigationRail + swappable view container + status
bar. `main(page)` is the `ft.run` target; it wires the shared services and
mounts the skeleton. Full view implementations land in Tasks 10-11.
"""

from __future__ import annotations

import asyncio
from typing import Any

import flet as ft

from gui.core.backend import InProcessBackend
from gui.core.config import ConfigStore
from gui.core.events import AppEvent, EventBus
from gui.core.jobs import JobQueue
from gui.ui.pages import EnhanceView, LogView, QueueView, SettingsView

# Seed for page.theme.color_scheme_seed (Material 3 color generation).
COLOR_SCHEME_SEED = "#3f51b5"

# Debounce window before persisted UI state hits the ConfigStore (spec 9).
SAVE_DEBOUNCE_S = 1.0

# Rail destination keys and labels; order defines the selected-index mapping.
VIEW_KEYS = ("enhance", "queue", "log", "settings")
VIEW_LABELS = ("Enhance", "Queue", "Log", "Settings")
_VIEW_ICONS = (
    (ft.Icons.ADD_CIRCLE_OUTLINE, ft.Icons.ADD_CIRCLE),
    (ft.Icons.LIST_ALT, ft.Icons.LIST),
    (ft.Icons.SUBJECT, ft.Icons.NOTES),
    (ft.Icons.SETTINGS_OUTLINED, ft.Icons.SETTINGS),
)

_THEME_MODES = {
    "SYSTEM": ft.ThemeMode.SYSTEM,
    "LIGHT": ft.ThemeMode.LIGHT,
    "DARK": ft.ThemeMode.DARK,
}


class DeepFilterApp:
    """Root UI: navigation rail, swappable views, status bar, persistence hooks."""

    def __init__(self, page: ft.Page, cfg: ConfigStore, bus: EventBus, queue: JobQueue) -> None:
        self.page = page
        self.cfg = cfg
        self.bus = bus
        self.queue = queue

        # Latest bus payloads, keyed for later views (Task 7 payload keys are
        # merged verbatim: job_state {job_id, state, error},
        # job_progress {job_id, progress}).
        self.jobs: dict[str, dict[str, Any]] = {}
        self.runtime: dict[str, Any] = {}

        self.views = [
            EnhanceView(cfg, queue, bus),
            QueueView(queue, bus),
            LogView(bus),
            SettingsView(cfg, bus),
        ]
        builders = [view.build for view in self.views]

        self._apply_theme()
        self._apply_window_geometry()

        # Bind the bus before subscribing so every publish dispatches on the
        # page loop; the dispatcher then refreshes the page only from there.
        self._loop = page.loop
        bus.bind_loop(self._loop)
        bus.subscribe(self._on_event)

        self._pending: dict[str, Any] = {}
        self._save_handle: asyncio.TimerHandle | None = None
        self._prev_on_resize = page.on_resize
        page.on_resize = self._on_page_resize
        self._prev_window_on_event = page.window.on_event
        page.window.on_event = self._on_window_event

        last_view = cfg.get("last_view")
        index = VIEW_KEYS.index(last_view) if last_view in VIEW_KEYS else 0
        self.rail = ft.NavigationRail(
            selected_index=index,
            label_type=ft.NavigationRailLabelType.ALL,
            destinations=[
                ft.NavigationRailDestination(
                    label=label,
                    icon=icons[0],
                    selected_icon=icons[1],
                )
                for label, icons in zip(VIEW_LABELS, _VIEW_ICONS)
            ],
            on_change=self._on_rail_change,
        )
        self.view_container = ft.Container(
            content=builders[index](),
            expand=True,
            padding=ft.Padding.symmetric(horizontal=20, vertical=16),
        )
        self.status_text = ft.Text("Ready", size=12)
        status_bar = ft.Container(
            content=ft.Row([self.status_text], spacing=8),
            padding=ft.Padding.symmetric(horizontal=12, vertical=6),
        )
        self.root = ft.Column(
            [
                ft.Row(
                    [self.rail, ft.VerticalDivider(width=1), self.view_container],
                    expand=True,
                    spacing=0,
                ),
                status_bar,
            ],
            expand=True,
            spacing=0,
        )

    # ----------------------------------------------------------------- layout

    def build(self) -> ft.Control:
        """Return the root control: NavigationRail + views + status bar."""
        return self.root

    def show(self, index: int) -> None:
        """Swap the visible view and persist the selection (debounced)."""
        if not 0 <= index < len(VIEW_KEYS):
            return
        self.rail.selected_index = index
        self.view_container.content = self.views[index].build()
        self._pending["last_view"] = VIEW_KEYS[index]
        self._schedule_save()
        self.page.update()

    # ------------------------------------------------------------------ theme

    def _apply_theme(self) -> None:
        """Set page.theme_mode from config and the color scheme seed."""
        raw = str(self.cfg.get("theme_mode") or "SYSTEM").upper()
        self.page.theme_mode = _THEME_MODES.get(raw, ft.ThemeMode.SYSTEM)
        if self.page.theme is None:
            self.page.theme = ft.Theme(color_scheme_seed=COLOR_SCHEME_SEED)
        else:
            self.page.theme.color_scheme_seed = COLOR_SCHEME_SEED

    def _apply_window_geometry(self) -> None:
        """Restore persisted window size (spec 9)."""
        width = self.cfg.get("window_width")
        height = self.cfg.get("window_height")
        if isinstance(width, (int, float)):
            self.page.window.width = int(width)
        if isinstance(height, (int, float)):
            self.page.window.height = int(height)

    # --------------------------------------------------------------- events

    def _on_event(self, event: AppEvent) -> None:
        """Apply a bus payload, then refresh the page from the loop thread."""
        if event.type in ("job_state", "job_progress"):
            job_id = event.payload.get("job_id")
            if job_id is not None:
                self.jobs.setdefault(job_id, {}).update(event.payload)
        elif event.type == "runtime":
            self.runtime.update(event.payload)
        self._request_page_update()

    def _request_page_update(self) -> None:
        """Call page.update() on the page loop thread, wherever we were called."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            on_loop_thread = asyncio.get_running_loop() is loop
        except RuntimeError:
            on_loop_thread = False
        if on_loop_thread:
            self.page.update()
        else:
            loop.call_soon_threadsafe(self.page.update)

    # ----------------------------------------------------------- persistence

    def _on_rail_change(self, e: ft.Event[ft.NavigationRail]) -> None:
        """Handle rail selection and persist the chosen view.

        Flet reports the new index as event data; fall back to the control
        property when the payload carries none.
        """
        index = e.control.selected_index
        if isinstance(e.data, int):
            index = e.data
        elif isinstance(e.data, str) and e.data.lstrip("-").isdigit():
            index = int(e.data)
        self.show(index)

    def _on_page_resize(self, e: Any) -> None:
        """Persist window size on resize (debounced 1 s; spec 9)."""
        width = getattr(e, "width", None)
        height = getattr(e, "height", None)
        if isinstance(width, (int, float)) and isinstance(height, (int, float)):
            self._pending["window_width"] = int(width)
            self._pending["window_height"] = int(height)
            self._schedule_save()
        if self._prev_on_resize is not None:
            self._prev_on_resize(e)

    def _on_window_event(self, e: Any) -> None:
        """Flush pending persistence when the window closes (spec 9)."""
        if self._prev_window_on_event is not None:
            self._prev_window_on_event(e)
        if e.type == ft.WindowEventType.CLOSE:
            window = self.page.window
            if "window_width" not in self._pending and isinstance(window.width, (int, float)):
                self._pending["window_width"] = int(window.width)
            if "window_height" not in self._pending and isinstance(window.height, (int, float)):
                self._pending["window_height"] = int(window.height)
            self._flush()

    def _schedule_save(self) -> None:
        """Debounce persistence by 1 s on the page loop (spec 9)."""
        loop = self._loop
        if loop is None or loop.is_closed():
            self._flush()  # no loop to debounce on: save immediately
            return
        if self._save_handle is not None:
            self._save_handle.cancel()
        self._save_handle = loop.call_later(SAVE_DEBOUNCE_S, self._flush_scheduled)

    def _flush_scheduled(self) -> None:
        self._save_handle = None
        self._flush()

    def _flush(self) -> None:
        """Write pending UI state to the ConfigStore and clear it."""
        if self._save_handle is not None:
            self._save_handle.cancel()
            self._save_handle = None
        pending, self._pending = self._pending, {}
        for key, value in pending.items():
            self.cfg.set(key, value)


def main(page: ft.Page) -> None:
    """Flet entry point: build services, mount the skeleton (queue not started)."""
    cfg = ConfigStore()
    bus = EventBus()
    queue = JobQueue(InProcessBackend(), bus)
    app = DeepFilterApp(page, cfg, bus, queue)
    page.title = "DeepFilterNet"
    page.add(app.build())
    page.update()
