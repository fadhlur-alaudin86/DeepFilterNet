"""Settings view placeholder (controls land in Task 11)."""

from __future__ import annotations

import flet as ft

from gui.core.config import ConfigStore
from gui.core.events import EventBus


class SettingsView:
    """Settings page: defaults, theme and runtime repair (Task 11).

    Constructor takes the event bus as well as the config store (ledger
    R6): Task 11 publishes runtime-repair progress events on it.
    """

    def __init__(self, cfg: ConfigStore, bus: EventBus) -> None:
        self.cfg = cfg
        self.bus = bus

    def build(self) -> ft.Control:
        return ft.Container(
            content=ft.Column(
                [
                    ft.Text("Settings", size=20, weight=ft.FontWeight.W_600),
                    ft.Text(
                        "Placeholder - settings controls arrive in Task 11.",
                        size=13,
                    ),
                ],
                spacing=8,
            ),
            expand=True,
        )
