"""Enhance view placeholder (full layout lands in Task 10)."""

from __future__ import annotations

import flet as ft

from gui.core.config import ConfigStore
from gui.core.events import EventBus
from gui.core.jobs import JobQueue


class EnhanceView:
    """Enhance page: file list, enhancement options and progress (Task 10)."""

    def __init__(self, cfg: ConfigStore, queue: JobQueue, bus: EventBus) -> None:
        self.cfg = cfg
        self.queue = queue
        self.bus = bus

    def build(self) -> ft.Control:
        return ft.Container(
            content=ft.Column(
                [
                    ft.Text("Enhance", size=20, weight=ft.FontWeight.W_600),
                    ft.Text(
                        "Placeholder - file list and options arrive in Task 10.",
                        size=13,
                    ),
                ],
                spacing=8,
            ),
            expand=True,
        )
