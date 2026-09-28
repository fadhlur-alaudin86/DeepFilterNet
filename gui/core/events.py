"""Thread-safe event bus and logging bridges for the GUI process."""

import asyncio
import inspect
import logging
import logging.handlers
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from loguru import logger


@dataclass(frozen=True)
class AppEvent:
    """Single application event.

    type: one of "job_state" | "job_progress" | "log" | "runtime".
    payload: event-specific data.
    """

    type: str
    payload: dict


class EventBus:
    """Publish/subscribe bus; dispatches callbacks inline or on a bound loop."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: list[Callable[[AppEvent], None]] = []
        self._loop: asyncio.AbstractEventLoop | None = None

    def subscribe(self, fn: Callable[[AppEvent], None]) -> None:
        """Register a callback invoked for every published event."""
        with self._lock:
            self._subscribers.append(fn)

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Bind an event loop; subsequent publishes run callbacks on it."""
        with self._lock:
            self._loop = loop

    def publish(self, event: AppEvent) -> None:
        """Deliver an event to all subscribers; safe to call from any thread."""
        with self._lock:
            loop = self._loop
        if loop is None:
            self._dispatch_sync(event)
        else:
            # Schedule on the bound loop; safe to call from any thread.
            asyncio.run_coroutine_threadsafe(self._dispatch(event), loop)

    def _subscribers_snapshot(self) -> list[Callable[[AppEvent], None]]:
        with self._lock:
            return list(self._subscribers)

    def _dispatch_sync(self, event: AppEvent) -> None:
        for fn in self._subscribers_snapshot():
            fn(event)

    async def _dispatch(self, event: AppEvent) -> None:
        """Run subscribers on the bound loop; sync callables are called directly."""
        for fn in self._subscribers_snapshot():
            result = fn(event)
            if inspect.isawaitable(result):
                await result


def setup_logging(bus: EventBus, level: str, log_dir: Path) -> Path:
    """Attach a rotating enhancement log handler to the root logger.

    Returns log_dir / "enhance.log". The bus parameter is part of the
    interface (log forwarding to the bus is owned by setup_df_log_bridge)
    and is unused in v1.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "enhance.log"
    handler = logging.handlers.RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=3)
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(level)
    return log_path


def setup_df_log_bridge(bus: EventBus, level: str) -> None:
    """Forward loguru records to the bus as "log" events.

    The sink id is exposed as setup_df_log_bridge.last_sink_id so it can be
    removed later via loguru.logger.remove(...).
    """
    sink_id = logger.add(
        lambda m: bus.publish(
            AppEvent("log", {"level": m.record["level"].name, "text": str(m).rstrip()})
        ),
        level=level,
    )
    setup_df_log_bridge.last_sink_id = sink_id
