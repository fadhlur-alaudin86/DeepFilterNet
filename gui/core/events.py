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

# enhance.log line layout (spec 10): one timestamped record per line.
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

# Loguru records are forwarded into stdlib logging under this logger name;
# they propagate to the rotating handler from setup_logging, which stays
# enhance.log's only writer (df passes log_file=None to init_df).
FORWARD_LOGGER_NAME = "df"

# loguru severity name -> stdlib level number for the file forward.
# TRACE(5) and SUCCESS(25) have no stdlib counterpart; they are kept as-is
# so the file preserves df's original severity ordering.
_LOGURU_TO_STD = {
    "TRACE": 5,
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "SUCCESS": 25,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


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
    and is unused in v1. Idempotent: a second call for the same file adds
    no handler (repeated app starts in one process must not duplicate lines).
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "enhance.log"
    root = logging.getLogger()
    for handler in root.handlers:
        if (
            isinstance(handler, logging.handlers.RotatingFileHandler)
            and Path(handler.baseFilename) == log_path
        ):
            root.setLevel(level)
            return log_path
    handler = logging.handlers.RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=3)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
    root.addHandler(handler)
    root.setLevel(level)
    return log_path


def setup_df_log_bridge(bus: EventBus, level: str) -> None:
    """Forward loguru records to the bus as "log" events and to enhance.log.

    The bus event drives the Log view console; the same record is forwarded
    into stdlib logging under FORWARD_LOGGER_NAME so it also lands in the
    rotating enhance.log handler (with timestamps) without a second file
    writer. Idempotent: re-installs a single sink, removing the previous
    one — required because df's init_logger calls logger.remove() on the
    first init_df(), silently deleting this sink (re-install after model
    init via InProcessBackend's on_model_ready hook).

    The sink id is exposed as setup_df_log_bridge.last_sink_id so it can be
    removed later via loguru.logger.remove(...).
    """

    def _sink(message) -> None:
        record = message.record
        name = str(record["level"].name).upper()
        text = str(record["message"]).rstrip()
        bus.publish(AppEvent("log", {"level": name, "text": text}))
        logging.getLogger(FORWARD_LOGGER_NAME).log(_LOGURU_TO_STD.get(name, logging.INFO), text)

    old_sink_id = getattr(setup_df_log_bridge, "last_sink_id", None)
    if old_sink_id is not None:
        try:
            logger.remove(old_sink_id)
        except ValueError:
            pass  # already removed (e.g. by df's init_logger)
    sink_id = logger.add(_sink, level=level)
    setup_df_log_bridge.last_sink_id = sink_id
