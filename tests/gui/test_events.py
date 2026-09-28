import logging
import logging.handlers
import threading
from unittest import mock

from loguru import logger

from gui.core.events import AppEvent, EventBus, setup_df_log_bridge, setup_logging


def test_publish_dispatches_to_subscribers():
    bus = EventBus()
    first: list[AppEvent] = []
    second: list[AppEvent] = []
    bus.subscribe(first.append)
    bus.subscribe(second.append)
    event = AppEvent("job_state", {"state": "running"})
    bus.publish(event)
    assert first == [event]
    assert second == [event]


def test_publish_without_loop_is_synchronous():
    bus = EventBus()
    seen: list[AppEvent] = []
    expected = AppEvent("log", {"level": "INFO", "text": "inline"})
    bus.subscribe(seen.append)
    bus.publish(expected)
    # No bound loop: the subscriber must already have run when publish returns.
    assert seen == [expected]


def test_publish_with_loop_schedules_coroutine():
    bus = EventBus()
    loop = mock.Mock()
    bus.bind_loop(loop)

    def worker():
        bus.publish(AppEvent("job_progress", {"pct": 50}))

    # Real asyncio loops have no run_coroutine_threadsafe method, so the
    # scheduling entry point itself is mocked to observe the exact call.
    with mock.patch("asyncio.run_coroutine_threadsafe") as schedule:
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert schedule.call_count == 1
    coro, bound_loop = schedule.call_args[0]
    assert bound_loop is loop
    coro.close()  # the mock never runs it; close to avoid a RuntimeWarning


def test_publish_from_multiple_threads():
    bus = EventBus()
    received: list[AppEvent] = []
    errors: list[Exception] = []
    lock = threading.Lock()

    def record(event: AppEvent) -> None:
        with lock:
            received.append(event)

    def worker(worker_id: int) -> None:
        try:
            for seq in range(50):
                bus.publish(AppEvent("job_progress", {"worker": worker_id, "seq": seq}))
        except Exception as exc:  # pragma: no cover - only on failure
            errors.append(exc)

    bus.subscribe(record)
    threads = [threading.Thread(target=worker, args=(wid,)) for wid in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(received) == 8 * 50
    assert len({(e.payload["worker"], e.payload["seq"]) for e in received}) == 8 * 50


def test_setup_logging_creates_rotating_file(tmp_path):
    root = logging.getLogger()
    saved_level = root.level
    saved_handlers = list(root.handlers)
    try:
        log_path = setup_logging(EventBus(), "INFO", tmp_path)
        assert log_path == tmp_path / "enhance.log"
        root.info("rotating file marker")
        for handler in root.handlers:
            handler.flush()
        assert log_path.exists()
        added = [handler for handler in root.handlers if handler not in saved_handlers]
        assert any(isinstance(handler, logging.handlers.RotatingFileHandler) for handler in added)
        assert "rotating file marker" in log_path.read_text(encoding="utf-8")
    finally:
        for handler in list(root.handlers):
            if handler not in saved_handlers:
                root.removeHandler(handler)
                handler.close()
        root.setLevel(saved_level)


def test_log_bridge_forwards_loguru_to_bus():
    bus = EventBus()
    received: list[AppEvent] = []
    bus.subscribe(received.append)
    setup_df_log_bridge(bus, "INFO")
    sink_id = setup_df_log_bridge.last_sink_id
    try:
        logger.info("hello")
        log_events = [event for event in received if event.type == "log"]
        hello = next(event for event in log_events if "hello" in event.payload["text"])
        assert hello.payload["level"] == "INFO"
    finally:
        logger.remove(sink_id)


def test_setup_logging_formatter_has_timestamps(tmp_path):
    root = logging.getLogger()
    saved_level = root.level
    saved_handlers = list(root.handlers)
    try:
        log_path = setup_logging(EventBus(), "INFO", tmp_path)
        (tmp_path / "enhance.log").write_text("", encoding="utf-8")
        root.warning("timestamp-marker")
        for handler in root.handlers:
            handler.flush()
        line = log_path.read_text(encoding="utf-8").strip().splitlines()[-1]
        assert "timestamp-marker" in line
        # "%Y-%m-%d %H:%M:%S WARNING df:" — date, time, level, logger name.
        assert line[:4].isdigit() and "WARNING" in line
    finally:
        for handler in list(root.handlers):
            if handler not in saved_handlers:
                root.removeHandler(handler)
                handler.close()
        root.setLevel(saved_level)


def test_setup_logging_idempotent_for_same_dir(tmp_path):
    root = logging.getLogger()
    saved_level = root.level
    saved_handlers = list(root.handlers)
    try:
        setup_logging(EventBus(), "INFO", tmp_path)
        setup_logging(EventBus(), "INFO", tmp_path)
        added = [
            handler
            for handler in root.handlers
            if handler not in saved_handlers
            and isinstance(handler, logging.handlers.RotatingFileHandler)
        ]
        assert len(added) == 1
    finally:
        for handler in list(root.handlers):
            if handler not in saved_handlers:
                root.removeHandler(handler)
                handler.close()
        root.setLevel(saved_level)


def test_log_bridge_reinstall_keeps_single_sink():
    bus = EventBus()
    received: list[AppEvent] = []
    bus.subscribe(received.append)
    setup_df_log_bridge(bus, "INFO")
    first_sink_id = setup_df_log_bridge.last_sink_id
    try:
        setup_df_log_bridge(bus, "INFO")  # e.g. after df wiped sinks
        assert setup_df_log_bridge.last_sink_id != first_sink_id
        logger.info("single-sink-marker")
        matches = [
            event
            for event in received
            if event.type == "log" and "single-sink-marker" in event.payload["text"]
        ]
        assert len(matches) == 1
    finally:
        logger.remove(setup_df_log_bridge.last_sink_id)
