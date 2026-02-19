import atexit
import logging
import os
import queue
from logging.handlers import QueueHandler, QueueListener
from typing import Optional

_listener: Optional[QueueListener] = None


def configure_async_logging(service_name: str, level: str = "INFO") -> logging.Logger:
    """Configure non-blocking logging via QueueHandler/QueueListener.

    Safe to call multiple times; it only initializes once per process.
    """
    global _listener

    logger = logging.getLogger(service_name)
    log_level = getattr(logging, os.getenv("LOG_LEVEL", level).upper(), logging.INFO)

    if _listener is None:
        log_queue: queue.Queue = queue.Queue(-1)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        queue_handler = QueueHandler(log_queue)

        root = logging.getLogger()
        root.handlers.clear()
        root.setLevel(log_level)
        root.addHandler(queue_handler)

        _listener = QueueListener(log_queue, stream_handler, respect_handler_level=True)
        _listener.start()

        def _stop_listener():
            if _listener is not None:
                _listener.stop()

        atexit.register(_stop_listener)

    logger.setLevel(log_level)
    return logger
