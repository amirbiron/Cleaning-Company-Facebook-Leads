import json
import logging
import os
import sys

LOG_LEVEL = os.environ.get("LOG_LEVEL", "DEBUG").upper()
LOG_FORMAT = os.environ.get("LOG_FORMAT", "").strip().lower()


class _JsonFormatter(logging.Formatter):
    """פורמט JSON לכלי מוניטורינג — שורה אחת לכל רשומה."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        if LOG_FORMAT == "json":
            handler.setFormatter(_JsonFormatter())
        else:
            handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            ))
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, LOG_LEVEL, logging.DEBUG))

    return logger
