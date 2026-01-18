import logging
from datetime import date
from logging.handlers import RotatingFileHandler
from pathlib import Path


LOG_DIR = Path("logs")
LOG_FORMAT = "{asctime} | {levelname} | {name} | {message}"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"log_{date.today()}.log"
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT, style="{")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(log_file, maxBytes=1_048_576, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger
