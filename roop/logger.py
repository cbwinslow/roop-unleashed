import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


LOG_FILE = Path("roop.log")


def setup_logger(log_level: str = "INFO", max_bytes: int = 5 * 1024 * 1024, backup_count: int = 5) -> logging.Logger:
    """Configure and return application logger with rotation."""
    logger = logging.getLogger("roop")
    if logger.handlers:
        return logger

    logger.setLevel(log_level.upper())
    handler = RotatingFileHandler(LOG_FILE, maxBytes=max_bytes, backupCount=backup_count)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

