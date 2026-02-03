"""
Logging setup for ACO Portfolio Optimizer.
Configures console and file logging with rotation.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .config import LoggingConfig


def setup_logging(config: LoggingConfig) -> logging.Logger:
    """
    Set up logging with console and file handlers.
    
    Args:
        config: Logging configuration
    
    Returns:
        Root logger for the application
    """
    # Create logs directory if needed
    log_file = Path(config.file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Get numeric level
    level = getattr(logging, config.level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(config.format)
    
    # Get root logger
    logger = logging.getLogger('aco_optimizer')
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        config.file,
        maxBytes=config.max_bytes,
        backupCount=config.backup_count
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger."""
    return logging.getLogger(f'aco_optimizer.{name}')
