"""
Logging utilities for C2-VLM.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    log_file: Optional[Path] = None,
    log_level: int = logging.INFO,
    logger_name: str = "c2_vlm"
) -> logging.Logger:
    """
    Set up logger with console and file handlers.
    
    Args:
        log_file: Path to log file (optional)
        log_level: Logging level
        logger_name: Name of the logger
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file is provided)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger