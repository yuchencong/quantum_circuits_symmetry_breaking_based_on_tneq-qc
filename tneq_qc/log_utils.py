"""
Logging utilities for MPI-based applications.

This module provides colored logging formatters and logger setup functions
using the LOG_FORMATER color definitions from callbacks.
"""

import logging
from .callbacks import LOG_FORMATER


class ColoredFormatter(logging.Formatter):
    """
    Colored log formatter using LOG_FORMATER color definitions.
    
    This formatter applies different colors to different log levels:
    - DEBUG: Cyan
    - INFO: Green
    - WARNING: Yellow
    - ERROR: Red
    - CRITICAL: Red background
    """
    
    # Define color mapping for different log levels
    LEVEL_COLORS = {
        'DEBUG': LOG_FORMATER.AZURE_F,      # Cyan - debug info
        'INFO': LOG_FORMATER.GREEN_F,       # Green - general info
        'WARNING': LOG_FORMATER.YELLOW_F,   # Yellow - warnings
        'ERROR': LOG_FORMATER.RED_F,        # Red - errors
        'CRITICAL': LOG_FORMATER.RED_B,     # Red background - critical errors
    }
    
    def format(self, record):
        """
        Format the log record with appropriate colors.
        
        Args:
            record: LogRecord instance
            
        Returns:
            Formatted and colored log message string
        """
        # Get the original formatted message
        log_message = super().format(record)
        
        # Get the color format for this log level
        color_format = self.LEVEL_COLORS.get(record.levelname, "{content}")
        
        # Apply color (keeping timestamp and level uncolored, only color the message)
        # Split into: timestamp, level, message
        parts = log_message.split(' - ', 2)
        if len(parts) == 3:
            timestamp, level, message = parts
            colored_message = color_format.format(content=message)
            return f"{timestamp} - {level} - {colored_message}"
        else:
            # If format doesn't match expected, color the whole message
            return color_format.format(content=log_message)


def setup_colored_logger(name, rank, level=logging.INFO):
    """
    Create a logger with colored output for MPI applications.
    
    Args:
        name: Logger name
        rank: MPI rank (used in log prefix)
        level: Logging level (default: logging.INFO)
    
    Returns:
        Configured logger instance with colored formatting
        
    Example:
        >>> from mpi4py import MPI
        >>> comm = MPI.COMM_WORLD
        >>> rank = comm.Get_rank()
        >>> logger = setup_colored_logger("MyApp", rank, logging.DEBUG)
        >>> logger.info("This is a green message")
        >>> logger.error("This is a red message")
    """
    logger = logging.getLogger(f"{name}_rank{rank}")
    logger.setLevel(level)
    logger.handlers.clear()  # Remove existing handlers
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Use colored formatter
    formatter = ColoredFormatter(
        fmt=f'[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    return logger


def setup_simple_logger(name, rank, level=logging.INFO):
    """
    Create a simple logger without colors for MPI applications.
    
    Useful for environments that don't support ANSI color codes.
    
    Args:
        name: Logger name
        rank: MPI rank (used in log prefix)
        level: Logging level (default: logging.INFO)
    
    Returns:
        Configured logger instance with simple formatting
    """
    logger = logging.getLogger(f"{name}_rank{rank}")
    logger.setLevel(level)
    logger.handlers.clear()  # Remove existing handlers
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Use simple formatter
    formatter = logging.Formatter(
        f'[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    return logger
