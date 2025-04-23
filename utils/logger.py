"""
Logger module for ATPT.

This module provides a centralized logging system for the ATPT project.
It configures loggers with both file and console handlers, allowing for
comprehensive logging throughout the application.
"""

import os
import logging
import sys
from datetime import datetime

def setup_logger(name, log_dir, level=logging.INFO):
    """
    Set up a logger with file and console handlers.
    
    Args:
        name (str): Name of the logger
        log_dir (str): Directory to save log files
        level (int): Logging level (default: logging.INFO)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Don't propagate to root logger
    logger.propagate = False
    
    # Clear existing handlers if any
    if logger.handlers:
        logger.handlers.clear()
    
    # Create log directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_file