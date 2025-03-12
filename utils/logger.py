#!/usr/bin/env python3
"""
Logging Utilities for Defect Detection System
Provides centralized logging configuration with rotation and multiple output formats.
"""

import os
import sys
import time
import logging
import traceback
from logging.handlers import RotatingFileHandler
from datetime import datetime

# Import centralized configuration
from config import config

# Configure logging format
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
CONSOLE_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

# Global dictionary to store logger instances
loggers = {}

def setup_logger(name, log_file=None, level=None, console=True, format_string=None):
    """
    Configure and return a logger with file and optional console handlers.
    
    Args:
        name: Logger name (typically module name)
        log_file: Path to log file (defaults to config.PATHS.LOG_FILE)
        level: Logging level (defaults to config.LOGGING.LEVEL)
        console: Whether to add a console handler
        format_string: Custom format string for log messages
        
    Returns:
        Configured logger instance
    """
    # Return existing logger if already configured
    if name in loggers:
        return loggers[name]
    
    # Set defaults from config
    if log_file is None:
        log_file = config.PATHS["LOG_FILE"]
    
    # Map string level to logging constant
    if level is None:
        level_name = config.LOGGING.get("LEVEL", "INFO")
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        level = level_map.get(level_name.upper(), logging.INFO)
    
    # Use default format if not specified
    if format_string is None:
        format_string = DEFAULT_FORMAT
    
    # Create logger
    logger = logging.getLogger(name)
    
    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter(format_string)
        
        # Ensure log directory exists
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            # Create file handler with rotation
            max_size = config.LOGGING.get("MAX_LOG_SIZE", 5 * 1024 * 1024)  # Default: 5MB
            backup_count = config.LOGGING.get("BACKUP_COUNT", 5)
            
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_size,
                backupCount=backup_count
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        # Add console handler if requested
        if console:
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(CONSOLE_FORMAT)
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
    
    # Store logger in dictionary
    loggers[name] = logger
    
    return logger

def get_logger(name):
    """
    Get an existing logger or create a new one.
    
    Args:
        name: Logger name (typically module name)
        
    Returns:
        Logger instance
    """
    if name in loggers:
        return loggers[name]
    else:
        return setup_logger(name)

class LogCapture:
    """
    Context manager for capturing log output to a string buffer.
    Useful for including logs in error reports or displaying in UI.
    """
    
    def __init__(self, logger_name, level=logging.INFO):
        """
        Initialize the log capture.
        
        Args:
            logger_name: Name of logger to capture
            level: Minimum log level to capture
        """
        self.logger_name = logger_name
        self.level = level
        self.logger = logging.getLogger(logger_name)
        self.log_output = []
        
        # StringHandler for capturing log output
        class StringHandler(logging.Handler):
            def __init__(self, log_list):
                super().__init__()
                self.log_list = log_list
            
            def emit(self, record):
                self.log_list.append(self.format(record))
        
        self.string_handler = StringHandler(self.log_output)
        self.string_handler.setLevel(self.level)
        self.string_handler.setFormatter(logging.Formatter(DEFAULT_FORMAT))
    
    def __enter__(self):
        """Add the string handler to the logger"""
        self.logger.addHandler(self.string_handler)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Remove the string handler from the logger"""
        self.logger.removeHandler(self.string_handler)
    
    def get_logs(self):
        """Get the captured log output as a string"""
        return "\n".join(self.log_output)

def log_exception(logger, message="An exception occurred", exc_info=None, level=logging.ERROR):
    """
    Log an exception with full traceback.
    
    Args:
        logger: Logger instance
        message: Message to log with the exception
        exc_info: Exception info tuple (from sys.exc_info()) or exception instance
        level: Log level for the message
    """
    if exc_info is None:
        exc_info = sys.exc_info()
    
    if isinstance(exc_info, Exception):
        # If exc_info is an exception instance, convert to exc_info tuple
        exc_type = type(exc_info)
        exc_value = exc_info
        exc_traceback = exc_info.__traceback__
        exc_info = (exc_type, exc_value, exc_traceback)
    
    # Extract exception details
    exc_type, exc_value, exc_traceback = exc_info
    if exc_type is not None:
        tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        logger.log(level, f"{message}: {exc_value}\n{tb_str}")
    else:
        logger.log(level, message)

def log_startup_info(logger):
    """
    Log useful system information on application startup.
    
    Args:
        logger: Logger instance
    """
    logger.info("=" * 80)
    logger.info("APPLICATION STARTING")
    logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {sys.platform}")
    
    # Log configuration info
    logger.info(f"Base directory: {config.BASE_DIR}")
    logger.info(f"Configuration loaded from: {config.__file__}")
    
    # Log hardware info if available
    try:
        import platform
        import psutil
        
        logger.info(f"Machine: {platform.machine()}")
        logger.info(f"Processor: {platform.processor()}")
        
        # Memory info
        memory = psutil.virtual_memory()
        logger.info(f"Memory: Total={memory.total/(1024**3):.1f}GB, Available={memory.available/(1024**3):.1f}GB")
        
        # Disk info
        disk = psutil.disk_usage('/')
        logger.info(f"Disk: Total={disk.total/(1024**3):.1f}GB, Free={disk.free/(1024**3):.1f}GB")
    except ImportError:
        logger.info("System information unavailable (psutil package not installed)")
    
    logger.info("=" * 80)

class PerformanceTimer:
    """
    Context manager for timing operations and logging performance metrics.
    """
    
    def __init__(self, logger, operation_name, log_level=logging.DEBUG):
        """
        Initialize the performance timer.
        
        Args:
            logger: Logger instance
            operation_name: Name of the operation being timed
            log_level: Log level for the timing message
        """
        self.logger = logger
        self.operation_name = operation_name
        self.log_level = log_level
        self.start_time = None
    
    def __enter__(self):
        """Start the timer"""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Log the elapsed time"""
        elapsed_time = time.time() - self.start_time
        self.logger.log(self.log_level, f"Performance: {self.operation_name} took {elapsed_time:.4f} seconds")

def setup_uncaught_exception_handler(logger):
    """
    Set up a global handler for uncaught exceptions.
    
    Args:
        logger: Logger instance to use for logging uncaught exceptions
    """
    def exception_handler(exc_type, exc_value, exc_traceback):
        """Handler for uncaught exceptions"""
        if issubclass(exc_type, KeyboardInterrupt):
            # Let keyboard interrupts pass through
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    
    # Set the exception hook
    sys.excepthook = exception_handler

# For testing
if __name__ == "__main__":
    print("Testing Logging Utilities")
    
    # Create a test logger
    test_logger = setup_logger("test", log_file="/tmp/test_log.log", level=logging.DEBUG)
    
    # Test basic logging
    print("\nTesting basic logging:")
    test_logger.debug("This is a debug message")
    test_logger.info("This is an info message")
    test_logger.warning("This is a warning message")
    test_logger.error("This is an error message")
    
    # Test exception logging
    print("\nTesting exception logging:")
    try:
        result = 1 / 0
    except Exception as e:
        log_exception(test_logger, "Division by zero error", e)
    
    # Test log capture
    print("\nTesting log capture:")
    with LogCapture("test") as capture:
        test_logger.info("This message will be captured")
        test_logger.warning("This warning will be captured too")
        
        print(f"Captured log output:\n{capture.get_logs()}")
    
    # Test performance timer
    print("\nTesting performance timer:")
    with PerformanceTimer(test_logger, "sleep operation"):
        time.sleep(0.5)  # Simulate a time-consuming operation
    
    print("\nLogging Utilities tests completed. Check /tmp/test_log.log for output.")