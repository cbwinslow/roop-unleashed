"""
Enhanced error handling framework for roop-unleashed.
Provides comprehensive error handling, logging, and recovery mechanisms.
"""

import logging
import traceback
import functools
import time
import os
import sys
from typing import Callable, Any, Optional, Type, Tuple
from contextlib import contextmanager


class RoopException(Exception):
    """Base exception class for roop-unleashed specific errors."""
    pass


class ConfigurationError(RoopException):
    """Raised when configuration is invalid or missing."""
    pass


class ModelError(RoopException):
    """Raised when there are issues with AI models."""
    pass


class ProcessingError(RoopException):
    """Raised when face processing operations fail."""
    pass


class GPUError(RoopException):
    """Raised when GPU operations fail."""
    pass


class ErrorHandler:
    """Centralized error handling and recovery system."""
    
    def __init__(self, settings=None):
        self.settings = settings
        self.setup_logging()
        self.error_count = 0
        self.last_error_time = 0
        
    def setup_logging(self):
        """Setup enhanced logging configuration."""
        log_level = logging.INFO
        log_file = "./logs/roop.log"
        
        if self.settings:
            log_level_str = self.settings.get_ai_setting('logging.level', 'INFO')
            log_level = getattr(logging, log_level_str.upper(), logging.INFO)
            log_file = self.settings.get_ai_setting('logging.file_path', log_file)
        
        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)

    def handle_error(self, error: Exception, context: str = "", 
                    recoverable: bool = True) -> bool:
        """
        Handle an error with appropriate logging and recovery.
        
        Args:
            error: The exception that occurred
            context: Additional context about where the error occurred
            recoverable: Whether recovery should be attempted
            
        Returns:
            bool: True if error was handled/recovered, False otherwise
        """
        self.error_count += 1
        current_time = time.time()
        
        error_msg = f"Error in {context}: {str(error)}"
        self.logger.error(error_msg)
        
        if self.settings and self.settings.get_ai_setting('error_handling.detailed_logging', True):
            self.logger.debug(f"Full traceback:\n{traceback.format_exc()}")
        
        # Check for error rate limiting
        if current_time - self.last_error_time < 1.0:  # Too many errors too quickly
            if self.error_count > 10:
                self.logger.critical("Too many errors in short time, stopping")
                return False
        else:
            self.error_count = 0
        
        self.last_error_time = current_time
        
        # Attempt recovery if possible
        if recoverable and self.settings:
            return self._attempt_recovery(error, context)
        
        return False

    def _attempt_recovery(self, error: Exception, context: str) -> bool:
        """Attempt to recover from an error."""
        try:
            if isinstance(error, GPUError):
                return self._recover_gpu_error()
            elif isinstance(error, ModelError):
                return self._recover_model_error()
            elif isinstance(error, ConfigurationError):
                return self._recover_config_error()
            else:
                # Generic recovery
                return self._generic_recovery()
        except Exception as recovery_error:
            self.logger.error(f"Recovery attempt failed: {recovery_error}")
            return False

    def _recover_gpu_error(self) -> bool:
        """Attempt to recover from GPU-related errors."""
        if self.settings.get_ai_setting('error_handling.fallback_to_cpu', True):
            self.logger.info("GPU error detected, falling back to CPU")
            self.settings.force_cpu = True
            return True
        return False

    def _recover_model_error(self) -> bool:
        """Attempt to recover from model-related errors."""
        self.logger.info("Model error detected, attempting to reload")
        # Could implement model reloading logic here
        return False

    def _recover_config_error(self) -> bool:
        """Attempt to recover from configuration errors."""
        self.logger.info("Configuration error detected, resetting to defaults")
        try:
            self.settings._create_minimal_config()
            self.settings.load()
            return True
        except Exception:
            return False

    def _generic_recovery(self) -> bool:
        """Generic recovery mechanism."""
        self.logger.info("Attempting generic error recovery")
        time.sleep(1)  # Brief pause before retry
        return True


def retry_on_error(max_retries: int = 3, delay: float = 1.0, 
                  exceptions: Tuple[Type[Exception], ...] = (Exception,)):
    """
    Decorator to automatically retry functions that fail.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds
        exceptions: Tuple of exception types to catch and retry on
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logging.warning(
                            f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay} seconds..."
                        )
                        time.sleep(delay)
                    else:
                        logging.error(
                            f"Function {func.__name__} failed after {max_retries + 1} attempts"
                        )
            
            raise last_exception
        return wrapper
    return decorator


@contextmanager
def error_context(context_name: str, error_handler: Optional[ErrorHandler] = None,
                 suppress_errors: bool = False):
    """
    Context manager for error handling with optional suppression.
    
    Args:
        context_name: Name of the context for error reporting
        error_handler: ErrorHandler instance to use
        suppress_errors: Whether to suppress errors and continue
    """
    try:
        yield
    except Exception as e:
        if error_handler:
            handled = error_handler.handle_error(e, context_name)
            if not handled and not suppress_errors:
                raise
        elif not suppress_errors:
            logging.error(f"Error in {context_name}: {e}")
            raise
        else:
            logging.warning(f"Suppressed error in {context_name}: {e}")


def safe_execute(func: Callable, *args, error_handler: Optional[ErrorHandler] = None,
                default_return: Any = None, **kwargs) -> Any:
    """
    Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        *args: Arguments to pass to function
        error_handler: ErrorHandler instance to use
        default_return: Value to return if function fails
        **kwargs: Keyword arguments to pass to function
        
    Returns:
        Function result or default_return if function fails
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if error_handler:
            error_handler.handle_error(e, f"safe_execute({func.__name__})")
        else:
            logging.error(f"Error in {func.__name__}: {e}")
        return default_return


class PerformanceMonitor:
    """Monitor performance and detect issues."""
    
    def __init__(self):
        self.start_times = {}
        self.durations = {}
        
    @contextmanager
    def time_operation(self, operation_name: str):
        """Time an operation and log if it's slow."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.durations[operation_name] = duration
            
            if duration > 30:  # Log operations that take more than 30 seconds
                logging.warning(f"Slow operation detected: {operation_name} took {duration:.2f}s")
    
    def get_stats(self) -> dict:
        """Get performance statistics."""
        return {
            'average_durations': {
                name: sum(durations) / len(durations)
                for name, durations in self.durations.items()
            },
            'total_operations': len(self.durations)
        }