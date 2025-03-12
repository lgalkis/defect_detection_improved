#!/usr/bin/env python3
"""
Settings Model for Defect Detection System
Provides thread-safe and process-safe settings management with file locking
"""

import os
import json
import time
import fcntl
from PyQt5.QtCore import QObject, pyqtSignal

# Import centralized configuration
from config import config
from utils.logger import get_logger

class SettingsModel(QObject):
    """
    Data model for application settings with change notifications.
    Uses file locking to prevent race conditions with other processes.
    """
    
    # Define signals for notifying views of data changes
    settings_changed = pyqtSignal(dict)
    thresholds_changed = pyqtSignal(float, float, float)
    counters_changed = pyqtSignal(int, int)
    alarm_changed = pyqtSignal(bool)
    
    def __init__(self, settings_file=None):
        """
        Initialize the settings model.
        
        Args:
            settings_file: Path to the settings file (defaults to config.PATHS["SETTINGS_FILE"])
        """
        super().__init__()
        self.logger = get_logger("settings_model")
        self.settings_file = settings_file or config.PATHS["SETTINGS_FILE"]
        self.is_saving = False
        self.settings_cache = None
        self.last_read_time = 0
        self.cache_ttl = 1.0  # Cache time-to-live in seconds
        
        # Ensure settings file exists
        self._ensure_settings_file()
    
    def _ensure_settings_file(self):
        """Ensure settings file exists and has valid JSON"""
        # Make sure directory exists
        os.makedirs(os.path.dirname(self.settings_file), exist_ok=True)
        
        # Check if file exists and is valid JSON
        try:
            if not os.path.exists(self.settings_file):
                self.logger.info(f"Settings file not found, creating with default values")
                with open(self.settings_file, "w") as f:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    json.dump(config.DEFAULT_SETTINGS, f, indent=4)
                    fcntl.flock(f, fcntl.LOCK_UN)
                # Set appropriate permissions
                try:
                    os.chmod(self.settings_file, 0o666)  # rw-rw-rw-
                except Exception as e:
                    self.logger.warning(f"Failed to set permissions on settings file: {e}")
            else:
                # Validate existing settings file
                with open(self.settings_file, "r") as f:
                    fcntl.flock(f, fcntl.LOCK_SH)
                    json.load(f)  # Just try to parse it
                    fcntl.flock(f, fcntl.LOCK_UN)
        except json.JSONDecodeError:
            # File exists but is invalid JSON, recreate it
            self.logger.warning(f"Invalid JSON in settings file, recreating with defaults")
            with open(self.settings_file, "w") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                json.dump(config.DEFAULT_SETTINGS, f, indent=4)
                fcntl.flock(f, fcntl.LOCK_UN)
        except Exception as e:
            self.logger.error(f"Unexpected error ensuring settings file: {e}")
    
    def get_settings(self, bypass_cache=False):
        """
        Get current settings with optional caching.
        
        Args:
            bypass_cache: Whether to bypass the cache and read from disk
            
        Returns:
            Dictionary containing the settings
        """
        current_time = time.time()
        
        # Use cache if it's fresh enough
        if not bypass_cache and self.settings_cache is not None and \
           (current_time - self.last_read_time) < self.cache_ttl:
            return dict(self.settings_cache)  # Return a copy to avoid modification
        
        try:
            with open(self.settings_file, "r") as f:
                fcntl.flock(f, fcntl.LOCK_SH)  # Shared lock for reading
                settings = json.load(f)
                fcntl.flock(f, fcntl.LOCK_UN)
            
            # Update cache
            self.settings_cache = dict(settings)
            self.last_read_time = current_time
            
            return dict(settings)  # Return a copy
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.error(f"Error reading settings: {e}")
            # Reset cache and recreate settings file
            self.settings_cache = None
            self._ensure_settings_file()
            return dict(config.DEFAULT_SETTINGS)
        except Exception as e:
            self.logger.error(f"Unexpected error reading settings: {e}")
            return dict(config.DEFAULT_SETTINGS)
    
    def save_settings(self, settings):
        """
        Save settings to file with exclusive lock.
        
        Args:
            settings: Dictionary of settings to save
            
        Returns:
            Boolean indicating success or failure
        """
        self.is_saving = True
        try:
            # Make a copy to avoid modifying the input
            settings_copy = dict(settings)
            
            with open(self.settings_file, "w") as f:
                fcntl.flock(f, fcntl.LOCK_EX)  # Exclusive lock for writing
                json.dump(settings_copy, f, indent=4)
                fcntl.flock(f, fcntl.LOCK_UN)
            
            # Update cache
            self.settings_cache = dict(settings_copy)
            self.last_read_time = time.time()
            
            # Emit signals
            self.settings_changed.emit(dict(settings_copy))
            
            # Emit specific signals for views that need them
            self.thresholds_changed.emit(
                settings_copy.get("threshold", 0.0),
                settings_copy.get("patch_threshold", 0.0),
                settings_copy.get("patch_defect_ratio", 0.0)
            )
            
            self.counters_changed.emit(
                settings_copy.get("good_count", 0),
                settings_copy.get("bad_count", 0)
            )
            
            self.alarm_changed.emit(settings_copy.get("alarm", 0) > 0)
            
            self.logger.debug("Settings saved successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error saving settings: {e}")
            return False
        finally:
            self.is_saving = False
    
    def update_settings(self, **kwargs):
        """
        Update specific settings while preserving others.
        
        Args:
            **kwargs: Key-value pairs of settings to update
            
        Returns:
            Boolean indicating success or failure
        """
        settings = self.get_settings(bypass_cache=True)
        
        # Update with new values
        settings.update(kwargs)
        
        # Save settings
        return self.save_settings(settings)
    
    def reset_counters(self):
        """
        Reset all counters while preserving thresholds and other settings.
        
        Returns:
            Boolean indicating success or failure
        """
        try:
            # Get current settings
            current = self.get_settings(bypass_cache=True)
            
            # Preserve threshold values
            threshold = current.get("threshold", config.DEFAULT_SETTINGS["threshold"])
            patch_threshold = current.get("patch_threshold", config.DEFAULT_SETTINGS["patch_threshold"])
            patch_defect_ratio = current.get("patch_defect_ratio", config.DEFAULT_SETTINGS["patch_defect_ratio"])
            
            # Reset to default settings
            settings = dict(config.DEFAULT_SETTINGS)
            
            # Restore threshold values
            settings["threshold"] = threshold
            settings["patch_threshold"] = patch_threshold
            settings["patch_defect_ratio"] = patch_defect_ratio
            
            # Add flags for inference process
            settings["_force_reload"] = True
            settings["_clear_images"] = True
            settings["_reset_csv"] = True
            settings["reset_timestamp"] = time.time()
            
            # Save the updated settings
            return self.save_settings(settings)
        except Exception as e:
            self.logger.error(f"Error resetting counters: {e}")
            return False
    
    def reset_alarm(self):
        """
        Reset the alarm state.
        
        Returns:
            Boolean indicating success or failure
        """
        try:
            current = self.get_settings(bypass_cache=True)
            current["alarm"] = 0
            return self.save_settings(current)
        except Exception as e:
            self.logger.error(f"Error resetting alarm: {e}")
            return False