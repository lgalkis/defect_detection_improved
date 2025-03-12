#!/usr/bin/env python3
"""
Settings Manager for Defect Detection System
Provides thread-safe and process-safe settings management with file locking.
"""

import os
import json
import time
import fcntl
import traceback
from config import config  # Import the centralized configuration

# Set up logger
logger = config.setup_logger("settings_manager")

class SettingsManager:
    """
    Thread-safe and process-safe settings manager with file locking
    to prevent race conditions between UI and inference processes.
    """
    
    def __init__(self, settings_file=None):
        """
        Initialize the settings manager.
        
        Args:
            settings_file: Path to the settings file (defaults to config.PATHS.SETTINGS_FILE)
        """
        self.settings_file = settings_file or config.PATHS["SETTINGS_FILE"]
        self.settings_cache = None
        self.last_read_time = 0
        self.cache_ttl = 1.0  # Cache time-to-live in seconds
        
        # Ensure the settings file exists
        self._ensure_settings_file()
    
    def _ensure_settings_file(self):
        """
        Ensure the settings file exists and has valid JSON.
        If it doesn't exist or is invalid, create it with default settings.
        """
        # Make sure directory exists
        os.makedirs(os.path.dirname(self.settings_file), exist_ok=True)
        
        # Check if file exists and is valid JSON
        try:
            if not os.path.exists(self.settings_file):
                logger.info(f"Settings file not found, creating with default values: {self.settings_file}")
                with open(self.settings_file, "w") as f:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    json.dump(config.DEFAULT_SETTINGS, f, indent=4)
                    fcntl.flock(f, fcntl.LOCK_UN)
                # Set appropriate permissions
                try:
                    os.chmod(self.settings_file, 0o666)  # Make writable by all users
                except Exception as e:
                    logger.warning(f"Failed to set permissions on settings file: {e}")
            else:
                # Validate existing settings file
                with open(self.settings_file, "r") as f:
                    fcntl.flock(f, fcntl.LOCK_SH)
                    json.load(f)  # Just try to parse it
                    fcntl.flock(f, fcntl.LOCK_UN)
        except json.JSONDecodeError:
            # File exists but is invalid JSON, recreate it
            logger.warning(f"Invalid JSON in settings file, recreating with defaults: {self.settings_file}")
            with open(self.settings_file, "w") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                json.dump(config.DEFAULT_SETTINGS, f, indent=4)
                fcntl.flock(f, fcntl.LOCK_UN)
        except Exception as e:
            logger.error(f"Unexpected error ensuring settings file: {e}")
            logger.error(traceback.format_exc())
    
    def get_settings(self, bypass_cache=False):
        """
        Get the current settings with optional caching.
        
        Args:
            bypass_cache: If True, always read from disk instead of cache
            
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
            logger.error(f"Error reading settings: {e}")
            # Reset cache and recreate settings file
            self.settings_cache = None
            self._ensure_settings_file()
            return dict(config.DEFAULT_SETTINGS)
        except Exception as e:
            logger.error(f"Unexpected error reading settings: {e}")
            logger.error(traceback.format_exc())
            return dict(config.DEFAULT_SETTINGS)
    
    def save_settings(self, settings):
        """
        Save settings to file with exclusive lock.
        
        Args:
            settings: Dictionary of settings to save
            
        Returns:
            Boolean indicating success or failure
        """
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
            
            logger.debug("Settings saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving settings: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def update_settings(self, **kwargs):
        """
        Update specific settings while preserving others.
        
        Args:
            **kwargs: Key-value pairs of settings to update
            
        Returns:
            Boolean indicating success or failure
        """
        try:
            # Get current settings
            current = self.get_settings(bypass_cache=True)
            
            # Update with new values
            current.update(kwargs)
            
            # Save back to file
            return self.save_settings(current)
        except Exception as e:
            logger.error(f"Error updating settings: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def increment_counter(self, counter_name):
        """
        Safely increment a counter in the settings.
        
        Args:
            counter_name: Name of the counter to increment
            
        Returns:
            New counter value or None if failed
        """
        try:
            current = self.get_settings(bypass_cache=True)
            
            # Get current value (default to 0 if not present)
            current_value = current.get(counter_name, 0)
            
            # Increment the counter
            new_value = current_value + 1
            current[counter_name] = new_value
            
            # Save the updated settings
            if self.save_settings(current):
                return new_value
            else:
                return None
        except Exception as e:
            logger.error(f"Error incrementing counter {counter_name}: {e}")
            return None
    
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
            new_settings = dict(config.DEFAULT_SETTINGS)
            
            # Restore threshold values
            new_settings["threshold"] = threshold
            new_settings["patch_threshold"] = patch_threshold
            new_settings["patch_defect_ratio"] = patch_defect_ratio
            
            # Save the updated settings
            if self.save_settings(new_settings):
                logger.info("Counters reset successfully")
                return True
            else:
                logger.error("Failed to save settings after resetting counters")
                return False
        except Exception as e:
            logger.error(f"Error resetting counters: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def update_after_analysis(self, is_defect, reconstruction_error, image_path, patch_metrics=None):
        """
        Update settings based on analysis results.
        
        Args:
            is_defect: Boolean indicating if a defect was detected
            reconstruction_error: Global reconstruction error value
            image_path: Path to the analyzed image
            patch_metrics: Dictionary of patch analysis metrics
            
        Returns:
            Boolean indicating success or failure
        """
        try:
            # Get current settings with fresh read
            current = self.get_settings(bypass_cache=True)
            
            # Update reconstruction error
            current["last_reconstruction_error"] = reconstruction_error
            
            # Update patch-related metrics if available
            if patch_metrics:
                current["latest_patch_value"] = patch_metrics.get("mean_patch_error", 0.0)
                current["latest_defect_ratio"] = patch_metrics.get("defect_patch_ratio", 0.0)

            if is_defect:
                # Update bad count and bad photo tracking
                current["bad_count"] = current.get("bad_count", 0) + 1
                current["last_bad_photo"] = image_path
                
                # Update list of recent bad photos
                last_10 = current.get("last_10_bad_photos", [])
                last_10.append(image_path)
                current["last_10_bad_photos"] = last_10[-10:]  # Keep only most recent 10
                
                # Set alarm if it's not already active
                if current.get("alarm", 0) == 0:
                    current["alarm"] = 1
            else:
                # Update good count and good photo reference
                current["good_count"] = current.get("good_count", 0) + 1
                current["last_good_photo"] = image_path
                # Keep the current alarm state - don't change it if it's been reset
            
            # Save all changes
            return self.save_settings(current)
        except Exception as e:
            logger.error(f"Error updating settings after analysis: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def check_for_reset_signal(self):
        """
        Check if a reset signal has been sent from the UI.
        
        Returns:
            Tuple of (reset_detected, clear_images, reset_csv)
        """
        reset_signal_file = config.PATHS["RESET_SIGNAL_FILE"]
        reset_detected = False
        clear_images = False
        reset_csv = False
        
        # Method 1: Check for reset signal file
        if os.path.exists(reset_signal_file):
            try:
                # Read the signal file
                with open(reset_signal_file, "r") as f:
                    signal_content = f.read()
                    # Check for flags
                    if "clear_images=true" in signal_content.lower():
                        clear_images = True
                    if "reset_csv=true" in signal_content.lower():
                        reset_csv = True
                    
                # Remove the signal file after reading
                os.remove(reset_signal_file)
                logger.info(f"Reset signal detected: {signal_content}")
                reset_detected = True
            except Exception as e:
                logger.error(f"Error processing reset signal file: {e}")
        
        # Method 2: Check for reset flags in settings
        try:
            current_settings = self.get_settings(bypass_cache=True)
            
            # Check for flags in settings
            if current_settings.get("_force_reload", False):
                logger.info("Force reload flag detected in settings")
                reset_detected = True
                
                # Clear the flag
                current_settings["_force_reload"] = False
                self.save_settings(current_settings)
                
            if current_settings.get("_clear_images", False):
                logger.info("Clear images flag detected in settings")
                clear_images = True
                
                # Clear the flag
                current_settings["_clear_images"] = False
                self.save_settings(current_settings)
                
            if current_settings.get("_reset_csv", False):
                logger.info("Reset CSV flag detected in settings")
                reset_csv = True
                
                # Clear the flag
                current_settings["_reset_csv"] = False
                self.save_settings(current_settings)
                
        except Exception as e:
            logger.error(f"Error checking settings for reset flags: {e}")
        
        return reset_detected, clear_images, reset_csv

# For testing
if __name__ == "__main__":
    # Test the settings manager
    settings_manager = SettingsManager()
    
    # Get current settings
    current_settings = settings_manager.get_settings()
    print("Current settings:")
    for key, value in current_settings.items():
        print(f"  {key}: {value}")
    
    # Test counter increment
    print("\nTesting counter increment:")
    new_image_counter = settings_manager.increment_counter("image_counter")
    print(f"Incremented image counter to: {new_image_counter}")
    
    # Test settings update
    print("\nTesting settings update:")
    result = settings_manager.update_settings(threshold=0.75)
    print(f"Update result: {result}")
    updated_settings = settings_manager.get_settings()
    print(f"New threshold: {updated_settings['threshold']}")
    
    # Test reset counters
    print("\nTesting counter reset:")
    result = settings_manager.reset_counters()
    print(f"Reset result: {result}")
    reset_settings = settings_manager.get_settings()
    print(f"After reset - image_counter: {reset_settings.get('image_counter', 0)}")
    print(f"After reset - threshold: {reset_settings.get('threshold', 0)}")
    
    print("\nSettings Manager tests completed successfully.")