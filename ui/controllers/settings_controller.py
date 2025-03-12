#!/usr/bin/env python3
"""
Settings Controller for Defect Detection System
Manages settings, authentication, and provides access to the settings model
"""

import os
import time
from PyQt5.QtCore import QObject, pyqtSignal, QTimer

# Import models
from models.settings_model import SettingsModel
from utils.logger import get_logger
from utils.security import verify_password

class SettingsController(QObject):
    """
    Controller for managing application settings.
    Mediates between SettingsModel and UI views.
    """
    
    # Define signals
    settings_updated = pyqtSignal(dict)
    authentication_result = pyqtSignal(bool, str)
    
    def __init__(self, settings_file=None, reset_on_start=False):
        """
        Initialize the settings controller.
        
        Args:
            settings_file: Path to the settings file (optional)
            reset_on_start: Whether to reset settings on startup
        """
        super().__init__()
        self.logger = get_logger("settings_controller")
        self.model = SettingsModel(settings_file)
        self.authenticated = False
        
        # Forward model signals to controller signals
        self.model.settings_changed.connect(self.settings_updated)
        
        # Set up file watcher to detect external changes to settings
        self._setup_file_watcher()
        
        # Reset settings if requested
        if reset_on_start:
            self.reset_counters()
    
    def _setup_file_watcher(self):
        """Set up a timer to periodically check settings for changes"""
        self.check_timer = QTimer(self)
        self.check_timer.timeout.connect(self._check_settings_changed)
        self.check_timer.start(1000)  # Check every second
    
    def _check_settings_changed(self):
        """Check if settings have changed externally"""
        if not self.model.is_saving:
            # Force a fresh read from disk
            settings = self.model.get_settings(bypass_cache=True)
            # Emit signal with fresh settings
            self.settings_updated.emit(settings)
    
    def authenticate(self, password):
        """
        Authenticate user with password.
        
        Args:
            password: Password to verify
            
        Returns:
            Boolean indicating authentication success
        """
        result = verify_password(password)
        
        if result:
            self.logger.info("User authenticated successfully")
            self.authenticated = True
            self.authentication_result.emit(True, "Authentication successful")
        else:
            self.logger.warning("Authentication failed")
            self.authenticated = False
            self.authentication_result.emit(False, "Invalid password")
        
        return result
    
    def logout(self):
        """
        Log out the current user.
        
        Returns:
            Boolean indicating success
        """
        self.authenticated = False
        self.logger.info("User logged out")
        return True
    
    def get_settings(self):
        """
        Get current settings.
        
        Returns:
            Dictionary of settings
        """
        return self.model.get_settings()
    
    def update_thresholds(self, threshold=None, patch_threshold=None, patch_defect_ratio=None):
        """
        Update threshold values.
        
        Args:
            threshold: Global threshold value
            patch_threshold: Patch threshold value
            patch_defect_ratio: Patch defect ratio value
            
        Returns:
            Boolean indicating success
        """
        if not self.authenticated:
            self.logger.warning("Attempted to update thresholds without authentication")
            return False
            
        self.logger.info(f"Updating thresholds: {threshold}, {patch_threshold}, {patch_defect_ratio}")
        
        # Create update keyword arguments
        kwargs = {}
        if threshold is not None:
            kwargs["threshold"] = threshold
        if patch_threshold is not None:
            kwargs["patch_threshold"] = patch_threshold
        if patch_defect_ratio is not None:
            kwargs["patch_defect_ratio"] = patch_defect_ratio
        
        # Update settings
        return self.model.update_settings(**kwargs)
    
    def reset_counters(self):
        """
        Reset all counters while preserving thresholds.
        
        Returns:
            Boolean indicating success
        """
        if not self.authenticated and not self.model.is_saving:
            self.logger.warning("Attempted to reset counters without authentication")
            return False
            
        self.logger.info("Resetting counters")
        
        # Create reset signal
        self.create_reset_signal()
        
        # Reset counters in settings
        return self.model.reset_counters()
    
    def reset_alarm(self):
        """
        Reset the alarm state.
        
        Returns:
            Boolean indicating success
        """
        self.logger.info("Resetting alarm")
        return self.model.reset_alarm()
    
    def create_reset_signal(self):
        """Create a reset signal file to notify the inference process."""
        try:
            # Import configuration
            from config import config
            
            reset_signal_file = config.PATHS["RESET_SIGNAL_FILE"]
            with open(reset_signal_file, "w") as f:
                f.write(f"reset_timestamp={time.time()}\n")
                f.write(f"clear_images=true\n")
                f.write(f"reset_csv=true\n")
            
            self.logger.info("Reset signal created with clear_images and reset_csv flags")
            return True
        except Exception as e:
            self.logger.error(f"Error creating reset signal: {e}")
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
            Boolean indicating success
        """
        try:
            # Get current settings with fresh read
            current = self.model.get_settings(bypass_cache=True)
            
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
            
            # Increment image counter
            current["image_counter"] = current.get("image_counter", 0) + 1
            
            # Save all changes
            return self.model.save_settings(current)
        except Exception as e:
            self.logger.error(f"Error updating settings after analysis: {e}")
            return False
