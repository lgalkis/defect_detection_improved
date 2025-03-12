#!/usr/bin/env python3
"""
Reorganized UI Structure for Defect Detection System
Using Model-View-Controller pattern for better maintainability
"""

"""
This code demonstrates a new structure for the UI, splitting the single
large file into modular components with clear separation of concerns.

File structure:
- ui/
  - __init__.py          # Package initialization
  - main.py              # Main entry point
  - models/              # Data models
    - __init__.py
    - settings_model.py  # Model for settings data
    - image_model.py     # Model for image data
  - views/               # UI views
    - __init__.py  
    - main_window.py     # Main window
    - image_panel.py     # Image display panels
    - metrics_panel.py   # Metrics display
    - dialogs.py         # Login and settings dialogs
  - controllers/         # Business logic
    - __init__.py
    - settings_controller.py  # Settings management
    - image_controller.py     # Image processing
    - hardware_controller.py  # Hardware integration
  - utils/               # Utilities
    - __init__.py
    - logger.py          # Logging utilities
    - security.py        # Authentication, encryption
"""

# --------------------------------------------------------------------------
# Example: Main UI Entry Point
# --------------------------------------------------------------------------

"""
File: ui/main.py
Main entry point for the Defect Detection UI
"""

import sys
import os
import argparse
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon

# Import centralized configuration
from config import config

# Import controllers and views
from .controllers.settings_controller import SettingsController
from .controllers.image_controller import ImageController
from .views.main_window import MainWindow
from .utils.logger import setup_logger

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Defect Detection UI")
    parser.add_argument("--reset", action="store_true", help="Reset settings on startup")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()

def main():
    """Main entry point for the UI application"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up logging
    log_level = "DEBUG" if args.debug else "INFO"
    logger = setup_logger("ui", config.PATHS["LOG_FILE"], level=log_level)
    logger.info("Starting UI application")
    
    # Initialize the application
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyleSheet("""
        /* Main UI styles */
        QMainWindow, QDialog, QWidget {
            background-color: #2c2c2c;
            color: #ffffff;
            font-family: 'Segoe UI', sans-serif;
        }
        QLabel {
            color: #ffffff;
        }
        QPushButton {
            background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                            stop: 0 #4CAF50, stop: 1 #2E7D32);
            border: 1px solid #1B5E20;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 16px;
            color: #ffffff;
            min-width: 40px;
        }
        QPushButton:hover {
            background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                            stop: 0 #66BB6A, stop: 1 #388E3C);
        }
        QPushButton:pressed {
            background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                            stop: 0 #2E7D32, stop: 1 #4CAF50);
            padding-left: 12px;
            padding-top: 12px;
        }
        QLineEdit {
            background-color: #3c3c3c;
            border: 1px solid #007acc;
            border-radius: 5px;
            padding: 5px;
            color: #ffffff;
        }
        QScrollArea {
            border: none;
        }
        QScrollBar:vertical {
            background: #3c3c3c;
            width: 10px;
            margin: 0px 0px 0px 0px;
        }
        QScrollBar::handle:vertical {
            background: #007acc;
            min-height: 20px;
            border-radius: 5px;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            background: none;
        }
    """)
    
    # Initialize controllers
    settings_controller = SettingsController(reset_on_start=args.reset)
    image_controller = ImageController(settings_controller)
    
    # Initialize main window
    icon_path = str(config.PATHS["APP_ICON"])
    window = MainWindow(settings_controller, image_controller)
    if os.path.exists(icon_path):
        window.setWindowIcon(QIcon(icon_path))
    
    # Show window and run application
    window.show()
    return app.exec_()

if __name__ == "__main__":
    sys.exit(main())

# --------------------------------------------------------------------------
# Example: Settings Model
# --------------------------------------------------------------------------

"""
File: ui/models/settings_model.py
Data model for application settings
"""

import os
import json
import time
import fcntl
from PyQt5.QtCore import QObject, pyqtSignal

# Import centralized configuration
from config import config
from ..utils.logger import get_logger

class SettingsModel(QObject):
    """
    Data model for application settings with change notifications.
    Uses file locking to prevent race conditions with other processes.
    """
    
    # Signals for notifying views of data changes
    settings_changed = pyqtSignal(dict)
    thresholds_changed = pyqtSignal(float, float, float)
    counters_changed = pyqtSignal(int, int)
    alarm_changed = pyqtSignal(bool)
    
    def __init__(self, settings_file=None):
        """Initialize the settings model"""
        super().__init__()
        self.logger = get_logger("settings_model")
        self.settings_file = settings_file or config.PATHS["SETTINGS_FILE"]
        self.is_saving = False
        self.settings_cache = None
        self.last_read_time = 0
        self.cache_ttl = 1.0  # Cache TTL in seconds
        
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
        """Get current settings with optional caching"""
        current_time = time.time()
        
        # Use cache if it's fresh enough
        if not bypass_cache and self.settings_cache is not None and \
           (current_time - self.last_read_time) < self.cache_ttl:
            return dict(self.settings_cache)  # Return a copy
        
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
        """Save settings to file with exclusive lock"""
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
    
    def update_thresholds(self, threshold=None, patch_threshold=None, patch_defect_ratio=None):
        """Update threshold values without changing other settings"""
        settings = self.get_settings(bypass_cache=True)
        
        # Update only provided values
        if threshold is not None:
            settings["threshold"] = threshold
        
        if patch_threshold is not None:
            settings["patch_threshold"] = patch_threshold
        
        if patch_defect_ratio is not None:
            settings["patch_defect_ratio"] = patch_defect_ratio
        
        return self.save_settings(settings)
    
    def reset_counters(self):
        """Reset all counters while preserving thresholds"""
        settings = self.get_settings(bypass_cache=True)
        
        # Preserve threshold values
        threshold = settings.get("threshold", config.DEFAULT_SETTINGS["threshold"])
        patch_threshold = settings.get("patch_threshold", config.DEFAULT_SETTINGS["patch_threshold"])
        patch_defect_ratio = settings.get("patch_defect_ratio", config.DEFAULT_SETTINGS["patch_defect_ratio"])
        
        # Reset to default settings
        settings = dict(config.DEFAULT_SETTINGS)
        
        # Restore threshold values
        settings["threshold"] = threshold
        settings["patch_threshold"] = patch_threshold
        settings["patch_defect_ratio"] = patch_defect_ratio
        
        # Add a flag to signal inference process to reset
        settings["_force_reload"] = True
        settings["_clear_images"] = True
        settings["_reset_csv"] = True
        
        return self.save_settings(settings)
    
    def reset_alarm(self):
        """Reset the alarm state"""
        settings = self.get_settings(bypass_cache=True)
        settings["alarm"] = 0
        return self.save_settings(settings)

# --------------------------------------------------------------------------
# Example: Settings Controller
# --------------------------------------------------------------------------

"""
File: ui/controllers/settings_controller.py
Controller for managing application settings
"""

import os
from PyQt5.QtCore import QObject, pyqtSignal, QTimer

# Import models
from ..models.settings_model import SettingsModel
from ..utils.logger import get_logger
from ..utils.security import verify_password

class SettingsController(QObject):
    """
    Controller for managing application settings
    Mediates between SettingsModel and UI views
    """
    
    # Signals
    settings_updated = pyqtSignal(dict)
    authentication_result = pyqtSignal(bool, str)
    
    def __init__(self, settings_file=None, reset_on_start=False):
        """Initialize the settings controller"""
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
        """Authenticate user with password"""
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
        """Log out the current user"""
        self.authenticated = False
        self.logger.info("User logged out")
        return True
    
    def get_settings(self):
        """Get current settings"""
        return self.model.get_settings()
    
    def update_thresholds(self, threshold=None, patch_threshold=None, patch_defect_ratio=None):
        """Update threshold values"""
        if not self.authenticated:
            self.logger.warning("Attempted to update thresholds without authentication")
            return False
            
        self.logger.info(f"Updating thresholds: {threshold}, {patch_threshold}, {patch_defect_ratio}")
        return self.model.update_thresholds(threshold, patch_threshold, patch_defect_ratio)
    
    def reset_counters(self):
        """Reset all counters while preserving thresholds"""
        if not self.authenticated and not self.model.is_saving:
            self.logger.warning("Attempted to reset counters without authentication")
            return False
            
        self.logger.info("Resetting counters")
        return self.model.reset_counters()
    
    def reset_alarm(self):
        """Reset the alarm state"""
        self.logger.info("Resetting alarm")
        return self.model.reset_alarm()

# --------------------------------------------------------------------------
# Example: Main Window View
# --------------------------------------------------------------------------

"""
File: ui/views/main_window.py
Main window for the Defect Detection UI
"""

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer

# Import other views
from .image_panel import GoodImagePanel, BadImagePanel
from .metrics_panel import MetricsPanel
from .dialogs import LoginDialog, SettingsDialog

# Import utilities
from ..utils.logger import get_logger

class MainWindow(QMainWindow):
    """Main application window with all UI components"""
    
    def __init__(self, settings_controller, image_controller):
        """Initialize the main window"""
        super().__init__()
        self.logger = get_logger("main_window")
        self.settings_controller = settings_controller
        self.image_controller = image_controller
        
        # UI state
        self.alarm_active = False
        self.flash_state = False
        
        # Set up UI
        self.setup_ui()
        
        # Connect signals
        self.connect_signals()
        
        # Initial update
        self.update_ui_from_settings()
    
    def setup_ui(self):
        """Set up the user interface"""
        self.setWindowTitle("Vision Pro v2")
        self.setGeometry(100, 100, 1280, 720)
        
        # Create main layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        # Add UI components
        self.create_alarm_banner(main_layout)
        self.create_main_content(main_layout)
        self.create_control_buttons(main_layout)
        
        # Set central widget
        self.setCentralWidget(main_widget)
        
        # Set up timers
        self.setup_timers()
    
    def create_alarm_banner(self, parent_layout):
        """Create the alarm banner for defect notifications"""
        self.alarm_label = QLabel("ALARM! DEFECT DETECTED!")
        self.alarm_label.setAlignment(Qt.AlignCenter)
        self.alarm_label.setStyleSheet(
            "background-color: #ff5555; color: white; font-size: 48px; border-radius: 15px;"
        )
        self.alarm_label.setFixedHeight(50)
        self.alarm_label.hide()
        parent_layout.addWidget(self.alarm_label)
    
    def create_main_content(self, parent_layout):
        """Create the main content area with image panels and metrics"""
        main_content_layout = QVBoxLayout()
        
        # Create image panels row
        panels_layout = QHBoxLayout()
        
        # Create good and bad image panels
        self.bad_panel = BadImagePanel(self.image_controller)
        self.good_panel = GoodImagePanel(self.image_controller)
        
        # Add panels to layout
        panels_layout.addWidget(self.bad_panel)
        panels_layout.addWidget(self.good_panel)
        
        # Add panels to main content with larger stretch factor
        main_content_layout.addLayout(panels_layout, 8)
        
        # Create metrics panel
        self.metrics_panel = MetricsPanel(self.settings_controller)
        
        # Add metrics panel to main content with smaller stretch factor
        main_content_layout.addWidget(self.metrics_panel, 1)
        
        # Add main content to parent layout
        parent_layout.addLayout(main_content_layout)
    
    def create_control_buttons(self, parent_layout):
        """Create control buttons for the application"""
        # Button row
        buttons_layout = QHBoxLayout()
        
        # Login/logout button
        self.login_button = QPushButton("Login")
        self.login_button.setFixedHeight(50)
        self.login_button.clicked.connect(self.toggle_login_logout)
        buttons_layout.addWidget(self.login_button)
        
        # Settings button
        self.settings_button = QPushButton("Settings")
        self.settings_button.setFixedHeight(50)
        self.settings_button.clicked.connect(self.open_settings)
        self.settings_button.setStyleSheet("""
            QPushButton {
                background-color: red;
                color: white;
                border: 1px solid #b20000;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #cc0000;
            }
            QPushButton:pressed {
                background-color: #990000;
                padding-left: 12px;
                padding-top: 12px;
            }
        """)
        self.settings_button.hide()  # Hidden until login
        buttons_layout.addWidget(self.settings_button)
        
        # Reset counters button
        self.reset_counter_button = QPushButton("Reset All Counters")
        self.reset_counter_button.setFixedHeight(50)
        self.reset_counter_button.clicked.connect(self.reset_counters)
        self.reset_counter_button.setStyleSheet("""
            QPushButton {
                background-color: #ff9800;
                color: white;
                border: 1px solid #f57c00;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #ffa726;
            }
            QPushButton:pressed {
                background-color: #f57c00;
                padding-left: 12px;
                padding-top: 12px;
            }
        """)
        self.reset_counter_button.hide()  # Hidden until login
        buttons_layout.addWidget(self.reset_counter_button)
        
        parent_layout.addLayout(buttons_layout)
        
        # Alarm reset button (separate row)
        self.reset_alarm_button = QPushButton("Reset Alarm")
        self.reset_alarm_button.setFixedHeight(50)
        self.reset_alarm_button.clicked.connect(self.reset_alarm)
        self.reset_alarm_button.setStyleSheet("""
            QPushButton {
                background-color: #007acc;
                color: white;
                border: 1px solid #005f99;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #0099ff;
            }
            QPushButton:pressed {
                background-color: #005f99;
                padding-left: 12px;
                padding-top: 12px;
            }
        """)
        self.reset_alarm_button.hide()  # Hidden until alarm is active
        parent_layout.addWidget(self.reset_alarm_button)
    
    def setup_timers(self):
        """Set up timers for UI updates and animations"""
        # Alarm flash timer
        self.alarm_timer = QTimer(self)
        self.alarm_timer.timeout.connect(self.flash_alarm)
        
        # Settings update timer
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_ui_from_settings)
        self.update_timer.start(1000)  # Update UI every second
    
    def connect_signals(self):
        """Connect signals from controllers to UI updates"""
        # Connect settings controller signals
        self.settings_controller.settings_updated.connect(self.handle_settings_updated)
        self.settings_controller.authentication_result.connect(self.handle_authentication_result)
        
        # Connect image controller signals
        self.image_controller.image_updated.connect(self.handle_image_updated)
    
    def handle_settings_updated(self, settings):
        """Handle updates from the settings controller"""
        # Update metrics display
        self.metrics_panel.update_display(settings)
        
        # Check alarm state
        alarm_active = settings.get("alarm", 0) > 0
        if alarm_active != self.alarm_active:
            self.set_alarm_state(alarm_active)
    
    def handle_authentication_result(self, success, message):
        """Handle authentication results"""
        if success:
            self.settings_button.show()
            self.reset_counter_button.show()
            self.login_button.setText("Logout")
        else:
            QMessageBox.warning(self, "Authentication Failed", message)
    
    def handle_image_updated(self, image_type, image_path):
        """Handle image updates from the image controller"""
        if image_type == "good":
            self.good_panel.update_image(image_path)
        elif image_type == "bad":
            self.bad_panel.update_image(image_path)
    
    def update_ui_from_settings(self):
        """Update UI components based on current settings"""
        settings = self.settings_controller.get_settings()
        
        # Update image panels
        self.bad_panel.update_image(settings.get("last_bad_photo", ""))
        self.good_panel.update_image(settings.get("last_good_photo", ""))
        
        # Update metrics display
        self.metrics_panel.update_display(settings)
        
        # Check alarm state
        alarm_active = settings.get("alarm", 0) > 0
        if alarm_active != self.alarm_active:
            self.set_alarm_state(alarm_active)
    
    def set_alarm_state(self, active):
        """Set the alarm state and update UI accordingly"""
        self.alarm_active = active
        
        if active:
            self.alarm_label.show()
            self.reset_alarm_button.show()
            self.alarm_timer.start(500)  # Flash every 500ms
        else:
            self.alarm_label.hide()
            self.reset_alarm_button.hide()
            self.alarm_timer.stop()
    
    def flash_alarm(self):
        """Toggle the alarm banner color to create a flashing effect"""
        self.flash_state = not self.flash_state
        color = "#ff5555" if self.flash_state else "#ffffff"
        text_color = "white" if self.flash_state else "black"
        self.alarm_label.setStyleSheet(
            f"background-color: {color}; color: {text_color}; "
            f"font-size: 48px; border-radius: 15px;"
        )
    
    def toggle_login_logout(self):
        """Toggle between logged in and logged out states"""
        if self.settings_controller.authenticated:
            # Log out
            self.settings_controller.logout()
            self.settings_button.hide()
            self.reset_counter_button.hide()
            self.login_button.setText("Login")
            QMessageBox.information(self, "Logged Out", "You have been logged out.")
        else:
            # Show login dialog
            dialog = LoginDialog(self)
            dialog.login_attempt.connect(self.settings_controller.authenticate)
            dialog.exec_()
    
    def open_settings(self):
        """Open the settings dialog"""
        if not self.settings_controller.authenticated:
            QMessageBox.warning(self, "Access Denied", "You must log in to access settings.")
            return
        
        dialog = SettingsDialog(self.settings_controller, self)
        dialog.exec_()
    
    def reset_counters(self):
        """Reset all counters"""
        if not self.settings_controller.authenticated:
            QMessageBox.warning(self, "Access Denied", "You must log in to reset counters.")
            return
        
        # Show confirmation dialog
        reply = QMessageBox.question(
            self, 'Confirm Reset',
            "This will reset all counters AND clear ALL images.\n\n"
            "This action cannot be undone. Continue?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            success = self.settings_controller.reset_counters()
            if success:
                QMessageBox.information(
                    self, "Success",
                    "Counters have been reset to 0.\n"
                    "Image folders will be cleared by the inference process."
                )
            else:
                QMessageBox.critical(
                    self, "Error",
                    "Failed to reset counters. See log for details."
                )
    
    def reset_alarm(self):
        """Reset the alarm state"""
        success = self.settings_controller.reset_alarm()
        if success:
            self.set_alarm_state(False)
        else:
            QMessageBox.critical(
                self, "Error",
                "Failed to reset alarm. See log for details."
            )

# --------------------------------------------------------------------------
# Example: Login Dialog
# --------------------------------------------------------------------------

"""
File: ui/views/dialogs.py
Dialog windows for the Defect Detection UI
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QGridLayout, QMessageBox, QWidget
)
from PyQt5.QtCore import Qt, pyqtSignal

from ..utils.logger import get_logger

class LoginDialog(QDialog):
    """Dialog for user authentication"""
    
    # Signal when user attempts login
    login_attempt = pyqtSignal(str)
    
    def __init__(self, parent=None):
        """Initialize the login dialog"""
        super().__init__(parent)
        self.logger = get_logger("login_dialog")
        self.setWindowTitle("Login")
        self.setFixedSize(480, 400)
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the user interface"""
        dialog_layout = QVBoxLayout(self)
        
        # Title
        code_label = QLabel("Enter Code:")
        code_label.setAlignment(Qt.AlignCenter)
        code_label.setStyleSheet("font-size: 32px;")
        dialog_layout.addWidget(code_label)
        
        # Password input
        self.code_input = QLineEdit()
        self.code_input.setEchoMode(QLineEdit.Password)
        self.code_input.setAlignment(Qt.AlignCenter)
        self.code_input.setStyleSheet("font-size: 20px; padding: 10px;")
        self.code_input.returnPressed.connect(self.handle_login)
        dialog_layout.addWidget(self.code_input)
        
        # Numeric keyboard
        keyboard_widget = self.create_numeric_keyboard()
        dialog_layout.addWidget(keyboard_widget)
    
    def create_numeric_keyboard(self):
        """Create numeric keypad for PIN entry"""
        keyboard_widget = QWidget()
        keyboard_layout = QGridLayout()
        
        # Define buttons
        buttons = [
            ["1", "2", "3"],
            ["4", "5", "6"],
            ["7", "8", "9"],
            ["0", "Clear", "OK"]
        ]
        
        # Create and connect buttons
        for row_idx, row in enumerate(buttons):
            for col_idx, button_text in enumerate(row):
                button = QPushButton(button_text)
                button.setFixedSize(80, 60)
                
                # Connect button to handler
                if button_text == "Clear":
                    button.clicked.connect(self.clear_input)
                elif button_text == "OK":
                    button.clicked.connect(self.handle_login)
                else:
                    button.clicked.connect(lambda _, text=button_text: self.add_digit(text))
                
                keyboard_layout.addWidget(button, row_idx, col_idx)
        
        keyboard_widget.setLayout(keyboard_layout)
        return keyboard_widget
    
    def clear_input(self):
        """Clear the input field"""
        self.code_input.clear()
    
    def add_digit(self, digit):
        """Add a digit to the input field"""
        self.code_input.setText(self.code_input.text() + digit)
    
    def handle_login(self):
        """Handle login attempt"""
        password = self.code_input.text()
        
        if not password:
            QMessageBox.warning(self, "Error", "Please enter a code.")
            return
        
        # Emit signal with entered password
        self.login_attempt.emit(password)
        
        # Dialog will be closed by the controller if authentication succeeds

class SettingsDialog(QDialog):
    """Dialog for configuring application settings"""
    
    def __init__(self, settings_controller, parent=None):
        """Initialize the settings dialog"""
        super().__init__(parent)
        self.logger = get_logger("settings_dialog")
        self.settings_controller = settings_controller
        self.setWindowTitle("Settings")
        self.setFixedSize(480, 600)
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the user interface"""
        dialog_layout = QVBoxLayout(self)
        
        # Global Threshold section
        global_threshold_label = QLabel("Set Global Threshold")
        global_threshold_label.setStyleSheet("font-size: 24px;")
        global_threshold_label.setAlignment(Qt.AlignCenter)
        dialog_layout.addWidget(global_threshold_label)
        
        # Get current settings
        current_settings = self.settings_controller.get_settings()
        
        # Global threshold input
        self.global_threshold_input = QLineEdit()
        self.global_threshold_input.setText(str(current_settings.get("threshold", 0.85)))
        self.global_threshold_input.setAlignment(Qt.AlignCenter)
        self.global_threshold_input.setStyleSheet("font-size: 20px; padding: 10px;")
        dialog_layout.addWidget(self.global_threshold_input)
        
        # Patch Threshold section
        patch_threshold_label = QLabel("Set Patch Threshold")
        patch_threshold_label.setStyleSheet("font-size: 24px;")
        patch_threshold_label.setAlignment(Qt.AlignCenter)
        dialog_layout.addWidget(patch_threshold_label)
        
        # Patch threshold input
        self.patch_threshold_input = QLineEdit()
        self.patch_threshold_input.setText(str(current_settings.get("patch_threshold", 0.7)))
        self.patch_threshold_input.setAlignment(Qt.AlignCenter)
        self.patch_threshold_input.setStyleSheet("font-size: 20px; padding: 10px;")
        dialog_layout.addWidget(self.patch_threshold_input)
        
        # Patch Defect Ratio section
        patch_ratio_label = QLabel("Set Patch Defect Ratio")
        patch_ratio_label.setStyleSheet("font-size: 24px;")
        patch_ratio_label.setAlignment(Qt.AlignCenter)
        dialog_layout.addWidget(patch_ratio_label)
        
        # Patch ratio input
        self.patch_ratio_input = QLineEdit()
        self.patch_ratio_input.setText(str(current_settings.get("patch_defect_ratio", 0.45)))
        self.patch_ratio_input.setAlignment(Qt.AlignCenter)
        self.patch_ratio_input.setStyleSheet("font-size: 20px; padding: 10px;")
        dialog_layout.addWidget(self.patch_ratio_input)
        
        # Field selector buttons
        field_selector_layout = QHBoxLayout()
        
        # Global threshold button
        global_btn = QPushButton("Global Threshold")
        global_btn.clicked.connect(lambda: self.set_active_input(self.global_threshold_input))
        
        # Patch threshold button
        patch_btn = QPushButton("Patch Threshold")
        patch_btn.clicked.connect(lambda: self.set_active_input(self.patch_threshold_input))
        
        # Ratio button
        ratio_btn = QPushButton("Defect Ratio")
        ratio_btn.clicked.connect(lambda: self.set_active_input(self.patch_ratio_input))
        
        # Add buttons to layout
        for btn in [global_btn, patch_btn, ratio_btn]:
            field_selector_layout.addWidget(btn)
        
        dialog_layout.addLayout(field_selector_layout)
        
        # Numeric keyboard
        keyboard_widget = self.create_numeric_keyboard()
        dialog_layout.addWidget(keyboard_widget)
        
        # Set default active input with highlight
        self.active_input = self.global_threshold_input
        self.set_active_input(self.global_threshold_input)
    
    def create_numeric_keyboard(self):
        """Create numeric keypad with decimal point for settings input"""
        keyboard_widget = QWidget()
        keyboard_layout = QGridLayout()
        
        # Define buttons
        buttons = [
            ["1", "2", "3"],
            ["4", "5", "6"],
            ["7", "8", "9"],
            ["0", ".", "Clear"],
            ["Cancel", "OK"]
        ]
        
        # Create and connect buttons
        for row_idx, row in enumerate(buttons):
            for col_idx, button_text in enumerate(row):
                button = QPushButton(button_text)
                button.setFixedSize(80, 60)
                
                # Connect button to handler
                if button_text == "Clear":
                    button.clicked.connect(self.clear_input)
                elif button_text == "OK":
                    button.clicked.connect(self.save_settings)
                elif button_text == "Cancel":
                    button.clicked.connect(self.reject)
                else:
                    button.clicked.connect(lambda _, text=button_text: self.add_character(text))
                
                # Special case for the last row that spans multiple columns
                if row_idx == 4:
                    if col_idx == 0:  # Cancel button
                        keyboard_layout.addWidget(button, row_idx, 0, 1, 1)
                    elif col_idx == 1:  # OK button
                        keyboard_layout.addWidget(button, row_idx, 1, 1, 2)
                else:
                    keyboard_layout.addWidget(button, row_idx, col_idx)
        
        keyboard_widget.setLayout(keyboard_layout)
        return keyboard_widget
    
    def set_active_input(self, input_field):
        """Set the active input field and update its styling"""
        # Reset styling on all input fields
        for field in [self.global_threshold_input, self.patch_threshold_input, self.patch_ratio_input]:
            field.setStyleSheet("font-size: 20px; padding: 10px;")
        
        # Set and highlight the active field
        self.active_input = input_field
        input_field.setFocus()
        input_field.setStyleSheet(
            "font-size: 20px; padding: 10px; "
            "background-color: #1e5799; color: white;"
        )
    
    def clear_input(self):
        """Clear the active input field"""
        if self.active_input:
            self.active_input.clear()
    
    def add_character(self, character):
        """Add a character to the active input field"""
        if self.active_input:
            self.active_input.setText(self.active_input.text() + character)
    
    def save_settings(self):
        """Validate and save the settings"""
        try:
            # Parse input values
            global_threshold = float(self.global_threshold_input.text())
            patch_threshold = float(self.patch_threshold_input.text())
            patch_ratio = float(self.patch_ratio_input.text())
            
            # Validate ranges
            if not (0 < global_threshold < 10):
                QMessageBox.warning(self, "Invalid Input", "Global threshold must be between 0 and 10.")
                return
                
            if not (0 < patch_threshold < 10):
                QMessageBox.warning(self, "Invalid Input", "Patch threshold must be between 0 and 10.")
                return
                
            if not (0 < patch_ratio < 1):
                QMessageBox.warning(self, "Invalid Input", "Patch defect ratio must be between 0 and 1.")
                return
            
            # Update thresholds
            success = self.settings_controller.update_thresholds(
                global_threshold, patch_threshold, patch_ratio
            )
            
            if success:
                QMessageBox.information(self, "Success", "Settings updated successfully.")
                self.accept()  # Close dialog
            else:
                QMessageBox.critical(self, "Error", "Failed to save settings.")
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter numeric values for all fields.")
        except Exception as e:
            self.logger.error(f"Error saving settings: {e}")
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")
