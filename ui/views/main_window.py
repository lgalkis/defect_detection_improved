#!/usr/bin/env python3
"""
Main Window for Defect Detection System
Provides the primary user interface with image displays and controls
"""

import os
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QMessageBox, QFrame
)
from PyQt5.QtCore import Qt, QTimer, pyqtSlot

# Import views
from views.image_panel import GoodImagePanel, BadImagePanel
from views.metrics_panel import MetricsPanel
from views.dialogs import LoginDialog, SettingsDialog
from views.preview_dialog import ImagePreviewDialog

# Import controllers
from controllers.hardware_controller import HardwareController

# Import utilities
from utils.logger import get_logger

class MainWindow(QMainWindow):
    """Main application window with all UI components"""
    
    def __init__(self, settings_controller, image_controller, simulation_mode=False):
        """Initialize the main window"""
        super().__init__()
        self.logger = get_logger("main_window")
        self.settings_controller = settings_controller
        self.image_controller = image_controller
        self.simulation_mode = simulation_mode
        
        # Initialize hardware controller if not in simulation mode
        self.hardware_controller = HardwareController(simulation_mode=simulation_mode)
        
        # UI state
        self.alarm_active = False
        self.flash_state = False
        self.settings_unlocked = False
        
        # Set up UI
        self.setup_ui()
        
        # Connect signals
        self.connect_signals()
        
        # Initial update
        self.update_ui_from_settings()
        
        self.logger.info("Main window initialized")
    
    def setup_ui(self):
        """Set up the user interface"""
        self.setWindowTitle("Defect Detection System")
        self.setGeometry(100, 100, 1280, 720)
        
        # Create main layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setSpacing(10)
        
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
        main_content_layout.setSpacing(15)
        
        # Create image panels row
        panels_layout = QHBoxLayout()
        panels_layout.setSpacing(20)
        
        # Create bad image panel
        self.bad_panel = BadImagePanel(self.image_controller)
        self.bad_panel.image_clicked.connect(self.show_bad_photo_preview)
        
        # Create good image panel
        self.good_panel = GoodImagePanel(self.image_controller)
        self.good_panel.image_clicked.connect(self.show_good_photo_preview)
        
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
        parent_layout.addLayout(main_content_layout, 1)
    
    def create_control_buttons(self, parent_layout):
        """Create control buttons for the application"""
        # Button row
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(15)
        
        # Login/logout button
        self.login_button = QPushButton("Login")
        self.login_button.setFixedHeight(50)
        self.login_button.clicked.connect(self.toggle_login_logout)
        buttons_layout.addWidget(self.login_button)
        
        # Settings button
        self.settings_button = QPushButton("Settings")
        self.settings_button.setFixedHeight(50)
        self.settings_button.clicked.connect(self.open_settings)
        self.settings_button.setProperty("warning", True)  # For stylesheet
        self.settings_button.hide()  # Hidden until login
        buttons_layout.addWidget(self.settings_button)
        
        # Reset counters button
        self.reset_counter_button = QPushButton("Reset All Counters")
        self.reset_counter_button.setFixedHeight(50)
        self.reset_counter_button.clicked.connect(self.reset_counters)
        self.reset_counter_button.setProperty("caution", True)  # For stylesheet
        self.reset_counter_button.hide()  # Hidden until login
        buttons_layout.addWidget(self.reset_counter_button)
        
        # Add trigger button only in simulation mode
        if self.simulation_mode:
            self.trigger_button = QPushButton("Simulate Trigger")
            self.trigger_button.setFixedHeight(50)
            self.trigger_button.clicked.connect(self.simulate_trigger)
            self.trigger_button.setProperty("action", True)  # For stylesheet
            buttons_layout.addWidget(self.trigger_button)
        
        parent_layout.addLayout(buttons_layout)
        
        # Alarm reset button (separate row)
        self.reset_alarm_button = QPushButton("Reset Alarm")
        self.reset_alarm_button.setFixedHeight(50)
        self.reset_alarm_button.clicked.connect(self.reset_alarm)
        self.reset_alarm_button.setProperty("highlight", True)  # For stylesheet
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
        
        # Add monitor timer if not in simulation mode
        if not self.simulation_mode:
            self.monitor_timer = QTimer(self)
            self.monitor_timer.timeout.connect(self.monitor_system)
            self.monitor_timer.start(5000)  # Check system every 5 seconds
    
    def connect_signals(self):
        """Connect signals from controllers to UI updates"""
        # Connect settings controller signals
        self.settings_controller.settings_updated.connect(self.handle_settings_updated)
        
        # Connect image controller signals
        self.image_controller.image_updated.connect(self.handle_image_updated)
    
    @pyqtSlot(dict)
    def handle_settings_updated(self, settings):
        """Handle updates from the settings controller"""
        # Update metrics display
        self.metrics_panel.update_display(settings)
        
        # Check alarm state
        alarm_active = settings.get("alarm", 0) > 0
        if alarm_active != self.alarm_active:
            self.set_alarm_state(alarm_active)
    
    @pyqtSlot(str, str)
    def handle_image_updated(self, image_type, image_path):
        """Handle image updates from the image controller"""
        if image_type == "normal":
            self.good_panel.update_image(image_path)
        elif image_type == "anomaly":
            self.bad_panel.update_image(image_path)
    
    def update_ui_from_settings(self):
        """Update UI components based on current settings"""
        settings = self.settings_controller.get_settings()
        
        # Update image panels from settings (in case they changed elsewhere)
        self.bad_panel.update_from_settings(settings)
        self.good_panel.update_from_settings(settings)
        
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
    
    def monitor_system(self):
        """Monitor system status and check for hardware issues"""
        # This is a simplified placeholder - a real implementation would check hardware status
        # through the hardware controller and take appropriate action
        pass
    
    def toggle_login_logout(self):
        """Toggle between logged in and logged out states"""
        if self.settings_unlocked:
            # Log out
            self.settings_unlocked = False
            self.settings_button.hide()
            self.reset_counter_button.hide()
            self.login_button.setText("Login")
            QMessageBox.information(self, "Logged Out", "You have been logged out.")
        else:
            # Show login dialog
            dialog = LoginDialog(self)
            dialog.login_attempt.connect(self.handle_login_attempt)
            dialog.exec_()
    
    def handle_login_attempt(self, password):
        """Handle login attempt from the login dialog"""
        result = self.settings_controller.authenticate(password)
        
        if result:
            self.settings_unlocked = True
            self.settings_button.show()
            self.reset_counter_button.show()
            self.login_button.setText("Logout")
            QMessageBox.information(self, "Success", "Access to settings unlocked.")
        else:
            QMessageBox.warning(self, "Error", "Invalid code.")
    
    def open_settings(self):
        """Open the settings dialog"""
        if not self.settings_unlocked:
            QMessageBox.warning(self, "Access Denied", "You must log in to access settings.")
            return
        
        dialog = SettingsDialog(self.settings_controller, self)
        dialog.exec_()
    
    def reset_counters(self):
        """Reset all counters"""
        if not self.settings_unlocked:
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
    
    def show_bad_photo_preview(self, image_path):
        """Show a preview dialog for a bad photo"""
        if image_path and os.path.exists(image_path):
            dialog = ImagePreviewDialog(self.image_controller, image_path, "anomaly", self)
            dialog.exec_()
    
    def show_good_photo_preview(self, image_path):
        """Show a preview dialog for a good photo"""
        if image_path and os.path.exists(image_path):
            dialog = ImagePreviewDialog(self.image_controller, image_path, "normal", self)
            dialog.exec_()
    
    def simulate_trigger(self):
        """Simulate a hardware trigger for testing (only in simulation mode)"""
        if not self.simulation_mode:
            return
            
        # Get a simulated image path
        temp_image_path = self.hardware_controller.capture_image("/tmp/simulated_capture.jpg")
        if temp_image_path:
            self.logger.info(f"Simulated capture: {temp_image_path}")
            
            # Simulate analysis (for demo purposes only)
            import random
            is_defect = random.random() < 0.3  # 30% chance of defect
            error_value = 0.9 if is_defect else 0.5
            
            # Update settings based on analysis
            settings = self.settings_controller.get_settings()
            
            if is_defect:
                # Path for bad images
                from config import config
                dest_path = os.path.join(config.PATHS["ANOMALY_DIR"], f"simulated_{int(time.time())}.jpg")
                self.image_controller.move_image(temp_image_path, dest_path)
                
                # Update settings
                self.settings_controller.update_after_analysis(
                    is_defect=True,
                    reconstruction_error=error_value,
                    image_path=dest_path
                )
            else:
                # Path for good images
                from config import config
                dest_path = os.path.join(config.PATHS["NORMAL_DIR"], f"simulated_{int(time.time())}.jpg")
                self.image_controller.move_image(temp_image_path, dest_path)
                
                # Update settings
                self.settings_controller.update_after_analysis(
                    is_defect=False,
                    reconstruction_error=error_value,
                    image_path=dest_path
                )
            
            # Update UI from updated settings
            self.update_ui_from_settings()
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Clean up resources
        if hasattr(self, 'hardware_controller'):
            self.hardware_controller.cleanup()
            
        # Accept the close event
        event.accept()
