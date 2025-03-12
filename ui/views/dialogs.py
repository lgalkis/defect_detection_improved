#!/usr/bin/env python3
"""
Dialog Windows for Defect Detection System
Provides login, settings, and other dialog windows
"""

import os
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QGridLayout, QMessageBox, QWidget
)
from PyQt5.QtCore import Qt, pyqtSignal

from utils.logger import get_logger

class LoginDialog(QDialog):
    """Dialog for user authentication."""
    
    # Signal when user attempts login
    login_attempt = pyqtSignal(str)
    
    def __init__(self, parent=None):
        """Initialize the login dialog."""
        super().__init__(parent)
        self.logger = get_logger("login_dialog")
        
        # Set window properties
        self.setWindowTitle("Login")
        self.setFixedSize(480, 400)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        
        # Set up UI
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the user interface."""
        dialog_layout = QVBoxLayout(self)
        dialog_layout.setSpacing(15)
        
        # Title
        title_label = QLabel("Enter Code:")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 32px; font-weight: bold;")
        dialog_layout.addWidget(title_label)
        
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
        """Create a numeric keypad for PIN entry."""
        keyboard_widget = QWidget()
        keyboard_layout = QGridLayout(keyboard_widget)
        keyboard_layout.setSpacing(10)
        
        # Define keypad layout
        keys = [
            ["1", "2", "3"],
            ["4", "5", "6"],
            ["7", "8", "9"],
            ["0", "Clear", "OK"]
        ]
        
        # Create buttons for each key
        for row_idx, row in enumerate(keys):
            for col_idx, key_text in enumerate(row):
                button = QPushButton(key_text)
                button.setFixedSize(80, 60)
                
                # Style button based on its function
                if key_text == "Clear":
                    button.setProperty("caution", True)  # For stylesheet
                    button.clicked.connect(self.clear_input)
                elif key_text == "OK":
                    button.setProperty("highlight", True)  # For stylesheet
                    button.clicked.connect(self.handle_login)
                else:
                    # Number button
                    button.setStyleSheet("""
                        QPushButton {
                            background-color: #3c3c3c;
                            color: white;
                            font-size: 20px;
                            border-radius: 5px;
                        }
                        QPushButton:hover {
                            background-color: #4c4c4c;
                        }
                        QPushButton:pressed {
                            background-color: #2c2c2c;
                        }
                    """)
                    button.clicked.connect(lambda _, digit=key_text: self.add_digit(digit))
                
                keyboard_layout.addWidget(button, row_idx, col_idx)
        
        return keyboard_widget
    
    def clear_input(self):
        """Clear the input field."""
        self.code_input.clear()
    
    def add_digit(self, digit):
        """Add a digit to the input field."""
        self.code_input.setText(self.code_input.text() + digit)
    
    def handle_login(self):
        """Handle login attempt."""
        password = self.code_input.text()
        
        if not password:
            QMessageBox.warning(self, "Error", "Please enter a code.")
            return
        
        # Emit signal with entered password
        self.login_attempt.emit(password)
        self.accept()  # Close dialog

class SettingsDialog(QDialog):
    """Dialog for configuring application settings."""
    
    def __init__(self, settings_controller, parent=None):
        """Initialize the settings dialog."""
        super().__init__(parent)
        self.logger = get_logger("settings_dialog")
        self.settings_controller = settings_controller
        
        # Set window properties
        self.setWindowTitle("Settings")
        self.setFixedSize(480, 600)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        
        # Set up UI
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the user interface."""
        dialog_layout = QVBoxLayout(self)
        dialog_layout.setSpacing(15)
        
        # Get current settings
        current_settings = self.settings_controller.get_settings()
        
        # Global Threshold section
        global_threshold_label = QLabel("Set Global Threshold")
        global_threshold_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        global_threshold_label.setAlignment(Qt.AlignCenter)
        dialog_layout.addWidget(global_threshold_label)
        
        self.global_threshold_input = QLineEdit()
        self.global_threshold_input.setText(str(current_settings.get("threshold", 0.85)))
        self.global_threshold_input.setAlignment(Qt.AlignCenter)
        self.global_threshold_input.setStyleSheet("font-size: 20px; padding: 10px;")
        dialog_layout.addWidget(self.global_threshold_input)
        
        # Patch Threshold section
        patch_threshold_label = QLabel("Set Patch Threshold")
        patch_threshold_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        patch_threshold_label.setAlignment(Qt.AlignCenter)
        dialog_layout.addWidget(patch_threshold_label)
        
        self.patch_threshold_input = QLineEdit()
        self.patch_threshold_input.setText(str(current_settings.get("patch_threshold", 0.7)))
        self.patch_threshold_input.setAlignment(Qt.AlignCenter)
        self.patch_threshold_input.setStyleSheet("font-size: 20px; padding: 10px;")
        dialog_layout.addWidget(self.patch_threshold_input)
        
        # Patch Defect Ratio section
        patch_ratio_label = QLabel("Set Patch Defect Ratio")
        patch_ratio_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        patch_ratio_label.setAlignment(Qt.AlignCenter)
        dialog_layout.addWidget(patch_ratio_label)
        
        self.patch_ratio_input = QLineEdit()
        self.patch_ratio_input.setText(str(current_settings.get("patch_defect_ratio", 0.45)))
        self.patch_ratio_input.setAlignment(Qt.AlignCenter)
        self.patch_ratio_input.setStyleSheet("font-size: 20px; padding: 10px;")
        dialog_layout.addWidget(self.patch_ratio_input)
        
        # Field selector buttons
        selector_layout = QHBoxLayout()
        selector_layout.setSpacing(10)
        
        global_btn = QPushButton("Global Threshold")
        global_btn.clicked.connect(lambda: self.set_active_input(self.global_threshold_input))
        selector_layout.addWidget(global_btn)
        
        patch_btn = QPushButton("Patch Threshold")
        patch_btn.clicked.connect(lambda: self.set_active_input(self.patch_threshold_input))
        selector_layout.addWidget(patch_btn)
        
        ratio_btn = QPushButton("Defect Ratio")
        ratio_btn.clicked.connect(lambda: self.set_active_input(self.patch_ratio_input))
        selector_layout.addWidget(ratio_btn)
        
        dialog_layout.addLayout(selector_layout)
        
        # Numeric keyboard
        numeric_keyboard = self.create_numeric_keyboard()
        dialog_layout.addWidget(numeric_keyboard)
        
        # Set default active input
        self.active_input = self.global_threshold_input
        self.set_active_input(self.global_threshold_input)
    
    def create_numeric_keyboard(self):
        """Create a numeric keypad with decimal point for settings input."""
        keyboard_widget = QWidget()
        keyboard_layout = QGridLayout(keyboard_widget)
        keyboard_layout.setSpacing(10)
        
        # Define keypad layout
        keys = [
            ["1", "2", "3"],
            ["4", "5", "6"],
            ["7", "8", "9"],
            ["0", ".", "Clear"]
        ]
        
        # Create buttons for each key
        for row_idx, row in enumerate(keys):
            for col_idx, key_text in enumerate(row):
                button = QPushButton(key_text)
                button.setFixedSize(60, 60)
                
                # Style button based on its function
                if key_text == "Clear":
                    button.setProperty("caution", True)  # For stylesheet
                    button.clicked.connect(self.clear_input)
                elif key_text == ".":
                    button.setStyleSheet("""
                        QPushButton {
                            background-color: #3c3c3c;
                            color: white;
                            font-size: 20px;
                            border-radius: 5px;
                        }
                        QPushButton:hover {
                            background-color: #4c4c4c;
                        }
                        QPushButton:pressed {
                            background-color: #2c2c2c;
                        }
                    """)
                    button.clicked.connect(lambda _, char=key_text: self.add_character(char))
                else:
                    # Number button
                    button.setStyleSheet("""
                        QPushButton {
                            background-color: #3c3c3c;
                            color: white;
                            font-size: 20px;
                            border-radius: 5px;
                        }
                        QPushButton:hover {
                            background-color: #4c4c4c;
                        }
                        QPushButton:pressed {
                            background-color: #2c2c2c;
                        }
                    """)
                    button.clicked.connect(lambda _, char=key_text: self.add_character(char))
                
                keyboard_layout.addWidget(button, row_idx, col_idx)
        
        # Add OK and Cancel buttons in the bottom row
        button_row = QHBoxLayout()
        
        cancel_button = QPushButton("Cancel")
        cancel_button.setFixedHeight(60)
        cancel_button.setProperty("warning", True)  # For stylesheet
        cancel_button.clicked.connect(self.reject)
        button_row.addWidget(cancel_button)
        
        ok_button = QPushButton("Save Settings")
        ok_button.setFixedHeight(60)
        ok_button.setProperty("highlight", True)  # For stylesheet
        ok_button.clicked.connect(self.save_settings)
        button_row.addWidget(ok_button)
        
        # Add button row to keyboard layout
        keyboard_layout.addLayout(button_row, 4, 0, 1, 3)
        
        return keyboard_widget
    
    def set_active_input(self, input_field):
        """Set the active input field and update its styling."""
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
        """Clear the active input field."""
        if self.active_input:
            self.active_input.clear()
    
    def add_character(self, character):
        """Add a character to the active input field."""
        if self.active_input:
            # If character is a decimal point, ensure only one is added
            if character == "." and "." in self.active_input.text():
                return
                
            self.active_input.setText(self.active_input.text() + character)
    
    def save_settings(self):
        """Validate and save the settings."""
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
