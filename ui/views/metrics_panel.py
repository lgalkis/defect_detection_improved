#!/usr/bin/env python3
"""
Metrics Panel for Defect Detection System
Displays system metrics and statistics
"""

import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGridLayout,
    QSizePolicy, QFrame, QGroupBox
)
from PyQt5.QtCore import Qt, pyqtSlot

from utils.logger import get_logger

class MetricsPanel(QWidget):
    """
    Panel for displaying system metrics and statistics.
    Shows thresholds, counters, and detection metrics.
    """
    
    def __init__(self, settings_controller, parent=None):
        """Initialize the metrics panel."""
        super().__init__(parent)
        self.logger = get_logger("metrics_panel")
        self.settings_controller = settings_controller
        
        # Set up UI
        self.setup_ui()
        
        # Initial update
        self.update_display(settings_controller.get_settings())
    
    def setup_ui(self):
        """Set up the user interface."""
        # Create main layout
        main_layout = QVBoxLayout(self)
        
        # Create titles layout for patch and global metrics
        titles_layout = QHBoxLayout()
        
        # Patch analysis title
        patch_title = QLabel("Patch Analysis")
        patch_title.setAlignment(Qt.AlignCenter)
        patch_title.setStyleSheet("font-size: 18px; font-weight: bold;")
        titles_layout.addWidget(patch_title)
        
        # Global metrics title
        global_title = QLabel("Global Metrics")
        global_title.setAlignment(Qt.AlignCenter)
        global_title.setStyleSheet("font-size: 18px; font-weight: bold;")
        titles_layout.addWidget(global_title)
        
        # Add titles to main layout
        main_layout.addLayout(titles_layout)
        
        # Create metrics content layout
        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(20)
        
        # Create patch metrics group
        patch_group = QGroupBox()
        patch_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #555555;
                border-radius: 8px;
                margin-top: 10px;
                background-color: #3c3c3c;
            }
        """)
        patch_layout = QGridLayout(patch_group)
        
        # Create labels for patch metrics
        self.patch_threshold_label = self._create_metrics_label("Patch Threshold:")
        self.patch_ratio_label = self._create_metrics_label("Patch Defect Ratio:")
        self.latest_patch_value_label = self._create_metrics_label("Latest Patch Value:")
        self.latest_defect_ratio_label = self._create_metrics_label("Latest Defect Ratio:")
        
        # Add labels to patch layout
        patch_layout.addWidget(self.patch_threshold_label, 0, 0)
        patch_layout.addWidget(self.patch_ratio_label, 1, 0)
        patch_layout.addWidget(self.latest_patch_value_label, 2, 0)
        patch_layout.addWidget(self.latest_defect_ratio_label, 3, 0)
        
        # Create global metrics group
        global_group = QGroupBox()
        global_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #555555;
                border-radius: 8px;
                margin-top: 10px;
                background-color: #3c3c3c;
            }
        """)
        global_layout = QGridLayout(global_group)
        
        # Create labels for global metrics
        self.threshold_label = self._create_metrics_label("Global Threshold:")
        self.reconstruction_error_label = self._create_metrics_label("Reconstruction Error:")
        self.good_count_label = self._create_metrics_label("Good Count:")
        self.bad_count_label = self._create_metrics_label("Bad Count:")
        self.image_counter_label = self._create_metrics_label("Image Counter:")
        
        # Add labels to global layout
        global_layout.addWidget(self.threshold_label, 0, 0)
        global_layout.addWidget(self.reconstruction_error_label, 1, 0)
        global_layout.addWidget(self.good_count_label, 2, 0)
        global_layout.addWidget(self.bad_count_label, 3, 0)
        global_layout.addWidget(self.image_counter_label, 4, 0)
        
        # Add metric groups to metrics layout
        metrics_layout.addWidget(patch_group)
        metrics_layout.addWidget(global_group)
        
        # Add metrics layout to main layout
        main_layout.addLayout(metrics_layout)
    
    def _create_metrics_label(self, text):
        """Create a label for displaying metrics."""
        label = QLabel(text)
        label.setStyleSheet("font-size: 16px; padding: 5px; color: #ffffff;")
        return label
    
    @pyqtSlot(dict)
    def update_display(self, settings):
        """Update the metrics display with current settings."""
        try:
            # Update patch metrics
            patch_threshold = settings.get("patch_threshold", 0.0)
            self.patch_threshold_label.setText(f"Patch Threshold: {patch_threshold}")
            
            patch_ratio = settings.get("patch_defect_ratio", 0.0)
            self.patch_ratio_label.setText(f"Patch Defect Ratio: {patch_ratio}")
            
            latest_patch_value = settings.get("latest_patch_value", 0.0)
            self.latest_patch_value_label.setText(f"Latest Patch Value: {latest_patch_value:.4f}")
            
            latest_defect_ratio = settings.get("latest_defect_ratio", 0.0)
            self.latest_defect_ratio_label.setText(f"Latest Defect Ratio: {latest_defect_ratio:.4f}")
            
            # Update global metrics
            threshold = settings.get("threshold", 0.0)
            self.threshold_label.setText(f"Global Threshold: {threshold}")
            
            reconstruction_error = settings.get("last_reconstruction_error", 0.0)
            self.reconstruction_error_label.setText(f"Reconstruction Error: {reconstruction_error:.4f}")
            
            good_count = settings.get("good_count", 0)
            self.good_count_label.setText(f"Good Count: {good_count}")
            
            bad_count = settings.get("bad_count", 0)
            self.bad_count_label.setText(f"Bad Count: {bad_count}")
            
            image_counter = settings.get("image_counter", 0)
            self.image_counter_label.setText(f"Image Counter: {image_counter}")
            
            # Update alarm status based on settings
            alarm_state = settings.get("alarm", 0) > 0
            
            # Highlight bad count when alarm is active
            if alarm_state:
                self.bad_count_label.setStyleSheet(
                    "font-size: 16px; padding: 5px; background-color: #aa0000; color: #ffffff; font-weight: bold;"
                )
            else:
                self.bad_count_label.setStyleSheet("font-size: 16px; padding: 5px; color: #ffffff;")
        except Exception as e:
            self.logger.error(f"Error updating metrics display: {e}")
    
    def update_alarm_state(self, alarm_active):
        """Update display based on alarm state."""
        # Highlight bad count when alarm is active
        if alarm_active:
            self.bad_count_label.setStyleSheet(
                "font-size: 16px; padding: 5px; background-color: #aa0000; color: #ffffff; font-weight: bold;"
            )
        else:
            self.bad_count_label.setStyleSheet("font-size: 16px; padding: 5px; color: #ffffff;")