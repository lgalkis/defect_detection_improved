#!/usr/bin/env python3
"""
Image Preview Dialog for Defect Detection System
Provides a detailed view of images with heatmap capabilities
"""

import os
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QWidget, QSizePolicy, QMessageBox, QFrame
)
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QPixmap

from utils.logger import get_logger

class ImagePreviewDialog(QDialog):
    """Dialog for displaying a full-size image with heatmap generation capability."""
    
    def __init__(self, image_controller, image_path, image_type, parent=None):
        """
        Initialize the image preview dialog.
        
        Args:
            image_controller: Controller for image operations
            image_path: Path to the image file
            image_type: "normal" or "anomaly"
            parent: Parent widget
        """
        super().__init__(parent)
        self.logger = get_logger("preview_dialog")
        self.image_controller = image_controller
        self.image_path = image_path
        self.image_type = image_type
        self.heatmap_path = None
        self.showing_heatmap = False
        
        # Set window properties
        self.setWindowTitle(f"Image Preview: {os.path.basename(image_path)}")
        self.resize(1024, 768)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        
        # Set up UI
        self.setup_ui()
        
        # Display the image
        self.display_image(image_path)
    
    def setup_ui(self):
        """Set up the user interface."""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        
        # Create scrollable image area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(Qt.AlignCenter)
        self.scroll_area.setMinimumSize(800, 600)
        
        # Create image label
        self.image_label = QLabel("Loading image...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Set label as scroll area widget
        self.scroll_area.setWidget(self.image_label)
        
        # Add scroll area to layout
        main_layout.addWidget(self.scroll_area, 1)
        
        # Create info panel
        info_panel = QFrame()
        info_panel.setFrameShape(QFrame.StyledPanel)
        info_panel.setFrameShadow(QFrame.Raised)
        info_panel.setStyleSheet("""
            QFrame {
                background-color: #3c3c3c;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        
        info_layout = QVBoxLayout(info_panel)
        
        # Image info label
        self.info_label = QLabel()
        self.info_label.setStyleSheet("font-size: 14px;")
        info_layout.addWidget(self.info_label)
        
        # Add info panel to layout
        main_layout.addWidget(info_panel)
        
        # Create button row
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        
        # Toggle heatmap button
        self.heatmap_button = QPushButton("Show Heatmap")
        self.heatmap_button.setFixedHeight(40)
        self.heatmap_button.setCheckable(True)
        self.heatmap_button.clicked.connect(self.toggle_heatmap)
        button_layout.addWidget(self.heatmap_button)
        
        # Generate heatmap button
        self.generate_button = QPushButton("Generate Heatmap")
        self.generate_button.setFixedHeight(40)
        self.generate_button.clicked.connect(self.generate_heatmap)
        button_layout.addWidget(self.generate_button)
        
        # Close button
        close_button = QPushButton("Close")
        close_button.setFixedHeight(40)
        close_button.clicked.connect(self.accept)
        button_layout.addWidget(close_button)
        
        # Add button row to layout
        main_layout.addLayout(button_layout)
        
        # Initialize button states
        self.check_heatmap_availability()
    
    def display_image(self, image_path):
        """
        Display an image in the dialog.
        
        Args:
            image_path: Path to the image file
        """
        if not image_path or not os.path.exists(image_path):
            self.image_label.setText("Image not found.")
            return
        
        try:
            # Load the image
            pixmap = QPixmap(image_path)
            if pixmap.isNull():
                self.image_label.setText("Failed to load image.")
                return
            
            # Set the pixmap
            self.image_label.setPixmap(pixmap)
            
            # Update info label with image information
            info = self.image_controller.get_image_info(image_path)
            if info:
                width = info.get("width", 0)
                height = info.get("height", 0)
                size_kb = info.get("size_bytes", 0) / 1024
                
                self.info_label.setText(
                    f"Filename: {os.path.basename(image_path)}\n"
                    f"Size: {width}x{height} pixels ({size_kb:.1f} KB)\n"
                    f"Type: {self.image_type.capitalize()}"
                )
        except Exception as e:
            self.logger.error(f"Error displaying image: {e}")
            self.image_label.setText(f"Error loading image: {e}")
    
    def check_heatmap_availability(self):
        """Check if a heatmap is available for the image."""
        # Check if heatmap file exists
        base_name, ext = os.path.splitext(self.image_path)
        potential_heatmap_path = f"{base_name}_heatmap.png"
        
        if os.path.exists(potential_heatmap_path):
            self.heatmap_path = potential_heatmap_path
            self.heatmap_button.setEnabled(True)
            self.generate_button.setText("Regenerate Heatmap")
        else:
            self.heatmap_path = None
            self.heatmap_button.setEnabled(False)
            self.generate_button.setText("Generate Heatmap")
    
    def toggle_heatmap(self):
        """Toggle between the original image and its heatmap."""
        if not self.heatmap_path or not os.path.exists(self.heatmap_path):
            self.heatmap_button.setChecked(False)
            return
        
        if self.heatmap_button.isChecked():
            # Show heatmap
            self.showing_heatmap = True
            self.display_image(self.heatmap_path)
            self.heatmap_button.setText("Show Original")
        else:
            # Show original
            self.showing_heatmap = False
            self.display_image(self.image_path)
            self.heatmap_button.setText("Show Heatmap")
    
    def generate_heatmap(self):
        """Generate a heatmap for the image."""
        # Show a progress message
        self.image_label.setText("Generating heatmap... Please wait.")
        QApplication.processEvents()  # Update UI
        
        try:
            # Call the image controller to generate the heatmap
            heatmap_path = self.image_controller.generate_heatmap(self.image_path)
            
            if heatmap_path and os.path.exists(heatmap_path):
                self.heatmap_path = heatmap_path
                self.heatmap_button.setEnabled(True)
                
                # Show the heatmap
                self.heatmap_button.setChecked(True)
                self.showing_heatmap = True
                self.display_image(self.heatmap_path)
                self.heatmap_button.setText("Show Original")
                
                QMessageBox.information(
                    self, 
                    "Heatmap Generated", 
                    "Heatmap was generated successfully."
                )
            else:
                self.display_image(self.image_path)
                QMessageBox.warning(
                    self, 
                    "Heatmap Generation Failed", 
                    "Failed to generate heatmap. See log for details."
                )
        except Exception as e:
            self.logger.error(f"Error generating heatmap: {e}")
            self.display_image(self.image_path)
            QMessageBox.critical(
                self, 
                "Error", 
                f"Error generating heatmap: {e}"
            )
