#!/usr/bin/env python3
"""
Image Panels for Defect Detection System
Provides image display components for normal and defective images
"""

import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QSizePolicy, QFrame
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QPixmap, QResizeEvent

from utils.logger import get_logger

class ImagePanel(QWidget):
    """Base class for image display panels."""
    
    # Signal when the image is clicked
    image_clicked = pyqtSignal(str)  # (image_path)
    
    def __init__(self, image_controller, panel_title="Image Preview", parent=None):
        """Initialize the image panel."""
        super().__init__(parent)
        self.logger = get_logger("image_panel")
        self.image_controller = image_controller
        self.panel_title = panel_title
        self.current_image_path = ""
        
        # Set up UI
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the user interface."""
        # Create main layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setSpacing(10)
        
        # Create title label
        self.title_label = QLabel(self.panel_title)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        self.main_layout.addWidget(self.title_label)
        
        # Create image frame
        self.image_frame = QFrame()
        self.image_frame.setFrameShape(QFrame.StyledPanel)
        self.image_frame.setFrameShadow(QFrame.Raised)
        self.image_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Create image label inside frame
        frame_layout = QVBoxLayout(self.image_frame)
        frame_layout.setContentsMargins(10, 10, 10, 10)
        
        self.image_label = QLabel("No image available")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("color: #888888; font-size: 16px;")
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setMinimumSize(320, 240)
        
        frame_layout.addWidget(self.image_label)
        
        # Add frame to main layout
        self.main_layout.addWidget(self.image_frame, 1)
        
        # Create info label
        self.info_label = QLabel("")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("font-size: 14px;")
        self.main_layout.addWidget(self.info_label)
        
        # Connect mouse events
        self.image_label.mousePressEvent = self.on_image_clicked
        self.image_frame.mousePressEvent = self.on_image_clicked
    
    def on_image_clicked(self, event):
        """Handle mouse click on the image."""
        if self.current_image_path:
            self.image_clicked.emit(self.current_image_path)
    
    def update_image(self, image_path):
        """
        Update the displayed image.
        
        Args:
            image_path: Path to the image file
        """
        if not image_path or not os.path.exists(image_path):
            self.clear_image()
            return
        
        # Update current image path
        self.current_image_path = image_path
        
        # Load image with controller
        pixmap = self.image_controller.load_image(image_path)
        
        if pixmap and not pixmap.isNull():
            # Scale pixmap to fit label while preserving aspect ratio
            self.display_image(pixmap)
            
            # Update info label with file information
            filename = os.path.basename(image_path)
            file_size = os.path.getsize(image_path) / 1024  # Size in KB
            
            # Try to get image dimensions
            width = pixmap.width()
            height = pixmap.height()
            
            self.info_label.setText(f"{filename} ({width}x{height}, {file_size:.1f} KB)")
        else:
            self.clear_image()
    
    def display_image(self, pixmap):
        """
        Display and scale an image pixmap.
        
        Args:
            pixmap: QPixmap to display
        """
        if pixmap and not pixmap.isNull():
            # Scale pixmap to fit label while preserving aspect ratio
            label_size = self.image_label.size()
            if label_size.width() > 0 and label_size.height() > 0:
                scaled_pixmap = pixmap.scaled(
                    label_size,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.image_label.setPixmap(scaled_pixmap)
                self.image_label.setAlignment(Qt.AlignCenter)
            else:
                # If label size is not valid yet, just set the pixmap directly
                self.image_label.setPixmap(pixmap)
                self.image_label.setAlignment(Qt.AlignCenter)
    
    def clear_image(self):
        """Clear the displayed image."""
        self.current_image_path = ""
        self.image_label.clear()
        self.image_label.setText("No image available")
        self.info_label.setText("")
    
    def update_from_settings(self, settings):
        """
        Update the panel based on settings.
        
        Args:
            settings: Settings dictionary
        """
        # This method should be implemented by subclasses
        pass
    
    def resizeEvent(self, event: QResizeEvent):
        """Handle resize events to rescale the image."""
        super().resizeEvent(event)
        
        # If we have a current image, reload and rescale it
        if self.current_image_path and os.path.exists(self.current_image_path):
            pixmap = self.image_controller.load_image(self.current_image_path)
            if pixmap and not pixmap.isNull():
                self.display_image(pixmap)

class GoodImagePanel(ImagePanel):
    """Panel for displaying normal (non-defective) images."""
    
    def __init__(self, image_controller, parent=None):
        """Initialize the good image panel."""
        super().__init__(image_controller, "Good Photo Preview", parent)
        
        # Customize frame style (green border)
        self.image_frame.setStyleSheet("""
            QFrame {
                border: 5px solid #00aa00;
                border-radius: 15px;
                background-color: #333333;
                padding: 10px;
            }
        """)
        
        # Initialize with the latest good image
        self.load_latest_image()
    
    def load_latest_image(self):
        """Load the latest good image."""
        latest_image = self.image_controller.get_recent_images("normal", 1)
        if latest_image:
            self.update_image(latest_image[0])
    
    def update_from_settings(self, settings):
        """Update the panel from settings."""
        last_good_photo = settings.get("last_good_photo", "")
        if last_good_photo and os.path.exists(last_good_photo):
            self.update_image(last_good_photo)

class BadImagePanel(ImagePanel):
    """Panel for displaying defective (anomaly) images."""
    
    def __init__(self, image_controller, parent=None):
        """Initialize the bad image panel."""
        super().__init__(image_controller, "Bad Photo Preview", parent)
        
        # Customize frame style (red border)
        self.image_frame.setStyleSheet("""
            QFrame {
                border: 5px solid #aa0000;
                border-radius: 15px;
                background-color: #333333;
                padding: 10px;
            }
        """)
        
        # Initialize with the latest bad image
        self.load_latest_image()
    
    def load_latest_image(self):
        """Load the latest bad image."""
        latest_image = self.image_controller.get_recent_images("anomaly", 1)
        if latest_image:
            self.update_image(latest_image[0])
    
    def update_from_settings(self, settings):
        """Update the panel from settings."""
        last_bad_photo = settings.get("last_bad_photo", "")
        if last_bad_photo and os.path.exists(last_bad_photo):
            self.update_image(last_bad_photo)
