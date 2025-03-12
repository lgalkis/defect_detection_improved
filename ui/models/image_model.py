#!/usr/bin/env python3
"""
Image Model for Defect Detection System
Manages image data storage and retrieval
"""

import os
import glob
import time
from PyQt5.QtCore import QObject, pyqtSignal
from utils.logger import get_logger

# Import centralized configuration
from config import config

class ImageModel(QObject):
    """
    Data model for managing images in the defect detection system.
    """
    
    # Define signals for notifying views of data changes
    image_added = pyqtSignal(str, str)  # (image_type, image_path)
    image_deleted = pyqtSignal(str, str)  # (image_type, image_path)
    
    def __init__(self):
        """Initialize the image model."""
        super().__init__()
        self.logger = get_logger("image_model")
        
        # Ensure image directories exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure that all required image directories exist."""
        try:
            # Make sure normal and anomaly directories exist
            os.makedirs(config.PATHS["NORMAL_DIR"], exist_ok=True)
            os.makedirs(config.PATHS["ANOMALY_DIR"], exist_ok=True)
            
            # Create backup directory if needed
            if "BACKUP_DIR" in config.PATHS:
                os.makedirs(config.PATHS["BACKUP_DIR"], exist_ok=True)
        except Exception as e:
            self.logger.error(f"Error creating image directories: {e}")
    
    def get_image_list(self, image_type="all", max_count=100, sort_by="time"):
        """
        Get a list of image paths of the specified type.
        
        Args:
            image_type: "normal", "anomaly", or "all"
            max_count: Maximum number of images to return
            sort_by: "time" (newest first) or "name"
            
        Returns:
            List of image paths
        """
        image_list = []
        
        try:
            # Get paths based on image type
            if image_type in ["normal", "all"]:
                normal_dir = config.PATHS["NORMAL_DIR"]
                if os.path.exists(normal_dir):
                    normal_images = glob.glob(os.path.join(normal_dir, "*.jpg"))
                    normal_images.extend(glob.glob(os.path.join(normal_dir, "*.jpeg")))
                    normal_images.extend(glob.glob(os.path.join(normal_dir, "*.png")))
                    image_list.extend(normal_images)
            
            if image_type in ["anomaly", "all"]:
                anomaly_dir = config.PATHS["ANOMALY_DIR"]
                if os.path.exists(anomaly_dir):
                    anomaly_images = glob.glob(os.path.join(anomaly_dir, "*.jpg"))
                    anomaly_images.extend(glob.glob(os.path.join(anomaly_dir, "*.jpeg")))
                    anomaly_images.extend(glob.glob(os.path.join(anomaly_dir, "*.png")))
                    image_list.extend(anomaly_images)
            
            # Sort the images
            if sort_by == "time":
                # Sort by modification time (newest first)
                image_list.sort(key=os.path.getmtime, reverse=True)
            else:
                # Sort by name
                image_list.sort()
            
            # Limit to max_count
            return image_list[:max_count]
        except Exception as e:
            self.logger.error(f"Error getting image list: {e}")
            return []
    
    def get_latest_image(self, image_type):
        """
        Get the most recent image of the specified type.
        
        Args:
            image_type: "normal" or "anomaly"
            
        Returns:
            Path to the most recent image or empty string if none found
        """
        try:
            # Get image directory based on type
            if image_type == "normal":
                image_dir = config.PATHS["NORMAL_DIR"]
            elif image_type == "anomaly":
                image_dir = config.PATHS["ANOMALY_DIR"]
            else:
                return ""
            
            if not os.path.exists(image_dir):
                return ""
            
            # Get all image files in the directory
            image_files = []
            for ext in [".jpg", ".jpeg", ".png"]:
                image_files.extend(glob.glob(os.path.join(image_dir, f"*{ext}")))
            
            if not image_files:
                return ""
            
            # Sort by modification time (newest first)
            image_files.sort(key=os.path.getmtime, reverse=True)
            
            # Return the most recent image
            return image_files[0]
        except Exception as e:
            self.logger.error(f"Error getting latest {image_type} image: {e}")
            return ""
    
    def add_image(self, image_path, image_type):
        """
        Add an image to the appropriate directory.
        
        Args:
            image_path: Path to the image
            image_type: "normal" or "anomaly"
            
        Returns:
            New path of the image
        """
        try:
            # Validate inputs
            if not os.path.exists(image_path):
                self.logger.error(f"Image does not exist: {image_path}")
                return ""
            
            if image_type not in ["normal", "anomaly"]:
                self.logger.error(f"Invalid image type: {image_type}")
                return ""
            
            # Get destination directory
            if image_type == "normal":
                dest_dir = config.PATHS["NORMAL_DIR"]
            else:
                dest_dir = config.PATHS["ANOMALY_DIR"]
            
            # Create destination directory if it doesn't exist
            os.makedirs(dest_dir, exist_ok=True)
            
            # Generate a unique filename based on timestamp
            filename = f"{int(time.time())}_{os.path.basename(image_path)}"
            dest_path = os.path.join(dest_dir, filename)
            
            # Copy the image
            import shutil
            shutil.copy2(image_path, dest_path)
            
            # Emit signal
            self.image_added.emit(image_type, dest_path)
            
            return dest_path
        except Exception as e:
            self.logger.error(f"Error adding image: {e}")
            return ""
    
    def delete_image(self, image_path):
        """
        Delete an image.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Boolean indicating success
        """
        try:
            if not os.path.exists(image_path):
                self.logger.error(f"Image does not exist: {image_path}")
                return False
            
            # Determine image type
            if config.PATHS["NORMAL_DIR"] in image_path:
                image_type = "normal"
            elif config.PATHS["ANOMALY_DIR"] in image_path:
                image_type = "anomaly"
            else:
                image_type = "unknown"
            
            # Delete the image
            os.remove(image_path)
            
            # Emit signal
            self.image_deleted.emit(image_type, image_path)
            
            return True
        except Exception as e:
            self.logger.error(f"Error deleting image: {e}")
            return False
    
    def backup_images(self, image_type="all"):
        """
        Create a backup of images.
        
        Args:
            image_type: "normal", "anomaly", or "all"
            
        Returns:
            Tuple of (success, backup_dir)
        """
        try:
            # Create backup directory with timestamp
            backup_root = config.PATHS.get("BACKUP_DIR", "backups")
            os.makedirs(backup_root, exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_dir = os.path.join(backup_root, f"backup_{timestamp}")
            os.makedirs(backup_dir, exist_ok=True)
            
            # Copy images based on type
            copied_count = 0
            
            if image_type in ["normal", "all"]:
                normal_dir = config.PATHS["NORMAL_DIR"]
                if os.path.exists(normal_dir):
                    # Create normal directory in backup
                    normal_backup_dir = os.path.join(backup_dir, "Normal")
                    os.makedirs(normal_backup_dir, exist_ok=True)
                    
                    # Copy normal images
                    for img in glob.glob(os.path.join(normal_dir, "*.*")):
                        if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                            import shutil
                            dest = os.path.join(normal_backup_dir, os.path.basename(img))
                            shutil.copy2(img, dest)
                            copied_count += 1
            
            if image_type in ["anomaly", "all"]:
                anomaly_dir = config.PATHS["ANOMALY_DIR"]
                if os.path.exists(anomaly_dir):
                    # Create anomaly directory in backup
                    anomaly_backup_dir = os.path.join(backup_dir, "Anomaly")
                    os.makedirs(anomaly_backup_dir, exist_ok=True)
                    
                    # Copy anomaly images
                    for img in glob.glob(os.path.join(anomaly_dir, "*.*")):
                        if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                            import shutil
                            dest = os.path.join(anomaly_backup_dir, os.path.basename(img))
                            shutil.copy2(img, dest)
                            copied_count += 1
            
            if copied_count == 0:
                self.logger.warning(f"No images found to backup")
                return False, None
            
            self.logger.info(f"Backed up {copied_count} images to {backup_dir}")
            return True, backup_dir
        except Exception as e:
            self.logger.error(f"Error backing up images: {e}")
            return False, None
    
    def clear_images(self, image_type="all"):
        """
        Delete all images of the specified type.
        
        Args:
            image_type: "normal", "anomaly", or "all"
            
        Returns:
            Tuple of (success, count_deleted)
        """
        try:
            # Perform backup first
            backup_success, _ = self.backup_images(image_type)
            if not backup_success:
                self.logger.warning("Failed to backup images before clearing")
            
            deleted_count = 0
            
            if image_type in ["normal", "all"]:
                normal_dir = config.PATHS["NORMAL_DIR"]
                if os.path.exists(normal_dir):
                    for img in glob.glob(os.path.join(normal_dir, "*.*")):
                        if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                            os.remove(img)
                            deleted_count += 1
            
            if image_type in ["anomaly", "all"]:
                anomaly_dir = config.PATHS["ANOMALY_DIR"]
                if os.path.exists(anomaly_dir):
                    for img in glob.glob(os.path.join(anomaly_dir, "*.*")):
                        if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                            os.remove(img)
                            deleted_count += 1
            
            self.logger.info(f"Cleared {deleted_count} images")
            return True, deleted_count
        except Exception as e:
            self.logger.error(f"Error clearing images: {e}")
            return False, 0