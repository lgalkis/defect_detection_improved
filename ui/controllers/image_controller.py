#!/usr/bin/env python3
"""
Image Controller for Defect Detection System
Manages image loading, display, and processing
"""

import os
import shutil
from PyQt5.QtCore import QObject, pyqtSignal, QThread
from PyQt5.QtGui import QPixmap

# Import models
from models.image_model import ImageModel
from utils.logger import get_logger
from utils.image_utils import resize_image, annotate_image

class ImageWorker(QThread):
    """Worker thread for image operations to prevent UI freezing."""
    
    # Define signals
    finished = pyqtSignal(str, str)  # (operation, result)
    error = pyqtSignal(str, str)     # (operation, error message)
    
    def __init__(self, operation, image_path, output_path=None, *args, **kwargs):
        """Initialize the image worker."""
        super().__init__()
        self.operation = operation  # 'load', 'move', 'resize', etc.
        self.image_path = image_path
        self.output_path = output_path
        self.args = args
        self.kwargs = kwargs
        self.logger = get_logger("image_worker")
    
    def run(self):
        """Execute the image operation in a separate thread."""
        try:
            result = None
            
            if self.operation == 'load':
                # Just verify the image exists and can be loaded
                pixmap = QPixmap(self.image_path)
                if not pixmap.isNull():
                    result = self.image_path
            
            elif self.operation == 'move':
                # Move/copy the image to a new location
                if os.path.exists(self.image_path):
                    os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
                    shutil.copy2(self.image_path, self.output_path)
                    result = self.output_path
            
            elif self.operation == 'resize':
                # Resize the image
                size = self.kwargs.get('size', (640, 480))
                image_type = self.kwargs.get('image_type', 'good')
                result = resize_image(self.image_path, self.output_path, size, image_type)
            
            elif self.operation == 'annotate':
                # Annotate the image with text or metrics
                text = self.kwargs.get('text', None)
                metrics = self.kwargs.get('metrics', None)
                result = annotate_image(self.image_path, self.output_path, text, metrics)
            
            # Send the result back
            self.finished.emit(self.operation, result)
            
        except Exception as e:
            self.logger.error(f"Error in {self.operation} operation: {e}")
            self.error.emit(self.operation, str(e))

class ImageController(QObject):
    """
    Controller for managing images in the defect detection system.
    Handles image loading, caching, and processing.
    """
    
    # Define signals
    image_updated = pyqtSignal(str, str)  # (image_type, image_path)
    operation_finished = pyqtSignal(str, bool)  # (operation, success)
    
    def __init__(self):
        """Initialize the image controller."""
        super().__init__()
        self.logger = get_logger("image_controller")
        self.model = ImageModel()
        self.active_workers = []
        
        # Cache for loaded QPixmaps
        self.pixmap_cache = {}
    
    def load_image(self, image_path, use_worker=False):
        """
        Load an image as a QPixmap.
        
        Args:
            image_path: Path to the image file
            use_worker: Whether to use a background thread
            
        Returns:
            QPixmap object or None if loading fails or thread is used
        """
        if not image_path or not os.path.exists(image_path):
            return None
            
        # Check cache first
        if image_path in self.pixmap_cache:
            return self.pixmap_cache[image_path]
        
        if use_worker:
            # Create and start worker thread
            worker = ImageWorker('load', image_path)
            worker.finished.connect(self._handle_worker_finished)
            worker.error.connect(self._handle_worker_error)
            worker.start()
            
            # Keep reference to prevent garbage collection
            self.active_workers.append(worker)
            return None
        else:
            try:
                # Load image directly
                pixmap = QPixmap(image_path)
                if pixmap.isNull():
                    self.logger.error(f"Failed to load image: {image_path}")
                    return None
                
                # Cache the pixmap
                self.pixmap_cache[image_path] = pixmap
                return pixmap
            except Exception as e:
                self.logger.error(f"Error loading image {image_path}: {e}")
                return None
    
    def move_image(self, source_path, dest_path, use_worker=False):
        """
        Move or copy an image to a new location.
        
        Args:
            source_path: Path to the source image
            dest_path: Destination path
            use_worker: Whether to use a background thread
            
        Returns:
            Destination path or None if failed/using worker
        """
        if not source_path or not os.path.exists(source_path):
            return None
            
        # Make sure destination directory exists
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        if use_worker:
            # Create and start worker thread
            worker = ImageWorker('move', source_path, dest_path)
            worker.finished.connect(self._handle_worker_finished)
            worker.error.connect(self._handle_worker_error)
            worker.start()
            
            # Keep reference to prevent garbage collection
            self.active_workers.append(worker)
            return None
        else:
            try:
                # Copy the file
                shutil.copy2(source_path, dest_path)
                
                # Update type (good/bad) based on destination path
                from config import config
                if config.PATHS["NORMAL_DIR"] in dest_path:
                    self.image_updated.emit("normal", dest_path)
                elif config.PATHS["ANOMALY_DIR"] in dest_path:
                    self.image_updated.emit("anomaly", dest_path)
                
                return dest_path
            except Exception as e:
                self.logger.error(f"Error moving image {source_path} to {dest_path}: {e}")
                return None
    
    def resize_image(self, source_path, output_path=None, size=None, image_type="good", use_worker=False):
        """
        Resize an image to the specified dimensions.
        
        Args:
            source_path: Path to the source image
            output_path: Output path (defaults to source_path)
            size: Tuple of (width, height)
            image_type: 'good' or 'bad' for quality settings
            use_worker: Whether to use a background thread
            
        Returns:
            Output path or None if failed/using worker
        """
        if not source_path or not os.path.exists(source_path):
            return None
            
        # Default output path to source path
        if output_path is None:
            output_path = source_path
            
        # Default size based on image type
        if size is None:
            from config import config
            size = (
                config.INFERENCE["GOOD_IMAGE_RESIZE"] if image_type == "good"
                else config.INFERENCE["BAD_IMAGE_RESIZE"]
            )
        
        if use_worker:
            # Create and start worker thread
            worker = ImageWorker('resize', source_path, output_path, size=size, image_type=image_type)
            worker.finished.connect(self._handle_worker_finished)
            worker.error.connect(self._handle_worker_error)
            worker.start()
            
            # Keep reference to prevent garbage collection
            self.active_workers.append(worker)
            return None
        else:
            try:
                # Resize the image
                result = resize_image(source_path, output_path, size, image_type)
                
                # Clear cache entry if it exists
                if source_path in self.pixmap_cache:
                    del self.pixmap_cache[source_path]
                if output_path in self.pixmap_cache:
                    del self.pixmap_cache[output_path]
                
                return result
            except Exception as e:
                self.logger.error(f"Error resizing image {source_path}: {e}")
                return None
    
    def annotate_image(self, source_path, output_path=None, text=None, metrics=None, use_worker=False):
        """
        Annotate an image with text and metrics.
        
        Args:
            source_path: Path to the source image
            output_path: Output path
            text: Text to add to the image
            metrics: Dictionary of metrics to display
            use_worker: Whether to use a background thread
            
        Returns:
            Output path or None if failed/using worker
        """
        if not source_path or not os.path.exists(source_path):
            return None
            
        # Default output path
        if output_path is None:
            base_name, ext = os.path.splitext(source_path)
            output_path = f"{base_name}_annotated{ext}"
        
        if use_worker:
            # Create and start worker thread
            worker = ImageWorker('annotate', source_path, output_path, text=text, metrics=metrics)
            worker.finished.connect(self._handle_worker_finished)
            worker.error.connect(self._handle_worker_error)
            worker.start()
            
            # Keep reference to prevent garbage collection
            self.active_workers.append(worker)
            return None
        else:
            try:
                # Annotate the image
                result = annotate_image(source_path, output_path, text, metrics)
                
                # Clear cache entry if it exists
                if output_path in self.pixmap_cache:
                    del self.pixmap_cache[output_path]
                
                return result
            except Exception as e:
                self.logger.error(f"Error annotating image {source_path}: {e}")
                return None
    
    def generate_heatmap(self, image_path, output_path=None):
        """
        Generate a heatmap visualization for an image.
        
        Args:
            image_path: Path to the image
            output_path: Output path for the heatmap
            
        Returns:
            Output path or None if failed
        """
        # This is a placeholder for heatmap generation
        # In a real implementation, this would use model inference to create a heatmap
        
        if not image_path or not os.path.exists(image_path):
            return None
            
        # Default output path
        if output_path is None:
            base_name, ext = os.path.splitext(image_path)
            output_path = f"{base_name}_heatmap.png"
        
        # In a real system, we would generate a heatmap here
        # For now, just create a simple colored version of the image as a placeholder
        try:
            from PIL import Image, ImageOps
            
            # Open the image
            img = Image.open(image_path)
            
            # Create a "heatmap" by using false color
            heatmap = ImageOps.colorize(
                ImageOps.grayscale(img),
                "#0000ff",  # Blue for "cold" areas
                "#ff0000"   # Red for "hot" areas
            )
            
            # Save the heatmap
            heatmap.save(output_path)
            
            self.logger.info(f"Generated heatmap: {output_path}")
            
            # Emit signal that a new image is available
            self.image_updated.emit("heatmap", output_path)
            
            return output_path
        except Exception as e:
            self.logger.error(f"Error generating heatmap: {e}")
            return None
    
    def get_recent_images(self, image_type, count=10):
        """
        Get a list of recent images of the specified type.
        
        Args:
            image_type: 'normal' or 'anomaly'
            count: Maximum number of images to return
            
        Returns:
            List of image paths
        """
        try:
            # Determine image directory
            from config import config
            if image_type == "normal":
                image_dir = config.PATHS["NORMAL_DIR"]
            elif image_type == "anomaly":
                image_dir = config.PATHS["ANOMALY_DIR"]
            else:
                return []
                
            # Check if directory exists
            if not os.path.exists(image_dir):
                return []
            
            # Get list of image files
            image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Sort by modification time (newest first)
            image_files.sort(key=os.path.getmtime, reverse=True)
            
            # Return up to 'count' images
            return image_files[:count]
        except Exception as e:
            self.logger.error(f"Error getting recent images: {e}")
            return []
    
    def get_image_info(self, image_path):
        """
        Get information about an image.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Dictionary with image information
        """
        if not image_path or not os.path.exists(image_path):
            return {}
            
        try:
            # Get basic file info
            file_info = {
                "filename": os.path.basename(image_path),
                "full_path": os.path.abspath(image_path),
                "size_bytes": os.path.getsize(image_path),
                "modified_time": os.path.getmtime(image_path)
            }
            
            # Get image dimensions if possible
            try:
                from PIL import Image
                with Image.open(image_path) as img:
                    file_info["width"] = img.width
                    file_info["height"] = img.height
                    file_info["format"] = img.format
                    file_info["mode"] = img.mode
            except:
                # If PIL fails, try using QPixmap
                pixmap = QPixmap(image_path)
                if not pixmap.isNull():
                    file_info["width"] = pixmap.width()
                    file_info["height"] = pixmap.height()
            
            # Determine image type based on path
            from config import config
            if config.PATHS["NORMAL_DIR"] in image_path:
                file_info["type"] = "normal"
            elif config.PATHS["ANOMALY_DIR"] in image_path:
                file_info["type"] = "anomaly"
            else:
                file_info["type"] = "unknown"
            
            return file_info
        except Exception as e:
            self.logger.error(f"Error getting image info for {image_path}: {e}")
            return {"error": str(e)}
    
    def clear_cache(self):
        """Clear the image cache to free memory."""
        self.pixmap_cache.clear()
        self.logger.debug("Image cache cleared")
    
    def _handle_worker_finished(self, operation, result):
        """Handle completion of worker thread."""
        # Find and remove the worker
        for worker in list(self.active_workers):
            if worker.operation == operation and worker.image_path == worker.image_path:
                self.active_workers.remove(worker)
                worker.deleteLater()
                break
        
        # Notify success
        self.operation_finished.emit(operation, result is not None)
        
        # Update UI based on operation
        if operation == 'move':
            # Determine image type based on destination path
            from config import config
            if config.PATHS["NORMAL_DIR"] in result:
                self.image_updated.emit("normal", result)
            elif config.PATHS["ANOMALY_DIR"] in result:
                self.image_updated.emit("anomaly", result)
    
    def _handle_worker_error(self, operation, error_message):
        """Handle error in worker thread."""
        self.logger.error(f"Error in image worker ({operation}): {error_message}")
        
        # Find and remove the worker
        for worker in list(self.active_workers):
            if worker.operation == operation and worker.image_path == worker.image_path:
                self.active_workers.remove(worker)
                worker.deleteLater()
                break
        
        # Notify failure
        self.operation_finished.emit(operation, False)
