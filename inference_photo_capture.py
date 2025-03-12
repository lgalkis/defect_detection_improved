#!/usr/bin/env python3
"""
Inference and Photo Capture Script for Defect Detection System
Handles automated image capture, analysis, and result recording.
"""

import os
import sys
import time
import csv
import threading
import argparse
import datetime
import signal
from pathlib import Path

# Import core components
from config import config
from models.model_manager import ModelManager
from utils.settings_manager import SettingsManager
from utils.hardware_controller import HardwareController
from utils.image_utils import resize_image, annotate_image, generate_heatmap, backup_images
from utils.logger import setup_logger, log_startup_info

class InferenceSystem:
    """
    Manages automated photo capture and inference process.
    
    This system runs in a loop, waiting for triggers, capturing photos,
    performing analysis, and recording results.
    """
    
    def __init__(self, args):
        """Initialize the inference system."""
        self.args = args
        self.running = False
        
        # Set up logging
        log_level = "DEBUG" if args.debug else "INFO"
        self.logger = setup_logger("inference", config.PATHS["INFERENCE_LOG"], level=log_level)
        log_startup_info(self.logger)
        
        # Initialize components
        self.logger.info("Initializing inference system...")
        
        # Initialize settings manager
        self.settings_manager = SettingsManager()
        
        # Initialize hardware controller
        self.hardware = HardwareController(simulation_mode=args.simulation)
        
        # Initialize model manager
        self.model_manager = ModelManager()
        
        # Load the model
        self.load_model()
        
        # Open and prepare CSV file
        self.prepare_csv()
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Reset counters if requested
        if args.reset or config.INFERENCE["RESET_COUNTER_ON_START"]:
            self.reset_counters()
        
        self.logger.info("Inference system initialized")
    
    def load_model(self):
        """Load the model for inference."""
        try:
            self.logger.info("Loading model...")
            self.model = self.model_manager.load_model()
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            self.model = None
    
    def prepare_csv(self):
        """Prepare the CSV file for recording results."""
        csv_path = config.PATHS["CSV_FILENAME"]
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        # Check if file exists
        file_exists = os.path.exists(csv_path)
        
        try:
            # Open file in append mode
            self.csv_file = open(csv_path, 'a', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            
            # Write header if file is new
            if not file_exists:
                self.csv_writer.writerow([
                    "timestamp", "filename", "global_threshold", "global_error",
                    "patch_threshold", "mean_patch_error", "patch_defect_ratio_threshold",
                    "patch_defect_ratio_value", "is_defect", "detection_method"
                ])
                self.logger.info(f"Created new CSV file: {csv_path}")
            else:
                self.logger.info(f"Appending to existing CSV file: {csv_path}")
        except Exception as e:
            self.logger.error(f"Error preparing CSV file: {e}")
            self.csv_file = None
            self.csv_writer = None
    
    def reset_counters(self):
        """Reset counters in settings file."""
        self.logger.info("Resetting counters...")
        self.settings_manager.reset_counters()
    
    def signal_handler(self, sig, frame):
        """Handle termination signals."""
        self.logger.info(f"Received signal {sig}, shutting down...")
        self.stop()
    
    def start(self):
        """Start the inference system."""
        if self.running:
            self.logger.warning("Inference system is already running")
            return
        
        self.running = True
        self.logger.info("Starting inference loop...")
        
        try:
            self.inference_loop()
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
        except Exception as e:
            self.logger.error(f"Error in inference loop: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the inference system and clean up resources."""
        self.logger.info("Stopping inference system...")
        self.running = False
        
        # Close CSV file
        if hasattr(self, 'csv_file') and self.csv_file:
            self.csv_file.close()
            self.csv_file = None
        
        # Clean up hardware
        if hasattr(self, 'hardware'):
            self.hardware.cleanup()
        
        self.logger.info("Inference system stopped")
    
    def inference_loop(self):
        """Main inference loop."""
        self.logger.info("Inference loop started")
        
        # Initialize counters for backup
        capture_count = 0
        last_backup_time = time.time()
        
        while self.running:
            try:
                # Check for reset signal
                reset_detected, clear_images, reset_csv = self.settings_manager.check_for_reset_signal()
                if reset_detected:
                    self.logger.info("Reset signal detected")
                    if clear_images:
                        self._clear_image_directories()
                    if reset_csv:
                        self._reset_csv_file()
                
                # Wait for trigger
                self.logger.info("Waiting for trigger...")
                if not self.hardware.wait_for_trigger(timeout=1):
                    continue
                
                # Capture image
                self.logger.info("Trigger detected, capturing image...")
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"image_{timestamp}.jpg"
                temp_path = os.path.join("/tmp", filename)
                
                if not self.hardware.capture_image(temp_path):
                    self.logger.error("Failed to capture image")
                    time.sleep(1)
                    continue
                
                # Analyze image
                self.logger.info(f"Analyzing image: {filename}")
                result = self.analyze_image(temp_path, filename)
                
                if result:
                    # Record result
                    self.record_result(result)
                    
                    # Update settings
                    self.update_settings(result)
                    
                    # Indicate if defect detected
                    if result["is_defect"]:
                        self.hardware.indicate_defect()
                    
                    # Increment capture count
                    capture_count += 1
                
                # Check if backup is needed
                current_time = time.time()
                if capture_count >= config.INFERENCE["MAX_IMAGES_BEFORE_BACKUP"] or \
                   (current_time - last_backup_time) > 86400:  # 24 hours
                    
                    self.logger.info("Performing backup of images...")
                    backup_normal = backup_images(config.PATHS["NORMAL_DIR"], 
                                                config.INFERENCE["MAX_IMAGES_BEFORE_BACKUP"])
                    backup_anomaly = backup_images(config.PATHS["ANOMALY_DIR"], 
                                                 config.INFERENCE["MAX_IMAGES_BEFORE_BACKUP"])
                    
                    if backup_normal[0] or backup_anomaly[0]:
                        self.logger.info("Backup completed successfully")
                        capture_count = 0
                        last_backup_time = current_time
                
                # Short delay
                time.sleep(0.5)
            
            except Exception as e:
                self.logger.error(f"Error in inference loop: {e}")
                time.sleep(5)  # Delay before retry
    
    def _clear_image_directories(self):
        """Clear image directories."""
        self.logger.info("Clearing image directories...")
        
        # Clear Normal directory
        normal_dir = config.PATHS["NORMAL_DIR"]
        if os.path.exists(normal_dir):
            for file in os.listdir(normal_dir):
                file_path = os.path.join(normal_dir, file)
                if os.path.isfile(file_path):
                    try:
                        os.unlink(file_path)
                    except Exception as e:
                        self.logger.error(f"Error deleting {file_path}: {e}")
        
        # Clear Anomaly directory
        anomaly_dir = config.PATHS["ANOMALY_DIR"]
        if os.path.exists(anomaly_dir):
            for file in os.listdir(anomaly_dir):
                file_path = os.path.join(anomaly_dir, file)
                if os.path.isfile(file_path):
                    try:
                        os.unlink(file_path)
                    except Exception as e:
                        self.logger.error(f"Error deleting {file_path}: {e}")
        
        self.logger.info("Image directories cleared")
    
    def _reset_csv_file(self):
        """Reset the CSV file."""
        self.logger.info("Resetting CSV file...")
        
        # Close current file if open
        if hasattr(self, 'csv_file') and self.csv_file:
            self.csv_file.close()
            self.csv_file = None
        
        # Backup existing file
        csv_path = config.PATHS["CSV_FILENAME"]
        if os.path.exists(csv_path):
            backup_dir = config.PATHS["BACKUP_CSV_DIR"]
            os.makedirs(backup_dir, exist_ok=True)
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(backup_dir, f"inference_results_{timestamp}.csv")
            
            try:
                import shutil
                shutil.copy2(csv_path, backup_path)
                self.logger.info(f"Backed up CSV to {backup_path}")
            except Exception as e:
                self.logger.error(f"Error backing up CSV: {e}")
        
        # Create new CSV file
        self.prepare_csv()
    
    def analyze_image(self, image_path, filename):
        """
        Analyze an image for defects.
        
        Args:
            image_path: Path to the captured image
            filename: Base filename
            
        Returns:
            Dictionary with analysis results
        """
        if not self.model:
            self.logger.error("Cannot analyze: Model not loaded")
            return None
        
        try:
            # Get current settings
            settings = self.settings_manager.get_settings()
            
            # Get thresholds
            global_threshold = settings.get("threshold", config.DEFAULT_SETTINGS["threshold"])
            patch_threshold = settings.get("patch_threshold", config.DEFAULT_SETTINGS["patch_threshold"])
            patch_defect_ratio = settings.get("patch_defect_ratio", config.DEFAULT_SETTINGS["patch_defect_ratio"])
            
            # Perform inference
            inference_result = self.model_manager.infer(image_path)
            
            if not inference_result:
                self.logger.error("Inference failed")
                return None
            
            global_error = inference_result["global_error"]
            detection_method = "Global"
            
            # Check for defect using global threshold
            is_defect = global_error > global_threshold
            
            # Apply patch-based analysis if enabled
            patch_value = 0
            patch_defect_ratio_value = 0
            
            if config.INFERENCE["USE_PATCH"]:
                # For a real implementation, this would perform patch-based analysis
                # For simplicity, we'll use simulated values
                patch_value = global_error * 1.1  # Just an example
                patch_defect_ratio_value = 0.3 if is_defect else 0.1  # Example values
                
                # Check if patch analysis indicates a defect
                if patch_value > patch_threshold and patch_defect_ratio_value > patch_defect_ratio:
                    is_defect = True
                    detection_method = "Patch"
            
            # Save image to appropriate directory
            if is_defect:
                dest_dir = config.PATHS["ANOMALY_DIR"]
                image_type = "bad"
                size = config.INFERENCE["BAD_IMAGE_RESIZE"]
            else:
                dest_dir = config.PATHS["NORMAL_DIR"]
                image_type = "good"
                size = config.INFERENCE["GOOD_IMAGE_RESIZE"]
            
            # Ensure directory exists
            os.makedirs(dest_dir, exist_ok=True)
            
            # Resize and save the image
            dest_path = os.path.join(dest_dir, filename)
            resize_image(image_path, dest_path, size, image_type)
            
            # Generate heatmap for defective images
            heatmap_path = None
            if is_defect and not self.args.no_heatmap:
                try:
                    base_name, ext = os.path.splitext(dest_path)
                    heatmap_path = f"{base_name}_heatmap.png"
                    generate_heatmap(image_path, self.model, output_path=heatmap_path)
                    self.logger.info(f"Generated heatmap: {heatmap_path}")
                except Exception as e:
                    self.logger.error(f"Error generating heatmap: {e}")
            
            # Create result dictionary
            result = {
                "timestamp": datetime.datetime.now().isoformat(),
                "filename": filename,
                "global_threshold": global_threshold,
                "global_error": global_error,
                "patch_threshold": patch_threshold,
                "mean_patch_error": patch_value,
                "patch_defect_ratio_threshold": patch_defect_ratio,
                "patch_defect_ratio_value": patch_defect_ratio_value,
                "is_defect": is_defect,
                "detection_method": detection_method,
                "image_path": dest_path,
                "heatmap_path": heatmap_path
            }
            
            # Log result
            status = "Defect" if is_defect else "Normal"
            self.logger.info(f"Analysis result: {status} (Error: {global_error:.4f}, Threshold: {global_threshold:.4f})")
            
            return result
        except Exception as e:
            self.logger.error(f"Error analyzing image: {e}")
            return None
    
    def record_result(self, result):
        """
        Record analysis result to CSV.
        
        Args:
            result: Analysis result dictionary
        """
        if not self.csv_writer:
            self.logger.error("Cannot record result: CSV writer not initialized")
            return
        
        try:
            # Write result to CSV
            self.csv_writer.writerow([
                result["timestamp"],
                result["filename"],
                result["global_threshold"],
                result["global_error"],
                result["patch_threshold"],
                result["mean_patch_error"],
                result["patch_defect_ratio_threshold"],
                result["patch_defect_ratio_value"],
                "Yes" if result["is_defect"] else "No",
                result["detection_method"]
            ])
            
            # Flush to ensure data is written
            self.csv_file.flush()
            
            self.logger.debug(f"Recorded result for {result['filename']} to CSV")
        except Exception as e:
            self.logger.error(f"Error recording result to CSV: {e}")
    
    def update_settings(self, result):
        """
        Update settings based on analysis result.
        
        Args:
            result: Analysis result dictionary
        """
        try:
            # Update settings based on result
            self.settings_manager.update_after_analysis(
                is_defect=result["is_defect"],
                reconstruction_error=result["global_error"],
                image_path=result["image_path"],
                patch_metrics={
                    "mean_patch_error": result["mean_patch_error"],
                    "defect_patch_ratio": result["patch_defect_ratio_value"]
                }
            )
            
            self.logger.debug("Updated settings based on analysis result")
        except Exception as e:
            self.logger.error(f"Error updating settings: {e}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Defect Detection Inference System")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--simulation", action="store_true", help="Run in simulation mode (no hardware)")
    parser.add_argument("--reset", action="store_true", help="Reset counters on startup")
    parser.add_argument("--no-heatmap", action="store_true", help="Disable heatmap generation")
    return parser.parse_args()

def main():
    """Main entry point."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Ensure configuration directories exist
    config.ensure_directories()
    
    # Initialize inference system
    system = InferenceSystem(args)
    
    # Start the system
    try:
        system.start()
    except KeyboardInterrupt:
        pass  # Handle in start() method
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
