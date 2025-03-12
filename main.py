#!/usr/bin/env python3
"""
Main Application for Defect Detection System
Implements the core detection and analysis workflow with hardware integration.
"""

import os
import sys
import time
import threading
import argparse
import signal
import datetime
from pathlib import Path

# Import core components
from config import config
from models.model_manager import ModelManager
from database.database_manager import DatabaseManager
from utils.settings_manager import SettingsManager
from utils.hardware_controller import HardwareController
from utils.image_utils import resize_image, annotate_image, generate_heatmap
from utils.system_monitor import SystemMonitor
from utils.logger import setup_logger, log_startup_info

class DefectDetectionSystem:
    """
    Main application class for defect detection system.
    Coordinates hardware, model inference, and result storage.
    """
    
    def __init__(self, args):
        """Initialize the defect detection system with command-line arguments."""
        self.args = args
        self.running = False
        self.detection_thread = None
        self.detection_lock = threading.RLock()
        
        # Set up logging
        log_level = "DEBUG" if args.debug else "INFO"
        self.logger = setup_logger("main", config.PATHS["INFERENCE_LOG"], level=log_level)
        log_startup_info(self.logger)
        
        # Initialize components
        self.logger.info("Initializing system components...")
        
        # Initialize settings manager
        self.settings_manager = SettingsManager()
        
        # Initialize database manager
        self.db_manager = DatabaseManager()
        
        # Initialize hardware controller with simulation mode
        self.hardware = HardwareController(simulation_mode=args.simulation)
        
        # Initialize model manager
        self.model_manager = ModelManager()
        
        # Initialize system monitor
        self.system_monitor = SystemMonitor()
        
        # Load model
        self.load_model()
        
        # Check for reset command
        if args.reset:
            self.reset_system()
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        self.logger.info("System initialization complete")
    
    def load_model(self):
        """Load the model and set initial thresholds."""
        try:
            self.logger.info("Loading model...")
            self.model = self.model_manager.load_model()
            self.logger.info("Model loaded successfully")
            
            # Check if there's a recommended threshold
            recommended_threshold = self.model_manager.get_recommended_threshold()
            if recommended_threshold:
                self.logger.info(f"Using recommended threshold: {recommended_threshold}")
                # Update settings only if different from current
                settings = self.settings_manager.get_settings()
                if abs(settings.get("threshold", 0) - recommended_threshold) > 0.01:
                    self.settings_manager.update_settings(threshold=recommended_threshold)
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            # Continue with default thresholds if model fails to load
            self.model = None
    
    def reset_system(self):
        """Reset the system counters and clear images if requested."""
        self.logger.info("Resetting system...")
        
        # Reset settings (counters only, preserve thresholds)
        self.settings_manager.reset_counters()
        
        # Check for reset signal flags
        reset_detected, clear_images, reset_csv = self.settings_manager.check_for_reset_signal()
        
        # Clear images if requested
        if clear_images or self.args.clear_images:
            self.logger.info("Clearing image directories...")
            
            # Clear Normal directory
            normal_dir = config.PATHS["NORMAL_DIR"]
            if os.path.exists(normal_dir):
                for file in os.listdir(normal_dir):
                    file_path = os.path.join(normal_dir, file)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
            
            # Clear Anomaly directory
            anomaly_dir = config.PATHS["ANOMALY_DIR"]
            if os.path.exists(anomaly_dir):
                for file in os.listdir(anomaly_dir):
                    file_path = os.path.join(anomaly_dir, file)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
            
            self.logger.info("Image directories cleared")
        
        # Reset CSV if requested
        if reset_csv or self.args.reset_csv:
            self.logger.info("Resetting CSV file...")
            csv_path = config.PATHS["CSV_FILENAME"]
            if os.path.exists(csv_path):
                # Backup the CSV first
                backup_dir = config.PATHS["BACKUP_CSV_DIR"]
                os.makedirs(backup_dir, exist_ok=True)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = os.path.join(backup_dir, f"inference_results_{timestamp}.csv")
                
                # Copy to backup
                import shutil
                shutil.copy2(csv_path, backup_path)
                
                # Create empty CSV with headers
                with open(csv_path, 'w') as f:
                    f.write("timestamp,filename,global_threshold,global_error,patch_threshold,")
                    f.write("mean_patch_error,patch_defect_ratio_threshold,patch_defect_ratio_value,is_defect,detection_method\n")
                
                self.logger.info(f"CSV reset (backup created at {backup_path})")
        
        self.logger.info("System reset complete")
    
    def signal_handler(self, sig, frame):
        """Handle termination signals for graceful shutdown."""
        self.logger.info(f"Received signal {sig}, shutting down...")
        self.stop()
    
    def start(self):
        """Start the detection system."""
        if self.running:
            self.logger.warning("System is already running")
            return
        
        self.running = True
        
        # Start system monitor
        self.system_monitor.start_monitoring()
        
        # Start detection thread
        self.detection_thread = threading.Thread(target=self.detection_loop)
        self.detection_thread.daemon = True
        self.detection_thread.start()
        
        self.logger.info("Defect detection system started")
        
        # If GUI mode is disabled, wait for detection thread
        if not self.args.gui:
            try:
                while self.running:
                    time.sleep(1)
            except KeyboardInterrupt:
                self.logger.info("Keyboard interrupt received, shutting down...")
                self.stop()
    
    def stop(self):
        """Stop the detection system."""
        self.logger.info("Stopping defect detection system...")
        self.running = False
        
        # Stop system monitor
        self.system_monitor.stop_monitoring()
        
        # Wait for detection thread to complete
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=5)
        
        # Clean up hardware resources
        self.hardware.cleanup()
        
        self.logger.info("System stopped")
    
    def detection_loop(self):
        """Main detection loop that runs in a separate thread."""
        self.logger.info("Detection loop started")
        
        while self.running:
            try:
                # Wait for trigger
                self.logger.info("Waiting for trigger...")
                if not self.hardware.wait_for_trigger(timeout=2):
                    continue
                
                # Lock to prevent concurrent detections
                with self.detection_lock:
                    self.logger.info("Trigger detected, capturing image...")
                    
                    # Capture image
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"image_{timestamp}.jpg"
                    temp_path = os.path.join("/tmp", filename)
                    
                    if not self.hardware.capture_image(temp_path):
                        self.logger.error("Failed to capture image")
                        continue
                    
                    # Perform analysis
                    self.logger.info("Analyzing image...")
                    result = self.analyze_image(temp_path, filename)
                    
                    if result:
                        # Store result in database
                        self.db_manager.store_detection_result(result)
                        
                        # Update settings based on result
                        self.settings_manager.update_after_analysis(
                            is_defect=result["is_defect"],
                            reconstruction_error=result["global_error"],
                            image_path=result["image_path"],
                            patch_metrics={
                                "mean_patch_error": result.get("patch_value", 0),
                                "defect_patch_ratio": result.get("patch_defect_ratio_value", 0)
                            }
                        )
                        
                        # Indicate defect if detected
                        if result["is_defect"]:
                            self.logger.warning(f"Defect detected! Error: {result['global_error']:.4f}")
                            self.hardware.indicate_defect()
                    
                    # Short delay before next capture
                    time.sleep(1)
            except Exception as e:
                self.logger.error(f"Error in detection loop: {e}")
                time.sleep(5)  # Delay before retry on error
    
    def analyze_image(self, image_path, filename):
        """
        Analyze an image for defects.
        
        Args:
            image_path: Path to the captured image
            filename: Base filename
            
        Returns:
            Result dictionary or None if analysis failed
        """
        if not self.model:
            self.logger.error("Cannot analyze image: Model not loaded")
            return None
        
        try:
            # Get current settings
            settings = self.settings_manager.get_settings()
            
            # Extract thresholds
            global_threshold = settings.get("threshold", config.DEFAULT_SETTINGS["threshold"])
            patch_threshold = settings.get("patch_threshold", config.DEFAULT_SETTINGS["patch_threshold"])
            patch_defect_ratio = settings.get("patch_defect_ratio", config.DEFAULT_SETTINGS["patch_defect_ratio"])
            
            # Perform global analysis
            inference_result = self.model_manager.infer(image_path)
            
            if not inference_result:
                self.logger.error("Inference failed")
                return None
            
            # Determine if defect is present
            is_defect = inference_result["global_error"] > global_threshold
            detection_method = "Global"
            patch_value = 0
            patch_defect_ratio_value = 0
            
            # If use patch analysis is enabled in config, do patch-based analysis
            if config.INFERENCE["USE_PATCH"] and not is_defect:
                # This would call a more detailed analysis using patches
                # For simplicity, we'll just use the global result for now
                patch_value = inference_result["global_error"] * 1.1  # Just an example
                patch_defect_ratio_value = 0.2  # Just an example
                
                # Check if patch metrics indicate a defect
                if patch_value > patch_threshold and patch_defect_ratio_value > patch_defect_ratio:
                    is_defect = True
                    detection_method = "Patch"
            
            # Copy image to appropriate directory based on result
            if is_defect:
                dest_dir = config.PATHS["ANOMALY_DIR"]
            else:
                dest_dir = config.PATHS["NORMAL_DIR"]
            
            # Ensure directory exists
            os.makedirs(dest_dir, exist_ok=True)
            
            # Resize the image
            if is_defect:
                size = config.INFERENCE["BAD_IMAGE_RESIZE"]
            else:
                size = config.INFERENCE["GOOD_IMAGE_RESIZE"]
            
            dest_path = os.path.join(dest_dir, filename)
            resize_image(image_path, dest_path, size, "bad" if is_defect else "good")
            
            # Create a heatmap if it's a defect
            heatmap_path = None
            if is_defect:
                try:
                    heatmap_path = generate_heatmap(image_path, self.model)
                except Exception as heatmap_error:
                    self.logger.error(f"Error generating heatmap: {heatmap_error}")
            
            # Prepare result for storage
            result = {
                "timestamp": datetime.datetime.now().isoformat(),
                "filename": filename,
                "global_threshold": global_threshold,
                "global_error": inference_result["global_error"],
                "patch_threshold": patch_threshold,
                "patch_value": patch_value,
                "patch_defect_ratio_threshold": patch_defect_ratio,
                "patch_defect_ratio_value": patch_defect_ratio_value,
                "is_defect": 1 if is_defect else 0,
                "detection_method": detection_method,
                "image_path": dest_path,
                "heatmap_path": heatmap_path
            }
            
            self.logger.info(f"Analysis complete: {'Defect' if is_defect else 'Normal'} (Error: {inference_result['global_error']:.4f})")
            return result
        except Exception as e:
            self.logger.error(f"Error analyzing image: {e}")
            return None
    
    def run_standalone_analysis(self, image_path):
        """
        Run analysis on a single image without trigger (for testing).
        
        Args:
            image_path: Path to the image to analyze
            
        Returns:
            Analysis result
        """
        if not os.path.exists(image_path):
            self.logger.error(f"Image not found: {image_path}")
            return None
        
        filename = os.path.basename(image_path)
        self.logger.info(f"Running standalone analysis on {filename}")
        
        result = self.analyze_image(image_path, filename)
        
        if result:
            # Print results
            print("\nAnalysis Results:")
            print(f"  Global Error:       {result['global_error']:.6f}")
            print(f"  Global Threshold:   {result['global_threshold']:.6f}")
            print(f"  Patch Value:        {result['patch_value']:.6f}")
            print(f"  Patch Threshold:    {result['patch_threshold']:.6f}")
            print(f"  Defect Ratio:       {result['patch_defect_ratio_value']:.6f}")
            print(f"  Defect Ratio Thresh:{result['patch_defect_ratio_threshold']:.6f}")
            print(f"  Detection Method:   {result['detection_method']}")
            print(f"  Defect Detected:    {'Yes' if result['is_defect'] else 'No'}")
            print(f"  Saved Image:        {result['image_path']}")
            
            # Store result in database
            self.db_manager.store_detection_result(result)
            
            return result
        else:
            print("Analysis failed")
            return None

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Defect Detection System")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--simulation", action="store_true", help="Run in simulation mode (no hardware)")
    parser.add_argument("--gui", action="store_true", help="Launch GUI interface")
    parser.add_argument("--analyze", help="Analyze a single image file (standalone mode)")
    parser.add_argument("--reset", action="store_true", help="Reset system counters")
    parser.add_argument("--clear-images", action="store_true", help="Clear image directories")
    parser.add_argument("--reset-csv", action="store_true", help="Reset CSV file")
    return parser.parse_args()

def main():
    """Main entry point for the application."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Ensure configuration directories exist
    config.ensure_directories()
    
    # Initialize system
    system = DefectDetectionSystem(args)
    
    # Handle standalone analysis mode
    if args.analyze:
        system.run_standalone_analysis(args.analyze)
        return 0
    
    # Check if GUI mode is requested
    if args.gui:
        try:
            # Import and run UI script
            sys.path.append(os.path.dirname(config.PATHS["UI_SCRIPT"]))
            from ui.main import main as ui_main
            return ui_main()
        except ImportError:
            system.logger.error("Failed to import UI module. Running in console mode instead.")
    
    # Start the system in normal mode
    try:
        system.start()
    except KeyboardInterrupt:
        system.logger.info("Keyboard interrupt received, shutting down...")
    finally:
        system.stop()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
