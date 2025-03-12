#!/usr/bin/env python3
"""
Hardware Controller for Defect Detection System
Manages hardware components with proper resource management and error recovery.
"""

import time
import os
import signal
import subprocess
import atexit
import threading
from contextlib import contextmanager

# Import hardware libraries with error handling
try:
    import board
    import neopixel
    import RPi.GPIO as GPIO
    from picamera2 import Picamera2
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False
    print("WARNING: Hardware libraries not available. Running in simulation mode.")

# Import centralized configuration
from config import config

# Set up logger
logger = config.setup_logger("hardware_controller")

class HardwareControllerError(Exception):
    """Custom exception for hardware controller errors"""
    pass

class HardwareController:
    """
    Manages hardware components with robust error handling and resource management.
    Supports graceful shutdown and recovery from hardware failures.
    """
    
    def __init__(self, simulation_mode=False):
        """
        Initialize hardware components with simulation mode support.
        
        Args:
            simulation_mode: If True, run without actual hardware (for development)
        """
        self.simulation_mode = simulation_mode or not HARDWARE_AVAILABLE
        
        # Initialized state tracking
        self.gpio_initialized = False
        self.led_initialized = False
        self.camera_initialized = False
        
        # Resource locks
        self.camera_lock = threading.RLock()
        
        logger.info(f"Initializing hardware controller (simulation_mode={self.simulation_mode})")
        
        # Set up components
        try:
            # Initialize GPIO
            self._setup_gpio()
            
            # Initialize LEDs
            self._setup_leds()
            
            # Initialize camera
            self._setup_camera()
            
            # Register cleanup on exit
            atexit.register(self.cleanup)
            
            logger.info("Hardware controller initialized successfully")
        except Exception as e:
            logger.error(f"Error during hardware initialization: {e}")
            
            # Attempt partial initialization for components that succeeded
            logger.info("Continuing with partial hardware initialization")
    
    def _setup_gpio(self):
        """Set up GPIO pins with error handling"""
        if self.simulation_mode:
            logger.info("Simulating GPIO setup")
            self.gpio_initialized = True
            return
            
        try:
            # Set up GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(config.HARDWARE["TRIGGER_PIN"], GPIO.IN, pull_up_down=GPIO.PUD_UP)
            logger.info(f"GPIO initialized with trigger pin {config.HARDWARE['TRIGGER_PIN']}")
            self.gpio_initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize GPIO: {e}")
            raise HardwareControllerError(f"GPIO initialization failed: {e}")
    
    def _setup_leds(self):
        """Set up LED indicators with error handling"""
        if self.simulation_mode:
            logger.info("Simulating LED setup")
            self.led_initialized = True
            return
            
        try:
            # Set up NeoPixel LEDs
            led_pin = eval(config.HARDWARE["LED_PIN"])  # Convert string to pin object
            self.pixels = neopixel.NeoPixel(
                led_pin, 
                config.HARDWARE["NUM_LEDS"], 
                brightness=config.HARDWARE["LED_BRIGHTNESS"], 
                auto_write=False
            )
            logger.info(f"LEDs initialized with {config.HARDWARE['NUM_LEDS']} pixels")
            self.led_initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize LEDs: {e}")
            self.pixels = None
    
    def _setup_camera(self):
        """Set up camera with error handling and recovery"""
        if self.simulation_mode:
            logger.info("Simulating camera setup")
            self.camera_initialized = True
            return
            
        # Force release camera resources first
        self.force_release_camera()
        
        try:
            with self.camera_lock:
                self.picam = Picamera2()
                camera_config = self.picam.create_still_configuration()
                self.picam.configure(camera_config)
                self.picam.start()
                time.sleep(2)  # Camera warm-up
                logger.info("Camera initialized successfully")
                self.camera_initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize camera on first attempt: {e}")
            
            # Try one more time after another forceful release
            try:
                self.force_release_camera()
                time.sleep(1)
                
                with self.camera_lock:
                    self.picam = Picamera2()
                    camera_config = self.picam.create_still_configuration()
                    self.picam.configure(camera_config)
                    self.picam.start()
                    time.sleep(2)  # Camera warm-up
                    logger.info("Camera initialized on second attempt")
                    self.camera_initialized = True
            except Exception as retry_error:
                logger.error(f"Failed to initialize camera on retry: {retry_error}")
                self.picam = None
                raise HardwareControllerError(f"Camera initialization failed: {retry_error}")
    
    def force_release_camera(self):
        """Force release camera resources using system commands"""
        if self.simulation_mode:
            return
            
        try:
            logger.info("Forcibly releasing camera resources...")
            
            # Try to kill any processes using the camera
            try:
                # Look for processes using v4l2 (Video4Linux2)
                subprocess.run(
                    ["sudo", "fuser", "-k", "/dev/video0"], 
                    stderr=subprocess.DEVNULL,
                    timeout=2
                )
            except subprocess.TimeoutExpired:
                logger.warning("Timeout while trying to release camera with fuser")
            
            # More aggressive approach - restart the camera module
            try:
                subprocess.run(
                    ["sudo", "rmmod", "bcm2835-v4l2"], 
                    stderr=subprocess.DEVNULL,
                    timeout=2
                )
                time.sleep(0.5)
                subprocess.run(
                    ["sudo", "modprobe", "bcm2835-v4l2"], 
                    stderr=subprocess.DEVNULL,
                    timeout=2
                )
            except subprocess.TimeoutExpired:
                logger.warning("Timeout while trying to reload camera module")
            
            logger.info("Camera resources should now be released")
        except Exception as e:
            logger.error(f"Error while trying to force release camera: {e}")
    
    @contextmanager
    def led_state(self, color):
        """
        Context manager for temporarily setting LED state.
        Automatically restores LEDs to off state when done.
        
        Args:
            color: RGB tuple for LED color
        """
        try:
            self.set_leds(color)
            yield
        finally:
            self.turn_off_leds()
    
    def set_leds(self, color):
        """
        Set the color of all LEDs.
        
        Args:
            color: RGB tuple for LED color
            
        Returns:
            Boolean indicating success or failure
        """
        if self.simulation_mode or not self.led_initialized:
            logger.debug(f"Simulating LED set to {color}")
            return True
            
        try:
            self.pixels.fill(color)
            self.pixels.show()
            return True
        except Exception as e:
            logger.error(f"Error setting LEDs: {e}")
            return False
    
    def turn_off_leds(self):
        """Turn off all LEDs"""
        return self.set_leds((0, 0, 0))
    
    def indicate_capturing(self):
        """Turn on indicator LEDs during image capture"""
        return self.set_leds((255, 255, 212))  # Warm white
    
    def indicate_defect(self):
        """Flash LEDs to indicate a defect was detected"""
        if self.simulation_mode or not self.led_initialized:
            logger.debug("Simulating defect indication")
            return
            
        try:
            # Flash red 3 times
            for _ in range(3):
                self.set_leds((255, 0, 0))  # Red
                time.sleep(0.2)
                self.turn_off_leds()
                time.sleep(0.2)
        except Exception as e:
            logger.error(f"Error indicating defect: {e}")
    
    def wait_for_trigger(self, timeout=None):
        """
        Wait for a trigger event from the hardware button.
        
        Args:
            timeout: Optional timeout in seconds (None waits indefinitely)
            
        Returns:
            Boolean indicating if trigger was detected
        """
        if self.simulation_mode or not self.gpio_initialized:
            logger.info("Simulating trigger wait")
            time.sleep(2)  # Simulate a delay
            return True
            
        logger.info(f"Waiting for trigger (timeout={timeout})")
        
        try:
            if timeout is None:
                # Wait indefinitely
                GPIO.wait_for_edge(config.HARDWARE["TRIGGER_PIN"], GPIO.FALLING)
                logger.info("Trigger detected")
                return True
            else:
                # Wait with timeout
                channel = GPIO.wait_for_edge(
                    config.HARDWARE["TRIGGER_PIN"],
                    GPIO.FALLING,
                    timeout=int(timeout * 1000)  # Convert to milliseconds
                )
                
                if channel is None:
                    logger.info("Trigger wait timed out")
                    return False
                else:
                    logger.info("Trigger detected")
                    return True
        except Exception as e:
            logger.error(f"Error waiting for trigger: {e}")
            return False
    
    def capture_image(self, image_path):
        """
        Capture an image from the camera with LED indication.
        
        Args:
            image_path: Path to save the captured image
            
        Returns:
            Boolean indicating success or failure
        """
        if self.simulation_mode:
            logger.info(f"Simulating image capture to {image_path}")
            
            # In simulation mode, create a dummy image
            try:
                from PIL import Image, ImageDraw
                
                # Create a blank image with text
                img = Image.new('RGB', (640, 480), color=(73, 109, 137))
                d = ImageDraw.Draw(img)
                d.text((10, 10), f"Simulated capture at {time.time()}", fill=(255, 255, 0))
                
                # Save the image
                os.makedirs(os.path.dirname(image_path), exist_ok=True)
                img.save(image_path)
                logger.info(f"Created simulated image: {image_path}")
                return True
            except Exception as e:
                logger.error(f"Error creating simulated image: {e}")
                return False
        
        # Real capture with camera
        try:
            # Check if camera is initialized
            if not self.camera_initialized or not hasattr(self, 'picam') or self.picam is None:
                logger.error("Cannot capture image: Camera not initialized")
                return False
                
            with self.led_state((255, 255, 212)):  # Use context manager for LED state
                logger.info(f"Capturing image to {image_path}")
                time.sleep(0.5)  # Short delay for stability
                
                with self.camera_lock:
                    self.picam.capture_file(image_path)
                
                logger.info(f"Image captured: {image_path}")
                return True
        except Exception as e:
            logger.error(f"Error capturing image: {e}")
            
            # Try to recover the camera
            try:
                self._setup_camera()
            except Exception as recovery_error:
                logger.error(f"Failed to recover camera after capture error: {recovery_error}")
                
            return False
    
    def reset_camera(self):
        """Reset and reinitialize the camera if it's having issues"""
        if self.simulation_mode:
            logger.info("Simulating camera reset")
            return True
            
        try:
            logger.info("Resetting camera...")
            
            # Properly close camera if it exists
            try:
                with self.camera_lock:
                    if hasattr(self, 'picam') and self.picam is not None:
                        self.picam.stop()
                        self.picam.close()
                        self.picam = None
                        self.camera_initialized = False
            except Exception as e:
                logger.error(f"Error closing camera during reset: {e}")
            
            # Force release any remaining resources
            self.force_release_camera()
            
            # Reinitialize
            self._setup_camera()
            return self.camera_initialized
        except Exception as e:
            logger.error(f"Failed to reset camera: {e}")
            return False
    
    def cleanup(self):
        """Clean up hardware resources on shutdown"""
        logger.info("Cleaning up hardware resources...")
        
        if self.simulation_mode:
            logger.info("Simulation mode: No hardware to clean up")
            return
            
        # Clean up camera
        try:
            if hasattr(self, 'picam') and self.picam is not None:
                with self.camera_lock:
                    logger.info("Stopping camera...")
                    self.picam.stop()
                    self.picam.close()
                    logger.info("Camera released")
        except Exception as e:
            logger.error(f"Error during camera cleanup: {e}")
        
        # Turn off LEDs
        try:
            if self.led_initialized:
                logger.info("Turning off LEDs...")
                self.turn_off_leds()
        except Exception as e:
            logger.error(f"Error turning off LEDs: {e}")
        
        # Clean up GPIO
        try:
            if self.gpio_initialized:
                logger.info("Cleaning up GPIO...")
                GPIO.cleanup()
        except Exception as e:
            logger.error(f"Error during GPIO cleanup: {e}")
    
    def reinitialize(self):
        """
        Attempt to reinitialize all hardware components after a failure.
        """
        logger.info("Reinitializing hardware components...")
        
        # Clean up first
        self.cleanup()
        
        # Reinitialize components
        success = True
        
        try:
            self._setup_gpio()
        except Exception as e:
            logger.error(f"Failed to reinitialize GPIO: {e}")
            success = False
            
        try:
            self._setup_leds()
        except Exception as e:
            logger.error(f"Failed to reinitialize LEDs: {e}")
            success = False
            
        try:
            self._setup_camera()
        except Exception as e:
            logger.error(f"Failed to reinitialize camera: {e}")
            success = False
        
        return success

# For testing
if __name__ == "__main__":
    print("Testing Hardware Controller")
    
    # Test in simulation mode first
    print("\nTesting in simulation mode:")
    hw = HardwareController(simulation_mode=True)
    
    # Test LED functions
    print("Testing LED functions...")
    hw.indicate_capturing()
    time.sleep(1)
    hw.turn_off_leds()
    hw.indicate_defect()
    
    # Test trigger wait with timeout
    print("Testing trigger wait (2 second timeout)...")
    hw.wait_for_trigger(timeout=2)
    
    # Test image capture
    print("Testing image capture...")
    test_image_path = "/tmp/test_capture.jpg"
    result = hw.capture_image(test_image_path)
    print(f"Capture result: {result}, Image path: {test_image_path}")
    
    # Clean up
    hw.cleanup()
    print("\nHardware Controller tests completed successfully.")
