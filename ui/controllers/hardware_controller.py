#!/usr/bin/env python3
"""
Hardware Controller for Defect Detection System
Manages camera, GPIO, and LED indicators
"""

import os
import time
import threading
from PyQt5.QtCore import QObject, pyqtSignal
from utils.logger import get_logger

# Import hardware libraries with graceful fallback for simulation mode
try:
    # These imports will only succeed on a Raspberry Pi with the proper hardware setup
    import board
    import neopixel
    import RPi.GPIO as GPIO
    from picamera2 import Picamera2
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False
    # Create mock classes for simulation mode
    class MockGPIO:
        BCM = "BCM"
        IN = "IN"
        OUT = "OUT"
        PUD_UP = "PUD_UP"
        FALLING = "FALLING"
        
        @staticmethod
        def setmode(mode): pass
        
        @staticmethod
        def setup(pin, direction, pull_up_down=None): pass
        
        @staticmethod
        def wait_for_edge(pin, edge, timeout=None): 
            # Simulate a button press after a short delay
            if timeout is None or timeout > 0.5:
                time.sleep(0.5)
                return pin
            return None
        
        @staticmethod
        def cleanup(): pass
    
    # Mock camera class
    class MockPicamera2:
        def __init__(self):
            self.running = False
        
        def create_still_configuration(self):
            return {}
        
        def configure(self, config):
            pass
        
        def start(self):
            self.running = True
        
        def stop(self):
            self.running = False
        
        def capture_file(self, filename):
            # Create a simple test image
            try:
                from PIL import Image, ImageDraw
                img = Image.new('RGB', (640, 480), color=(73, 109, 137))
                d = ImageDraw.Draw(img)
                d.text((10, 10), f"Simulated capture at {time.time()}", fill=(255, 255, 0))
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                img.save(filename)
            except ImportError:
                # If PIL is not available, just create an empty file
                with open(filename, 'wb') as f:
                    f.write(b"SIMULATED IMAGE")
    
    # Mock neopixel class
    class MockNeoPixel:
        def __init__(self, pin, count, brightness=0.2, auto_write=False):
            self.pin = pin
            self.count = count
            self.brightness = brightness
            self.auto_write = auto_write
            self.pixels = [(0, 0, 0)] * count
        
        def fill(self, color):
            self.pixels = [color] * self.count
        
        def show(self):
            pass
    
    # Replace imports with mocks
    GPIO = MockGPIO
    Picamera2 = MockPicamera2
    neopixel = None

class HardwareController(QObject):
    """
    Manages hardware components with robust error handling and recovery.
    """
    
    # Define signals
    trigger_detected = pyqtSignal()
    image_captured = pyqtSignal(str)  # (image_path)
    hardware_error = pyqtSignal(str)  # (error_message)
    
    def __init__(self, simulation_mode=False):
        """
        Initialize hardware components with simulation mode support.
        
        Args:
            simulation_mode: If True, run without actual hardware (for development)
        """
        super().__init__()
        self.logger = get_logger("hardware_controller")
        self.simulation_mode = simulation_mode or not HARDWARE_AVAILABLE
        
        # Configuration from config
        from config import config
        self.trigger_pin = config.HARDWARE.get("TRIGGER_PIN", 16)
        self.led_pin_str = config.HARDWARE.get("LED_PIN", "board.D18")
        self.led_count = config.HARDWARE.get("NUM_LEDS", 12)
        self.led_brightness = config.HARDWARE.get("LED_BRIGHTNESS", 0.4)
        
        # Initialized state tracking
        self.gpio_initialized = False
        self.led_initialized = False
        self.camera_initialized = False
        self.camera_lock = threading.RLock()
        
        self.logger.info(f"Initializing hardware controller (simulation_mode={self.simulation_mode})")
        
        # Set up components
        try:
            # Initialize GPIO
            self._setup_gpio()
            
            # Initialize LEDs
            self._setup_leds()
            
            # Initialize camera
            self._setup_camera()
            
            self.logger.info("Hardware controller initialized successfully")
        except Exception as e:
            self.logger.error(f"Error during hardware initialization: {e}")
            self.hardware_error.emit(f"Hardware initialization error: {e}")
    
    def _setup_gpio(self):
        """Set up GPIO pins with error handling."""
        if self.simulation_mode:
            self.logger.info("Simulating GPIO setup")
            self.gpio_initialized = True
            return
            
        try:
            # Set up GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.trigger_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            self.logger.info(f"GPIO initialized with trigger pin {self.trigger_pin}")
            self.gpio_initialized = True
        except Exception as e:
            self.logger.error(f"Failed to initialize GPIO: {e}")
            self.hardware_error.emit(f"GPIO initialization failed: {e}")
    
    def _setup_leds(self):
        """Set up LED indicators with error handling."""
        if self.simulation_mode:
            self.logger.info("Simulating LED setup")
            self.pixels = MockNeoPixel(None, self.led_count, self.led_brightness, auto_write=False)
            self.led_initialized = True
            return
            
        try:
            # Set up NeoPixel LEDs
            # Handle the case where the pin is specified as a string like "board.D18"
            if neopixel:
                if isinstance(self.led_pin_str, str) and self.led_pin_str.startswith("board."):
                    led_pin = getattr(board, self.led_pin_str.split(".")[1])
                else:
                    led_pin = self.led_pin_str
                
                self.pixels = neopixel.NeoPixel(
                    led_pin, 
                    self.led_count, 
                    brightness=self.led_brightness, 
                    auto_write=False
                )
                self.logger.info(f"LEDs initialized with {self.led_count} pixels")
                self.led_initialized = True
            else:
                self.logger.warning("Neopixel library not available, LED control disabled")
                self.pixels = MockNeoPixel(None, self.led_count, self.led_brightness, auto_write=False)
                self.led_initialized = True
        except Exception as e:
            self.logger.error(f"Failed to initialize LEDs: {e}")
            self.hardware_error.emit(f"LED initialization failed: {e}")
            self.pixels = None
    
    def _setup_camera(self):
        """Set up camera with error handling and recovery."""
        if self.simulation_mode:
            self.logger.info("Simulating camera setup")
            self.picam = Picamera2()
            self.camera_initialized = True
            return
            
        try:
            with self.camera_lock:
                self.picam = Picamera2()
                camera_config = self.picam.create_still_configuration()
                self.picam.configure(camera_config)
                self.picam.start()
                time.sleep(2)  # Camera warm-up
                self.logger.info("Camera initialized successfully")
                self.camera_initialized = True
        except Exception as e:
            self.logger.error(f"Failed to initialize camera: {e}")
            self.hardware_error.emit(f"Camera initialization failed: {e}")
            self.picam = None
    
    def set_leds(self, color):
        """
        Set the color of all LEDs.
        
        Args:
            color: RGB tuple for LED color
            
        Returns:
            Boolean indicating success or failure
        """
        if not self.led_initialized or not hasattr(self, 'pixels') or self.pixels is None:
            return False
            
        try:
            self.pixels.fill(color)
            self.pixels.show()
            return True
        except Exception as e:
            self.logger.error(f"Error setting LEDs: {e}")
            return False
    
    def turn_off_leds(self):
        """Turn off all LEDs."""
        return self.set_leds((0, 0, 0))
    
    def indicate_capturing(self):
        """Turn on indicator LEDs during image capture."""
        return self.set_leds((255, 255, 212))  # Warm white
    
    def indicate_defect(self):
        """Flash LEDs to indicate a defect was detected."""
        if not self.led_initialized:
            return
            
        try:
            # Flash red 3 times
            for _ in range(3):
                self.set_leds((255, 0, 0))  # Red
                time.sleep(0.2)
                self.turn_off_leds()
                time.sleep(0.2)
        except Exception as e:
            self.logger.error(f"Error indicating defect: {e}")
    
    def wait_for_trigger(self, timeout=None):
        """
        Wait for a trigger event from the hardware button.
        
        Args:
            timeout: Optional timeout in seconds (None waits indefinitely)
            
        Returns:
            Boolean indicating if trigger was detected
        """
        if self.simulation_mode or not self.gpio_initialized:
            self.logger.info("Simulating trigger wait")
            time.sleep(0.5)  # Simulate a delay
            return True
            
        self.logger.info(f"Waiting for trigger (timeout={timeout})")
        
        try:
            if timeout is None:
                # Wait indefinitely
                GPIO.wait_for_edge(self.trigger_pin, GPIO.FALLING)
                self.logger.info("Trigger detected")
                self.trigger_detected.emit()
                return True
            else:
                # Wait with timeout
                channel = GPIO.wait_for_edge(
                    self.trigger_pin,
                    GPIO.FALLING,
                    timeout=int(timeout * 1000)  # Convert to milliseconds
                )
                
                if channel is None:
                    self.logger.info("Trigger wait timed out")
                    return False
                else:
                    self.logger.info("Trigger detected")
                    self.trigger_detected.emit()
                    return True
        except Exception as e:
            self.logger.error(f"Error waiting for trigger: {e}")
            return False
    
    def capture_image(self, output_path):
        """
        Capture an image from the camera with LED indication.
        
        Args:
            output_path: Path to save the captured image
            
        Returns:
            Path to the captured image or None if failed
        """
        if not self.camera_initialized:
            self.logger.error("Cannot capture image: Camera not initialized")
            return None
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Turn on indicator LEDs
            self.indicate_capturing()
            
            # Capture image
            with self.camera_lock:
                self.picam.capture_file(output_path)
            
            # Turn off indicator LEDs
            self.turn_off_leds()
            
            # Emit signal with captured image path
            self.image_captured.emit(output_path)
            
            self.logger.info(f"Image captured: {output_path}")
            return output_path
        except Exception as e:
            self.logger.error(f"Error capturing image: {e}")
            self.hardware_error.emit(f"Image capture failed: {e}")
            self.turn_off_leds()
            
            # Try to recover the camera if there was an error
            self.reset_camera()
            
            return None
    
    def reset_camera(self):
        """Reset and reinitialize the camera if it's having issues."""
        if self.simulation_mode:
            return True
            
        try:
            self.logger.info("Resetting camera...")
            
            # Properly close camera if it exists
            try:
                with self.camera_lock:
                    if hasattr(self, 'picam') and self.picam is not None:
                        self.picam.stop()
                        self.picam = None
                        self.camera_initialized = False
            except Exception as e:
                self.logger.error(f"Error closing camera during reset: {e}")
            
            # Wait a bit before reinitializing
            time.sleep(1)
            
            # Reinitialize
            self._setup_camera()
            return self.camera_initialized
        except Exception as e:
            self.logger.error(f"Failed to reset camera: {e}")
            return False
    
    def cleanup(self):
        """Clean up hardware resources."""
        self.logger.info("Cleaning up hardware resources...")
        
        if self.simulation_mode:
            return
            
        # Turn off LEDs
        if self.led_initialized:
            self.turn_off_leds()
        
        # Clean up camera
        try:
            if hasattr(self, 'picam') and self.picam is not None:
                with self.camera_lock:
                    self.picam.stop()
                    self.picam = None
                    self.camera_initialized = False
                self.logger.info("Camera resources released")
        except Exception as e:
            self.logger.error(f"Error releasing camera resources: {e}")
        
        # Clean up GPIO
        if self.gpio_initialized:
            try:
                GPIO.cleanup()
                self.gpio_initialized = False
                self.logger.info("GPIO cleaned up")
            except Exception as e:
                self.logger.error(f"Error cleaning up GPIO: {e}")
