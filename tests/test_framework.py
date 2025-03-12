#!/usr/bin/env python3
"""
Automated Testing Framework for Defect Detection System
Provides tools for unit testing, integration testing, and system testing.
"""

import os
import sys
import unittest
import tempfile
import shutil
import json
import time
import threading
import subprocess
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import modules to test
try:
    from config import config
except ImportError:
    print("Config module not found. Make sure the project root is in your Python path.")
    sys.exit(1)

# Set test configuration
# This creates a parallel configuration for testing to avoid modifying the real system
class TestConfig:
    """Test configuration"""
    
    # Create test directories in temp folder
    TEST_ROOT = tempfile.mkdtemp(prefix="defect_test_")
    
    # Base paths
    BASE_DIR = Path(TEST_ROOT)
    
    # Test directories
    NORMAL_DIR = BASE_DIR / "Normal"
    ANOMALY_DIR = BASE_DIR / "Anomaly"
    CONFIG_FOLDER = BASE_DIR / "Config_Files"
    
    # Test files
    SETTINGS_FILE = CONFIG_FOLDER / "settings.json"
    CSV_FILENAME = BASE_DIR / "inference_results.csv"
    TEST_DB_FILE = BASE_DIR / "test_database.db"
    TEST_LOG_FILE = BASE_DIR / "test.log"
    
    # Default settings
    DEFAULT_SETTINGS = {
        "threshold": 0.85,
        "patch_threshold": 0.7,
        "patch_defect_ratio": 0.45,
        "good_count": 0,
        "bad_count": 0,
        "last_bad_photo": "",
        "last_good_photo": "",
        "alarm": 0,
        "last_reconstruction_error": 0,
        "last_10_bad_photos": [],
        "image_counter": 0,
        "latest_patch_value": 0.0,
        "latest_defect_ratio": 0.0
    }
    
    @classmethod
    def setup(cls):
        """Set up test environment"""
        # Create test directories
        cls.NORMAL_DIR.mkdir(exist_ok=True, parents=True)
        cls.ANOMALY_DIR.mkdir(exist_ok=True, parents=True)
        cls.CONFIG_FOLDER.mkdir(exist_ok=True, parents=True)
        
        # Create settings file
        with open(cls.SETTINGS_FILE, "w") as f:
            json.dump(cls.DEFAULT_SETTINGS, f, indent=4)
    
    @classmethod
    def teardown(cls):
        """Clean up test environment"""
        try:
            shutil.rmtree(cls.TEST_ROOT)
        except Exception as e:
            print(f"Error cleaning up test directory: {e}")

# Mock classes for hardware testing
class MockCamera:
    """Mock camera for testing"""
    
    def __init__(self):
        self.capture_file_mock = MagicMock()
    
    def capture_file(self, output_path):
        """Mock capturing an image file"""
        self.capture_file_mock(output_path)
        # Create a test image file
        with open(output_path, "wb") as f:
            f.write(b"TEST_IMAGE")

class MockGPIO:
    """Mock GPIO for testing"""
    
    # GPIO modes
    BCM = "BCM"
    IN = "IN"
    OUT = "OUT"
    PUD_UP = "PUD_UP"
    FALLING = "FALLING"
    
    @staticmethod
    def setmode(mode):
        """Mock setting GPIO mode"""
        pass
    
    @staticmethod
    def setup(pin, direction, pull_up_down=None):
        """Mock setting up a GPIO pin"""
        pass
    
    @staticmethod
    def wait_for_edge(pin, edge):
        """Mock waiting for a GPIO edge"""
        # Wait a short time to simulate a trigger
        time.sleep(0.1)
    
    @staticmethod
    def cleanup():
        """Mock cleaning up GPIO"""
        pass

class MockNeoPixel:
    """Mock NeoPixel for testing"""
    
    def __init__(self, pin, count, brightness=0.2, auto_write=False):
        """Initialize mock NeoPixel"""
        self.pin = pin
        self.count = count
        self.brightness = brightness
        self.auto_write = auto_write
        self.pixels = [(0, 0, 0)] * count
    
    def fill(self, color):
        """Mock filling all pixels with a color"""
        self.pixels = [color] * self.count
    
    def show(self):
        """Mock showing pixels"""
        pass

# Base test case with setup and teardown
class BaseTestCase(unittest.TestCase):
    """Base test case for defect detection system tests"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for the class"""
        TestConfig.setup()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment after class tests"""
        TestConfig.teardown()
    
    def setUp(self):
        """Set up test environment before each test"""
        # Reset settings file to defaults
        with open(TestConfig.SETTINGS_FILE, "w") as f:
            json.dump(TestConfig.DEFAULT_SETTINGS, f, indent=4)
    
    def tearDown(self):
        """Clean up after each test"""
        # Clean up any test files
        for path in TestConfig.NORMAL_DIR.glob("*"):
            if path.is_file():
                path.unlink()
        
        for path in TestConfig.ANOMALY_DIR.glob("*"):
            if path.is_file():
                path.unlink()

# Test cases for settings manager
class SettingsManagerTests(BaseTestCase):
    """Test cases for settings manager"""
    
    def test_load_settings(self):
        """Test loading settings from file"""
        # Import the module here to use test config
        from utils.settings_manager import SettingsManager
        
        # Create settings manager with test config
        settings_manager = SettingsManager(TestConfig.SETTINGS_FILE)
        
        # Test loading settings
        settings = settings_manager.get_settings()
        self.assertEqual(settings["threshold"], TestConfig.DEFAULT_SETTINGS["threshold"])
        self.assertEqual(settings["patch_threshold"], TestConfig.DEFAULT_SETTINGS["patch_threshold"])
    
    def test_save_settings(self):
        """Test saving settings to file"""
        from utils.settings_manager import SettingsManager
        
        # Create settings manager with test config
        settings_manager = SettingsManager(TestConfig.SETTINGS_FILE)
        
        # Modify settings
        new_settings = dict(TestConfig.DEFAULT_SETTINGS)
        new_settings["threshold"] = 0.95
        
        # Save settings
        result = settings_manager.save_settings(new_settings)
        self.assertTrue(result)
        
        # Reload settings and verify changes
        settings = settings_manager.get_settings(bypass_cache=True)
        self.assertEqual(settings["threshold"], 0.95)
    
    def test_reset_counters(self):
        """Test resetting counters"""
        from utils.settings_manager import SettingsManager
        
        # Create settings manager with test config
        settings_manager = SettingsManager(TestConfig.SETTINGS_FILE)
        
        # Modify settings
        new_settings = dict(TestConfig.DEFAULT_SETTINGS)
        new_settings["good_count"] = 10
        new_settings["bad_count"] = 5
        new_settings["threshold"] = 0.9
        
        # Save settings
        settings_manager.save_settings(new_settings)
        
        # Reset counters
        result = settings_manager.reset_counters()
        self.assertTrue(result)
        
        # Verify counters are reset but threshold is preserved
        settings = settings_manager.get_settings(bypass_cache=True)
        self.assertEqual(settings["good_count"], 0)
        self.assertEqual(settings["bad_count"], 0)
        self.assertEqual(settings["threshold"], 0.9)

# Test cases for image processor
class ImageProcessorTests(BaseTestCase):
    """Test cases for image processor"""
    
    def create_test_image(self, path, size=(640, 480)):
        """Create a test image for testing"""
        from PIL import Image, ImageDraw
        
        # Create a blank image
        img = Image.new('RGB', size, color=(73, 109, 137))
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), f"Test image at {time.time()}", fill=(255, 255, 0))
        
        # Save the image
        img.save(path)
        return path
    
    def test_resize_image(self):
        """Test resizing images"""
        from utils.image_utils import resize_image
        
        # Create a test image
        test_image = TestConfig.NORMAL_DIR / "test_resize.jpg"
        self.create_test_image(test_image, (1024, 768))
        
        # Resize the image
        output_path = TestConfig.NORMAL_DIR / "resized.jpg"
        result = resize_image(test_image, output_path, (320, 240), "good")
        
        # Verify the result
        self.assertEqual(result, output_path)
        self.assertTrue(output_path.exists())
        
        # Check the size
        from PIL import Image
        with Image.open(output_path) as img:
            self.assertEqual(img.size, (320, 240))
    
    def test_annotate_image(self):
        """Test annotating images"""
        from utils.image_utils import annotate_image
        
        # Create a test image
        test_image = TestConfig.NORMAL_DIR / "test_annotate.jpg"
        self.create_test_image(test_image)
        
        # Annotate the image
        metrics = {
            "Global Error": 0.85,
            "Threshold": 0.9,
            "Status": "Normal"
        }
        output_path = TestConfig.NORMAL_DIR / "annotated.jpg"
        result = annotate_image(test_image, output_path, "Test Annotation", metrics)
        
        # Verify the result
        self.assertEqual(result, output_path)
        self.assertTrue(output_path.exists())

# Test cases for hardware controller with mocks
class HardwareControllerTests(BaseTestCase):
    """Test cases for hardware controller with mock hardware"""
    
    @patch("utils.hardware_controller.Picamera2", MockCamera)
    @patch("utils.hardware_controller.GPIO", MockGPIO)
    @patch("utils.hardware_controller.neopixel.NeoPixel", MockNeoPixel)
    def test_capture_image(self):
        """Test capturing an image"""
        from utils.hardware_controller import HardwareController
        
        # Create hardware controller with test config
        hardware = HardwareController()
        
        # Capture an image
        image_path = TestConfig.NORMAL_DIR / "test_capture.jpg"
        result = hardware.capture_image(image_path)
        
        # Verify the result
        self.assertTrue(result)
        self.assertTrue(image_path.exists())
    
    @patch("utils.hardware_controller.Picamera2", MockCamera)
    @patch("utils.hardware_controller.GPIO", MockGPIO)
    @patch("utils.hardware_controller.neopixel.NeoPixel", MockNeoPixel)
    def test_wait_for_trigger(self):
        """Test waiting for a trigger"""
        from utils.hardware_controller import HardwareController
        
        # Create hardware controller with test config
        hardware = HardwareController()
        
        # Wait for trigger with timeout
        start_time = time.time()
        result = hardware.wait_for_trigger(timeout=1)
        elapsed_time = time.time() - start_time
        
        # Verify the result
        self.assertTrue(result)
        self.assertLess(elapsed_time, 0.5)  # Should be quick with mock

# Test cases for database manager
class DatabaseManagerTests(BaseTestCase):
    """Test cases for database manager"""
    
    def test_store_and_retrieve(self):
        """Test storing and retrieving detection results"""
        from database.database_manager import DatabaseManager
        
        # Create database manager with test config
        db_manager = DatabaseManager(TestConfig.TEST_DB_FILE)
        
        # Store a test result
        result_data = {
            "filename": "test_image.jpg",
            "global_threshold": 0.85,
            "global_error": 0.75,
            "patch_threshold": 0.7,
            "patch_value": 0.65,
            "patch_defect_ratio_threshold": 0.45,
            "patch_defect_ratio_value": 0.4,
            "is_defect": 0,
            "detection_method": "Global",
            "image_path": str(TestConfig.NORMAL_DIR / "test_image.jpg")
        }
        
        record_id = db_manager.store_detection_result(result_data)
        self.assertIsNotNone(record_id)
        
        # Retrieve the result
        retrieved = db_manager.get_detection_by_id(record_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["filename"], "test_image.jpg")
        self.assertEqual(retrieved["global_threshold"], 0.85)
        self.assertEqual(retrieved["is_defect"], 0)
    
    def test_query_by_date_range(self):
        """Test querying detection results by date range"""
        from database.database_manager import DatabaseManager
        import datetime
        
        # Create database manager with test config
        db_manager = DatabaseManager(TestConfig.TEST_DB_FILE)
        
        # Store test results with different dates
        yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)).isoformat()
        today = datetime.datetime.now().isoformat()
        
        # Yesterday's result
        result1 = {
            "timestamp": yesterday,
            "filename": "yesterday.jpg",
            "global_threshold": 0.85,
            "global_error": 0.75,
            "is_defect": 0
        }
        
        # Today's result
        result2 = {
            "timestamp": today,
            "filename": "today.jpg",
            "global_threshold": 0.85,
            "global_error": 0.95,
            "is_defect": 1
        }
        
        db_manager.store_detection_result(result1)
        db_manager.store_detection_result(result2)
        
        # Query by today's date
        today_date = datetime.date.today().isoformat()
        results = db_manager.get_detections_by_date_range(today_date, today_date)
        
        # Verify results
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["filename"], "today.jpg")

# Test cases for security utils
class SecurityUtilsTests(BaseTestCase):
    """Test cases for security utilities"""
    
    def test_password_hashing(self):
        """Test password hashing and verification"""
        from utils.security import hash_password, verify_password
        
        # Create a password hash
        password = "Test@123"
        password_hash, salt = hash_password(password)
        
        # Verify the hash
        self.assertIsNotNone(password_hash)
        self.assertIsNotNone(salt)
        
        # Test verification with the correct password
        # Need to mock the config.SECURITY settings
        with patch("utils.security.config") as mock_config:
            mock_config.SECURITY = {
                "UI_PASSWORD_HASH": password_hash,
                "PASSWORD_SALT": salt
            }
            
            result = verify_password(password)
            self.assertTrue(result)
            
            # Test with incorrect password
            result = verify_password("WrongPassword")
            self.assertFalse(result)
    
    def test_password_strength(self):
        """Test password strength checker"""
        from utils.security import is_password_strong
        
        # Test strong passwords
        self.assertTrue(is_password_strong("StrongPass123!"))
        self.assertTrue(is_password_strong("C0mpl3x@Pass"))
        
        # Test weak passwords
        self.assertFalse(is_password_strong("password"))
        self.assertFalse(is_password_strong("12345678"))
        self.assertFalse(is_password_strong("NoDigits!"))
        self.assertFalse(is_password_strong("noUppercase123!"))
        self.assertFalse(is_password_strong("NOLOWERCASE123!"))
        self.assertFalse(is_password_strong("NoSpecial123"))

# Integration tests
class IntegrationTests(BaseTestCase):
    """Integration tests for multiple components"""
    
    @patch("utils.hardware_controller.Picamera2", MockCamera)
    @patch("utils.hardware_controller.GPIO", MockGPIO)
    @patch("utils.hardware_controller.neopixel.NeoPixel", MockNeoPixel)
    def test_capture_and_analyze(self):
        """Test capturing and analyzing an image"""
        from utils.hardware_controller import HardwareController
        from utils.settings_manager import SettingsManager
        
        # Create components with test config
        hardware = HardwareController()
        settings_manager = SettingsManager(TestConfig.SETTINGS_FILE)
        
        # Capture an image
        image_path = TestConfig.NORMAL_DIR / "test_integration.jpg"
        hardware.capture_image(image_path)
        
        # Verify the image exists
        self.assertTrue(image_path.exists())
        
        # Update settings to reflect the capture
        settings_manager.update_settings(
            last_good_photo=str(image_path),
            good_count=1,
            image_counter=1
        )
        
        # Check settings were updated
        settings = settings_manager.get_settings(bypass_cache=True)
        self.assertEqual(settings["good_count"], 1)
        self.assertEqual(settings["image_counter"], 1)
        self.assertEqual(settings["last_good_photo"], str(image_path))

# System test with subprocess
class SystemTests(BaseTestCase):
    """System tests running actual scripts"""
    
    def test_config_script(self):
        """Test running the configuration script"""
        # Skip if config.py is not executable
        if not os.access("config.py", os.X_OK):
            self.skipTest("config.py is not executable")
        
        # Run the config script
        result = subprocess.run(
            ["python", "config.py"],
            capture_output=True,
            text=True
        )
        
        # Check exit code
        self.assertEqual(result.returncode, 0)
        
        # Check output contains expected text
        self.assertIn("Configuration module loaded", result.stdout)
    
    @unittest.skip("Requires UI environment")
    def test_ui_startup(self):
        """Test UI startup (skipped by default, requires display)"""
        # This test is skipped by default as it requires a display
        # Run the UI script with a short timeout
        process = subprocess.Popen(
            ["python", "ui/main.py", "--debug"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        try:
            # Wait a short time for startup
            time.sleep(2)
            
            # Check if process is still running
            if process.poll() is None:
                # Process is still running, so startup succeeded
                self.assertIsNone(process.poll())
            else:
                # Process exited, check for errors
                stdout, stderr = process.communicate()
                self.fail(f"UI process exited with code {process.returncode}. "
                         f"Stderr: {stderr.decode()}")
        finally:
            # Kill the process if it's still running
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()

# Performance tests
class PerformanceTests(BaseTestCase):
    """Performance tests for critical operations"""
    
    def test_image_processing_performance(self):
        """Test image processing performance"""
        from utils.image_utils import resize_image, annotate_image
        import time
        
        # Create test images
        num_images = 10
        test_images = []
        for i in range(num_images):
            path = TestConfig.NORMAL_DIR / f"perf_test_{i}.jpg"
            self.create_test_image(path, (1024, 768))
            test_images.append(path)
        
        # Measure resize performance
        start_time = time.time()
        for i, path in enumerate(test_images):
            output_path = TestConfig.NORMAL_DIR / f"perf_resized_{i}.jpg"
            resize_image(path, output_path, (320, 240), "good")
        
        resize_time = time.time() - start_time
        avg_resize_time = resize_time / num_images
        
        # Measure annotation performance
        start_time = time.time()
        for i, path in enumerate(test_images):
            output_path = TestConfig.NORMAL_DIR / f"perf_annotated_{i}.jpg"
            metrics = {"Test": i, "Value": 0.5 + i/10}
            annotate_image(path, output_path, f"Test {i}", metrics)
        
        annotate_time = time.time() - start_time
        avg_annotate_time = annotate_time / num_images
        
        # Print performance results
        print(f"\nPerformance Results:")
        print(f"Average resize time: {avg_resize_time:.6f} seconds")
        print(f"Average annotate time: {avg_annotate_time:.6f} seconds")
        
        # Check performance is within acceptable limits
        # Adjust thresholds based on your system capability
        self.assertLess(avg_resize_time, 0.5, "Resize operation too slow")
        self.assertLess(avg_annotate_time, 0.5, "Annotate operation too slow")
    
    def test_database_performance(self):
        """Test database operation performance"""
        from database.database_manager import DatabaseManager
        import time
        
        # Create database manager with test config
        db_manager = DatabaseManager(TestConfig.TEST_DB_FILE)
        
        # Prepare test data
        num_records = 100
        test_records = []
        for i in range(num_records):
            record = {
                "filename": f"perf_test_{i}.jpg",
                "global_threshold": 0.85,
                "global_error": 0.7 + (i % 30) / 100,
                "is_defect": i % 5 == 0,  # Every 5th record is a defect
                "detection_method": "Global"
            }
            test_records.append(record)
        
        # Measure insert performance
        start_time = time.time()
        for record in test_records:
            db_manager.store_detection_result(record)
        
        insert_time = time.time() - start_time
        avg_insert_time = insert_time / num_records
        
        # Measure query performance
        start_time = time.time()
        results = db_manager.get_recent_detections(50)
        query_time = time.time() - start_time
        
        # Print performance results
        print(f"\nDatabase Performance Results:")
        print(f"Average insert time: {avg_insert_time:.6f} seconds")
        print(f"Query time for 50 records: {query_time:.6f} seconds")
        
        # Check performance is within acceptable limits
        self.assertLess(avg_insert_time, 0.05, "Database insert too slow")
        self.assertLess(query_time, 0.1, "Database query too slow")

# Test runner
def run_tests():
    """Run all tests"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(SettingsManagerTests))
    test_suite.addTest(unittest.makeSuite(ImageProcessorTests))
    test_suite.addTest(unittest.makeSuite(HardwareControllerTests))
    test_suite.addTest(unittest.makeSuite(DatabaseManagerTests))
    test_suite.addTest(unittest.makeSuite(SecurityUtilsTests))
    test_suite.addTest(unittest.makeSuite(IntegrationTests))
    
    # Add performance tests if enabled
    if os.environ.get("RUN_PERFORMANCE_TESTS") == "1":
        test_suite.addTest(unittest.makeSuite(PerformanceTests))
    
    # Add system tests if enabled
    if os.environ.get("RUN_SYSTEM_TESTS") == "1":
        test_suite.addTest(unittest.makeSuite(SystemTests))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(test_suite)

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run tests for defect detection system")
    parser.add_argument("--performance", action="store_true", help="Run performance tests")
    parser.add_argument("--system", action="store_true", help="Run system tests")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    # Set environment variables based on arguments
    if args.performance or args.all:
        os.environ["RUN_PERFORMANCE_TESTS"] = "1"
    
    if args.system or args.all:
        os.environ["RUN_SYSTEM_TESTS"] = "1"
    
    # Run tests
    result = run_tests()
    
    # Exit with appropriate code
    sys.exit(not result.wasSuccessful())