#!/usr/bin/env python3
"""
Centralized Configuration for Defect Detection System
This module provides a unified configuration source for all system components.
"""

import os
from pathlib import Path
import json
import logging
from logging.handlers import RotatingFileHandler

class ProjectConfig:
    """Centralized configuration for the defect detection system"""
    
    # Environment variable support
    BASE_DIR = Path(os.environ.get("DEFECT_DETECTION_BASE_DIR", "/home/pierre/Project"))
    
    # Hardware settings with environment variable overrides
    HARDWARE = {
        "TRIGGER_PIN": int(os.environ.get("DEFECT_TRIGGER_PIN", "16")),
        "LED_PIN": os.environ.get("DEFECT_LED_PIN", "board.D18"),
        "NUM_LEDS": int(os.environ.get("DEFECT_NUM_LEDS", "12")),
        "LED_BRIGHTNESS": float(os.environ.get("DEFECT_LED_BRIGHTNESS", "0.4"))
    }
    
    # Paths - dynamically constructed based on BASE_DIR
    PATHS = {
        "NORMAL_DIR": BASE_DIR / "Normal",
        "ANOMALY_DIR": BASE_DIR / "Anomaly",
        "CONFIG_FOLDER": BASE_DIR / "Config_Files",
        "SETTINGS_FILE": BASE_DIR / "Config_Files" / "settings.json",
        "CSV_FILENAME": BASE_DIR / "inference_results.csv",
        "MODEL_FILE": BASE_DIR / "best_autoencoder.pth",
        "BACKUP_DIR": BASE_DIR / "Backup_Photos",
        "BACKUP_CSV_DIR": BASE_DIR / "Backup_CSVs",
        "RESET_SIGNAL_FILE": BASE_DIR / "reset_signal.tmp",
        "UI_SCRIPT": BASE_DIR / "defect_handling_ui_2_dark.py",
        "INFERENCE_SCRIPT": BASE_DIR / "inference_photo_capture.py",
        "APP_ICON": BASE_DIR / "app_icon.png",
        "LOG_FILE": BASE_DIR / "ui_debug.log",
        "INFERENCE_LOG": BASE_DIR / "inference.log"
    }
    
    # Security settings
    SECURITY = {
        # Default password is 1234 (sha256 hashed), can be overridden by environment variable
        "UI_PASSWORD_HASH": os.environ.get("DEFECT_UI_PASSWORD_HASH", 
                                         "03ac674216f3e15c761ee1a5e255f067953623c8b388b4459e13f978d7c846f4"),
        "PASSWORD_SALT": os.environ.get("DEFECT_PASSWORD_SALT", "")  # Add a salt for better security
    }
    
    # Model parameters
    MODEL = {
        "IMAGE_SIZE": (256, 256),
        "NORMALIZE_MEAN": [0.485, 0.456, 0.406],
        "NORMALIZE_STD": [0.229, 0.224, 0.225],
        "LATENT_DIM": 32
    }
    
    # Inference settings
    INFERENCE = {
        "RESET_COUNTER_ON_START": os.environ.get("DEFECT_RESET_COUNTER", "True").lower() == "true",
        "USE_PATCH": True,
        "PATCH_SIZE": 64,
        "STRIDE": 32,
        "GOOD_IMAGE_RESIZE": (640, 480),
        "BAD_IMAGE_RESIZE": (1024, 768),
        "MAX_IMAGES_BEFORE_BACKUP": 1000
    }
    
    # Default settings for settings.json
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
    
    # Logging configuration
    LOGGING = {
        "MAX_LOG_SIZE": 5 * 1024 * 1024,  # 5 MB
        "BACKUP_COUNT": 5,
        "LEVEL": os.environ.get("DEFECT_LOG_LEVEL", "INFO"),
        "FORMAT": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    }
    
    @classmethod
    def ensure_directories(cls):
        """Create all necessary directories"""
        for path_name, path in cls.PATHS.items():
            if path_name.endswith("_DIR") or path_name.endswith("_FOLDER"):
                os.makedirs(path, exist_ok=True)
        
        # Ensure CSV directory exists
        os.makedirs(os.path.dirname(cls.PATHS["CSV_FILENAME"]), exist_ok=True)
    
    @classmethod
    def setup_logger(cls, name, log_file=None):
        """Configure and return a logger with rotating file handler"""
        if log_file is None:
            log_file = cls.PATHS["LOG_FILE"]
            
        # Create log directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Map string level to logging constant
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        level = level_map.get(cls.LOGGING["LEVEL"].upper(), logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(cls.LOGGING["FORMAT"])
        
        # Create file handler
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=cls.LOGGING["MAX_LOG_SIZE"],
            backupCount=cls.LOGGING["BACKUP_COUNT"]
        )
        file_handler.setFormatter(formatter)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # Configure logger
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    @classmethod
    def validate_password(cls, password):
        """Validate a password against the stored hash"""
        import hashlib
        
        # Hash the password with salt if provided
        password_with_salt = password + cls.SECURITY["PASSWORD_SALT"]
        password_hash = hashlib.sha256(password_with_salt.encode()).hexdigest()
        
        # Compare with stored hash
        return password_hash == cls.SECURITY["UI_PASSWORD_HASH"]
    
    @classmethod
    def get_settings(cls):
        """Load settings from the settings file or create with defaults if not found"""
        try:
            if not os.path.exists(cls.PATHS["SETTINGS_FILE"]):
                # Create settings file with defaults
                cls.ensure_directories()
                with open(cls.PATHS["SETTINGS_FILE"], "w") as f:
                    json.dump(cls.DEFAULT_SETTINGS, f, indent=4)
                return dict(cls.DEFAULT_SETTINGS)
            
            # Load existing settings
            with open(cls.PATHS["SETTINGS_FILE"], "r") as f:
                return json.load(f)
        except Exception as e:
            logger = cls.setup_logger("config")
            logger.error(f"Failed to load settings: {e}")
            return dict(cls.DEFAULT_SETTINGS)
    
    @classmethod
    def save_settings(cls, settings):
        """Save settings to the settings file"""
        try:
            cls.ensure_directories()
            with open(cls.PATHS["SETTINGS_FILE"], "w") as f:
                json.dump(settings, f, indent=4)
            return True
        except Exception as e:
            logger = cls.setup_logger("config")
            logger.error(f"Failed to save settings: {e}")
            return False

# Create a singleton instance for convenience
config = ProjectConfig()

# Initialize logger for this module
logger = config.setup_logger("config")
logger.info("Configuration module loaded")

if __name__ == "__main__":
    # When run directly, print config for debugging
    import pprint
    print("Defect Detection System Configuration:")
    print(f"BASE_DIR: {config.BASE_DIR}")
    print("\nHARDWARE:")
    pprint.pprint(config.HARDWARE)
    print("\nKey Paths:")
    for key, path in config.PATHS.items():
        print(f"{key}: {path}")
    
    # Test directory creation
    config.ensure_directories()
    print("\nAll directories created successfully.")
