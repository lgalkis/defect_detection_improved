#!/usr/bin/env python3
"""
Installation and Setup Script for Defect Detection System
Sets up the environment, installs dependencies, and configures the system.
"""

import os
import sys
import shutil
import subprocess
import argparse
import json
import platform
import getpass
import logging
import traceback
import time
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("setup_log.txt", mode="w")
    ]
)
logger = logging.getLogger("setup")

# Default configuration
DEFAULT_CONFIG = {
    "BASE_DIR": "/home/pierre/Project",
    "MODEL_FILE": "best_autoencoder.pth",
    "THRESHOLD": 0.85,
    "PATCH_THRESHOLD": 0.7,
    "PATCH_DEFECT_RATIO": 0.45,
    "CAMERA_ENABLED": True,
    "GPIO_ENABLED": True,
    "LED_ENABLED": True,
    "UI_ENABLED": True,
    "MONITOR_ENABLED": True,
    "DEPENDENCIES": {
        "required": [
            "torch", 
            "torchvision", 
            "numpy", 
            "Pillow", 
            "matplotlib"
        ],
        "optional": [
            "psutil", 
            "pandas"
        ],
        "hardware": [
            "RPi.GPIO", 
            "picamera2", 
            "adafruit-circuitpython-neopixel"
        ],
        "ui": [
            "PyQt5"
        ]
    }
}

class SetupManager:
    """
    Manages the setup process for the defect detection system.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the setup manager.
        
        Args:
            config_path: Path to the configuration file (optional)
        """
        self.config = dict(DEFAULT_CONFIG)
        self.config_path = config_path or "setup_config.json"
        self.setup_success = False
        self.has_sudo = self._check_sudo_access()
        
        # Load custom configuration if provided
        if os.path.exists(self.config_path):
            self._load_config()
    
    def _load_config(self):
        """Load configuration from file"""
        try:
            with open(self.config_path, "r") as f:
                loaded_config = json.load(f)
            
            # Update config with loaded values
            for key, value in loaded_config.items():
                self.config[key] = value
            
            logger.info(f"Loaded configuration from {self.config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
    
    def _save_config(self):
        """Save current configuration to file"""
        try:
            with open(self.config_path, "w") as f:
                json.dump(self.config, indent=4, sort_keys=True, fp=f)
            logger.info(f"Saved configuration to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def _check_sudo_access(self):
        """Check if the script has sudo/root access"""
        if os.name == "nt":  # Windows
            import ctypes
            return ctypes.windll.shell32.IsUserAnAdmin() != 0
        else:  # Unix/Linux
            return os.geteuid() == 0
    
    def _run_command(self, command, verbose=True):
        """
        Run a shell command and return the result.
        
        Args:
            command: Command to run (string or list)
            verbose: Whether to log the output
            
        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        try:
            if isinstance(command, list):
                cmd_str = " ".join(command)
            else:
                cmd_str = command
                command = command.split()
            
            if verbose:
                logger.info(f"Running command: {cmd_str}")
            
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if verbose:
                if result.stdout:
                    for line in result.stdout.splitlines():
                        logger.debug(f"STDOUT: {line}")
                
                if result.stderr:
                    for line in result.stderr.splitlines():
                        logger.debug(f"STDERR: {line}")
            
            return result.returncode, result.stdout, result.stderr
        except Exception as e:
            logger.error(f"Error running command '{cmd_str}': {e}")
            return -1, "", str(e)
    
    def check_system_compatibility(self):
        """
        Check if the current system is compatible with the defect detection system.
        
        Returns:
            Tuple of (is_compatible, messages)
        """
        compatible = True
        messages = []
        
        # Check Python version
        python_version = tuple(map(int, platform.python_version_tuple()))
        if python_version < (3, 8):
            compatible = False
            messages.append(f"Python version 3.8 or higher is required. Current version: {platform.python_version()}")
        else:
            messages.append(f"Python version: {platform.python_version()} [OK]")
        
        # Check platform
        system = platform.system()
        messages.append(f"Platform: {system} {platform.release()}")
        
        if system == "Linux":
            # Check for Raspberry Pi
            is_raspberry_pi = False
            if os.path.exists("/proc/device-tree/model"):
                with open("/proc/device-tree/model", "r") as f:
                    model = f.read()
                    if "Raspberry Pi" in model:
                        is_raspberry_pi = True
                        messages.append(f"Detected Raspberry Pi: {model}")
            
            # Check for camera
            camera_detected = False
            if os.path.exists("/dev/video0"):
                camera_detected = True
                messages.append("Camera detected [OK]")
            elif self.config["CAMERA_ENABLED"]:
                compatible = False
                messages.append("Camera not detected but is enabled in configuration")
            
            # Check for GPIO
            gpio_detected = False
            if is_raspberry_pi:
                gpio_detected = True
                messages.append("GPIO detected [OK]")
            elif self.config["GPIO_ENABLED"]:
                compatible = False
                messages.append("GPIO not detected but is enabled in configuration")
        
        # Check for CUDA if using torch
        if "torch" in self.config["DEPENDENCIES"]["required"]:
            try:
                import torch
                if torch.cuda.is_available():
                    messages.append(f"CUDA available: {torch.cuda.get_device_name(0)} [OK]")
                else:
                    messages.append("CUDA not available. CPU will be used for inference (slower).")
            except ImportError:
                messages.append("PyTorch not installed yet.")
        
        return compatible, messages
    
    def check_dependencies(self):
        """
        Check if required dependencies are installed.
        
        Returns:
            Tuple of (all_installed, missing_required, missing_optional)
        """
        missing_required = []
        missing_optional = []
        installed = []
        
        # Check required dependencies
        for package in self.config["DEPENDENCIES"]["required"]:
            if not self._is_package_installed(package):
                missing_required.append(package)
            else:
                installed.append(package)
        
        # Check optional dependencies
        for package in self.config["DEPENDENCIES"]["optional"]:
            if not self._is_package_installed(package):
                missing_optional.append(package)
            else:
                installed.append(package)
        
        # Check hardware dependencies if enabled
        if self.config["CAMERA_ENABLED"] or self.config["GPIO_ENABLED"] or self.config["LED_ENABLED"]:
            for package in self.config["DEPENDENCIES"]["hardware"]:
                if not self._is_package_installed(package):
                    if (package == "RPi.GPIO" and self.config["GPIO_ENABLED"]) or \
                       (package == "picamera2" and self.config["CAMERA_ENABLED"]) or \
                       (package == "adafruit-circuitpython-neopixel" and self.config["LED_ENABLED"]):
                        missing_required.append(package)
                    else:
                        missing_optional.append(package)
                else:
                    installed.append(package)
        
        # Check UI dependencies if enabled
        if self.config["UI_ENABLED"]:
            for package in self.config["DEPENDENCIES"]["ui"]:
                if not self._is_package_installed(package):
                    missing_required.append(package)
                else:
                    installed.append(package)
        
        all_installed = len(missing_required) == 0
        
        return all_installed, missing_required, missing_optional, installed
    
    def _is_package_installed(self, package_name):
        """
        Check if a Python package is installed.
        
        Args:
            package_name: Name of the package
            
        Returns:
            Boolean indicating if the package is installed
        """
        try:
            # Try to import the package
            package = __import__(package_name)
            return True
        except ImportError:
            # Special case for packages with different import names
            if package_name == "Pillow":
                try:
                    import PIL
                    return True
                except ImportError:
                    return False
            elif package_name == "adafruit-circuitpython-neopixel":
                try:
                    import neopixel
                    return True
                except ImportError:
                    return False
            elif package_name == "picamera2":
                try:
                    from picamera2 import Picamera2
                    return True
                except ImportError:
                    return False
            return False
    
    def install_dependencies(self, required_only=False):
        """
        Install missing dependencies.
        
        Args:
            required_only: Whether to install only required dependencies
            
        Returns:
            Boolean indicating success
        """
        all_installed, missing_required, missing_optional, installed = self.check_dependencies()
        
        if all_installed and required_only:
            logger.info("All required dependencies are already installed.")
            return True
        
        # Combine lists of packages to install
        to_install = list(missing_required)
        if not required_only:
            to_install.extend(missing_optional)
        
        if not to_install:
            logger.info("No dependencies to install.")
            return True
        
        logger.info(f"Installing dependencies: {', '.join(to_install)}")
        
        # Install packages using pip
        pip_cmd = [sys.executable, "-m", "pip", "install"]
        
        for package in to_install:
            cmd = pip_cmd + [package]
            logger.info(f"Installing {package}...")
            
            returncode, stdout, stderr = self._run_command(cmd)
            
            if returncode != 0:
                logger.error(f"Failed to install {package}. Error: {stderr}")
                
                # Try installing with sudo if available
                if self.has_sudo and platform.system() != "Windows":
                    logger.info(f"Trying to install {package} with sudo...")
                    sudo_cmd = ["sudo"] + cmd
                    returncode, stdout, stderr = self._run_command(sudo_cmd)
                    
                    if returncode != 0:
                        logger.error(f"Failed to install {package} with sudo. Error: {stderr}")
                        return False
            
            logger.info(f"Successfully installed {package}")
        
        return True
    
    def setup_directories(self):
        """
        Create necessary directories for the defect detection system.
        
        Returns:
            Boolean indicating success
        """
        base_dir = self.config["BASE_DIR"]
        
        directories = [
            base_dir,
            os.path.join(base_dir, "Normal"),
            os.path.join(base_dir, "Anomaly"),
            os.path.join(base_dir, "Config_Files"),
            os.path.join(base_dir, "Backup_Photos"),
            os.path.join(base_dir, "Backup_CSVs"),
            os.path.join(base_dir, "models"),
            os.path.join(base_dir, "logs"),
            os.path.join(base_dir, "visualizations")
        ]
        
        success = True
        
        for directory in directories:
            try:
                os.makedirs(directory, exist_ok=True)
                logger.info(f"Created directory: {directory}")
            except Exception as e:
                logger.error(f"Failed to create directory {directory}: {e}")
                success = False
                
                # Try with sudo if available
                if self.has_sudo and platform.system() != "Windows":
                    try:
                        sudo_cmd = ["sudo", "mkdir", "-p", directory]
                        returncode, stdout, stderr = self._run_command(sudo_cmd)
                        
                        if returncode == 0:
                            logger.info(f"Created directory with sudo: {directory}")
                            
                            # Change ownership to current user
                            username = getpass.getuser()
                            chown_cmd = ["sudo", "chown", "-R", f"{username}:{username}", directory]
                            self._run_command(chown_cmd)
                            
                            success = True
                        else:
                            logger.error(f"Failed to create directory with sudo: {stderr}")
                    except Exception as e:
                        logger.error(f"Error using sudo to create directory: {e}")
        
        return success
    
    def create_settings_file(self):
        """
        Create the settings file for the defect detection system.
        
        Returns:
            Boolean indicating success
        """
        config_folder = os.path.join(self.config["BASE_DIR"], "Config_Files")
        settings_file = os.path.join(config_folder, "settings.json")
        
        # Create default settings
        settings = {
            "threshold": self.config["THRESHOLD"],
            "patch_threshold": self.config["PATCH_THRESHOLD"],
            "patch_defect_ratio": self.config["PATCH_DEFECT_RATIO"],
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
        
        try:
            # Create config folder if it doesn't exist
            os.makedirs(config_folder, exist_ok=True)
            
            # Write settings file
            with open(settings_file, "w") as f:
                json.dump(settings, f, indent=4)
            
            # Set permissions to allow all users to read/write
            if platform.system() != "Windows":
                os.chmod(settings_file, 0o666)
            
            logger.info(f"Created settings file: {settings_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to create settings file: {e}")
            return False
    
    def download_model(self, model_url=None):
        """
        Download the model file if it doesn't exist.
        
        Args:
            model_url: URL to download the model from (optional)
            
        Returns:
            Boolean indicating success
        """
        model_path = os.path.join(self.config["BASE_DIR"], self.config["MODEL_FILE"])
        
        # If model already exists, skip download
        if os.path.exists(model_path):
            logger.info(f"Model file already exists: {model_path}")
            return True
        
        # If no URL provided, prompt user to provide model file manually
        if not model_url:
            logger.info("No model URL provided.")
            logger.info(f"Please manually place the model file at: {model_path}")
            return False
        
        try:
            import requests
            
            logger.info(f"Downloading model from {model_url}...")
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            
            with open(model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Downloaded model to {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            logger.info(f"Please manually place the model file at: {model_path}")
            return False
    
    def create_systemd_service(self):
        """
        Create a systemd service file for auto-starting the defect detection system.
        Only applicable for Linux systems.
        
        Returns:
            Boolean indicating success
        """
        if platform.system() != "Linux":
            logger.info("Systemd service creation is only supported on Linux.")
            return False
        
        if not self.has_sudo:
            logger.warning("Sudo access is required to create systemd service.")
            return False
        
        service_name = "defect-detection.service"
        service_path = f"/etc/systemd/system/{service_name}"
        
        # Create service file content
        service_content = f"""[Unit]
Description=Defect Detection System
After=network.target

[Service]
User={getpass.getuser()}
WorkingDirectory={self.config["BASE_DIR"]}
ExecStart={sys.executable} {os.path.join(self.config["BASE_DIR"], "main.py")}
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
        
        try:
            # Write service file to temporary location
            temp_service_path = os.path.join(self.config["BASE_DIR"], service_name)
            with open(temp_service_path, "w") as f:
                f.write(service_content)
            
            # Copy to system directory with sudo
            sudo_cmd = ["sudo", "cp", temp_service_path, service_path]
            returncode, stdout, stderr = self._run_command(sudo_cmd)
            
            if returncode != 0:
                logger.error(f"Failed to create systemd service: {stderr}")
                return False
            
            # Set permissions
            self._run_command(["sudo", "chmod", "644", service_path])
            
            # Reload systemd
            self._run_command(["sudo", "systemctl", "daemon-reload"])
            
            logger.info(f"Created systemd service: {service_name}")
            logger.info("To enable the service, run: sudo systemctl enable defect-detection")
            logger.info("To start the service, run: sudo systemctl start defect-detection")
            
            # Clean up temporary file
            os.remove(temp_service_path)
            
            return True
        except Exception as e:
            logger.error(f"Failed to create systemd service: {e}")
            return False
    
    def copy_source_files(self, source_dir):
        """
        Copy source files from the provided directory to the base directory.
        
        Args:
            source_dir: Directory containing source files
            
        Returns:
            Boolean indicating success
        """
        if not os.path.exists(source_dir):
            logger.error(f"Source directory does not exist: {source_dir}")
            return False
        
        base_dir = self.config["BASE_DIR"]
        
        try:
            # Copy all Python files
            for filename in os.listdir(source_dir):
                if filename.endswith(".py"):
                    src_path = os.path.join(source_dir, filename)
                    dst_path = os.path.join(base_dir, filename)
                    
                    shutil.copy2(src_path, dst_path)
                    logger.info(f"Copied {filename} to {base_dir}")
                    
                    # Make scripts executable
                    if platform.system() != "Windows":
                        os.chmod(dst_path, 0o755)
            
            return True
        except Exception as e:
            logger.error(f"Failed to copy source files: {e}")
            return False
    
    def show_completion_message(self, successful):
        """
        Show a completion message with next steps.
        
        Args:
            successful: Whether the setup was successful
        """
        if successful:
            logger.info("=" * 80)
            logger.info("SETUP COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            logger.info(f"Base directory: {self.config['BASE_DIR']}")
            logger.info("\nNext steps:")
            logger.info("1. Navigate to the base directory:")
            logger.info(f"   cd {self.config['BASE_DIR']}")
            logger.info("2. Run the main application:")
            logger.info("   python main.py")
            
            if platform.system() == "Linux":
                logger.info("\nTo start the system automatically at boot:")
                logger.info("   sudo systemctl enable defect-detection")
                logger.info("   sudo systemctl start defect-detection")
            
            logger.info("\nFor more information, refer to the documentation.")
        else:
            logger.info("=" * 80)
            logger.info("SETUP COMPLETED WITH ERRORS")
            logger.info("=" * 80)
            logger.info("Please review the log file for details: setup_log.txt")
            logger.info("Fix the issues and run the setup script again.")
    
    def run_setup(self):
        """
        Run the complete setup process.
        
        Returns:
            Boolean indicating success
        """
        logger.info("=" * 80)
        logger.info("DEFECT DETECTION SYSTEM SETUP")
        logger.info("=" * 80)
        
        # Check system compatibility
        logger.info("\nChecking system compatibility...")
        compatible, messages = self.check_system_compatibility()
        
        for message in messages:
            logger.info(message)
        
        if not compatible:
            logger.error("System is not compatible with the defect detection system.")
            logger.error("Please resolve the issues and run the setup script again.")
            return False
        
        # Check and install dependencies
        logger.info("\nChecking dependencies...")
        all_installed, missing_required, missing_optional, installed = self.check_dependencies()
        
        if installed:
            logger.info(f"Installed packages: {', '.join(installed)}")
        
        if missing_required:
            logger.info(f"Missing required packages: {', '.join(missing_required)}")
            
            # Install required dependencies
            if not self.install_dependencies(required_only=True):
                logger.error("Failed to install required dependencies.")
                return False
        
        if missing_optional:
            logger.info(f"Missing optional packages: {', '.join(missing_optional)}")
            
            # Ask if user wants to install optional dependencies
            print("\nDo you want to install optional dependencies? (y/n) ", end="")
            choice = input().strip().lower()
            
            if choice == "y" or choice == "yes":
                if not self.install_dependencies(required_only=False):
                    logger.warning("Failed to install some optional dependencies.")
        
        # Create directories
        logger.info("\nCreating directories...")
        if not self.setup_directories():
            logger.error("Failed to create all required directories.")
            return False
        
        # Create settings file
        logger.info("\nCreating settings file...")
        if not self.create_settings_file():
            logger.error("Failed to create settings file.")
            return False
        
        # Download model
        logger.info("\nChecking model file...")
        model_url = None  # Replace with actual URL if available
        self.download_model(model_url)
        
        # Create systemd service (Linux only)
        if platform.system() == "Linux":
            logger.info("\nSetting up systemd service...")
            self.create_systemd_service()
        
        # Save configuration
        self._save_config()
        
        self.setup_success = True
        return True

def main():
    """Main function to run the setup script"""
    parser = argparse.ArgumentParser(description="Setup script for defect detection system")
    parser.add_argument("--config", "-c", help="Path to configuration file")
    parser.add_argument("--base-dir", "-d", help="Base directory for installation")
    parser.add_argument("--source-dir", "-s", help="Directory containing source files to copy")
    parser.add_argument("--no-model", action="store_true", help="Skip model download")
    parser.add_argument("--no-deps", action="store_true", help="Skip dependency installation")
    parser.add_argument("--no-service", action="store_true", help="Skip systemd service creation")
    args = parser.parse_args()
    
    # Create setup manager
    setup_manager = SetupManager(config_path=args.config)
    
    # Override configuration with command-line arguments
    if args.base_dir:
        setup_manager.config["BASE_DIR"] = args.base_dir
    
    # Copy source files if provided
    if args.source_dir:
        setup_manager.copy_source_files(args.source_dir)
    
    # Run setup
    successful = setup_manager.run_setup()
    
    # Show completion message
    setup_manager.show_completion_message(successful)
    
    return 0 if successful else 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("\nSetup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)