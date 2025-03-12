#!/usr/bin/env python3
"""
System Monitoring Utilities for Defect Detection System
Monitors system health, resource usage, and alerts on critical conditions.
"""

import os
import time
import json
import threading
import platform
import socket
import datetime
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

# Import centralized configuration
from config import config

# Set up logger
logger = config.setup_logger("system_monitor")

# Try to import psutil (optional dependency)
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    logger.warning("psutil not available. Limited system monitoring functionality.")

class SystemMonitor:
    """
    Monitors system health and resource usage.
    Provides alerts for critical conditions.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the system monitor.
        
        Args:
            config_path: Path to the monitor configuration file
        """
        self.config_path = config_path or os.path.join(config.CONFIG_FOLDER, "monitor_config.json")
        self.stats_path = os.path.join(config.BASE_DIR, "system_stats.json")
        self.threshold_alerts = []
        self.monitoring_active = False
        self.monitor_thread = None
        self.monitor_interval = 60  # Default: check every 60 seconds
        
        # Load monitor configuration
        self.load_config()
        
        # System info
        self.system_info = self.get_system_info()
    
    def load_config(self):
        """Load monitor configuration from file"""
        default_config = {
            "enabled": True,
            "interval": 60,  # Check every 60 seconds
            "thresholds": {
                "disk_usage": 90,  # Alert if disk usage exceeds 90%
                "memory_usage": 90,  # Alert if memory usage exceeds 90%
                "cpu_usage": 95,     # Alert if CPU usage exceeds 95%
                "temperature": 80,   # Alert if CPU temperature exceeds 80°C
                "log_size": 100      # Alert if log file exceeds 100 MB
            },
            "backup": {
                "enabled": True,
                "max_history": 30,    # Keep 30 days of history
                "backup_interval": 86400  # Backup stats daily (in seconds)
            },
            "alerts": {
                "email": {
                    "enabled": False,
                    "recipient": "admin@example.com",
                    "from": "monitor@example.com",
                    "smtp_server": "smtp.example.com",
                    "smtp_port": 587,
                    "username": "",
                    "password": ""
                },
                "log": {
                    "enabled": True
                }
            }
        }
        
        # Create config directory if it doesn't exist
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        # Load configuration or create default
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    self.config = json.load(f)
                logger.info(f"Loaded monitor configuration from {self.config_path}")
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading monitor configuration: {e}")
                self.config = default_config
                self.save_config()
        else:
            logger.info(f"Monitor configuration not found. Creating default at {self.config_path}")
            self.config = default_config
            self.save_config()
        
        # Update monitor interval
        self.monitor_interval = self.config.get("interval", 60)
    
    def save_config(self):
        """Save monitor configuration to file"""
        try:
            with open(self.config_path, "w") as f:
                json.dump(self.config, f, indent=4)
            logger.debug(f"Saved monitor configuration to {self.config_path}")
        except (IOError, PermissionError) as e:
            logger.error(f"Error saving monitor configuration: {e}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Collect system information.
        
        Returns:
            Dictionary with system information
        """
        info = {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
            "machine": platform.machine(),
            "system": platform.system(),
            "release": platform.release(),
            "cores": 1  # Default value
        }
        
        # Add psutil information if available
        if HAS_PSUTIL:
            try:
                info["cores"] = psutil.cpu_count(logical=True)
                info["physical_cores"] = psutil.cpu_count(logical=False)
                
                # Memory information
                memory = psutil.virtual_memory()
                info["total_memory"] = memory.total
                info["memory_gb"] = memory.total / (1024 ** 3)
                
                # Disk information
                disk = psutil.disk_usage(config.BASE_DIR)
                info["total_disk"] = disk.total
                info["disk_gb"] = disk.total / (1024 ** 3)
                
                # Network information
                info["network_interfaces"] = list(psutil.net_if_addrs().keys())
                
                # Boot time
                info["boot_time"] = datetime.datetime.fromtimestamp(
                    psutil.boot_time()
                ).strftime("%Y-%m-%d %H:%M:%S")
            except Exception as e:
                logger.error(f"Error collecting system information: {e}")
        
        return info
    
    def collect_system_stats(self) -> Dict[str, Any]:
        """
        Collect current system statistics.
        
        Returns:
            Dictionary with system statistics
        """
        stats = {
            "timestamp": datetime.datetime.now().isoformat(),
            "disk_usage": None,
            "memory_usage": None,
            "cpu_usage": None,
            "temperature": None,
            "log_size": None,
            "uptime": None,
            "inference_stats": self.collect_inference_stats()
        }
        
        # Add disk usage
        try:
            disk_path = config.BASE_DIR
            if os.path.exists(disk_path):
                usage = shutil.disk_usage(disk_path)
                stats["disk_usage"] = {
                    "total": usage.total,
                    "used": usage.used,
                    "free": usage.free,
                    "percent": usage.used / usage.total * 100
                }
        except Exception as e:
            logger.error(f"Error collecting disk usage: {e}")
        
        # Add log file size
        try:
            log_path = config.PATHS.get("LOG_FILE")
            if log_path and os.path.exists(log_path):
                log_size = os.path.getsize(log_path)
                stats["log_size"] = {
                    "bytes": log_size,
                    "mb": log_size / (1024 ** 2)
                }
        except Exception as e:
            logger.error(f"Error collecting log file size: {e}")
        
        # Add psutil stats if available
        if HAS_PSUTIL:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                stats["cpu_usage"] = cpu_percent
                
                # Memory usage
                memory = psutil.virtual_memory()
                stats["memory_usage"] = {
                    "total": memory.total,
                    "available": memory.available,
                    "used": memory.used,
                    "percent": memory.percent
                }
                
                # System uptime
                boot_time = psutil.boot_time()
                uptime = time.time() - boot_time
                stats["uptime"] = {
                    "seconds": uptime,
                    "days": uptime / 86400
                }
                
                # Temperature (if available)
                if hasattr(psutil, "sensors_temperatures"):
                    temps = psutil.sensors_temperatures()
                    if temps:
                        # Find CPU temperature (implementation varies by platform)
                        cpu_temp = None
                        for name, entries in temps.items():
                            if name.lower() in ("cpu", "coretemp", "k10temp", "cpu_thermal"):
                                if entries:
                                    cpu_temp = entries[0].current
                                    break
                        
                        if cpu_temp is not None:
                            stats["temperature"] = cpu_temp
            except Exception as e:
                logger.error(f"Error collecting psutil stats: {e}")
        else:
            # Fallback to basic stats on systems without psutil
            try:
                # Use basic commands to get CPU and memory on Linux
                if platform.system() == "Linux":
                    # Memory usage
                    with open("/proc/meminfo", "r") as f:
                        meminfo = f.read()
                    
                    total = int(self._extract_value(meminfo, "MemTotal:"))
                    free = int(self._extract_value(meminfo, "MemFree:"))
                    available = int(self._extract_value(meminfo, "MemAvailable:"))
                    
                    used = total - available
                    percent = (used / total) * 100
                    
                    stats["memory_usage"] = {
                        "total": total * 1024,  # Convert from KB to bytes
                        "available": available * 1024,
                        "used": used * 1024,
                        "percent": percent
                    }
                    
                    # CPU usage (load average)
                    with open("/proc/loadavg", "r") as f:
                        loadavg = f.read().split()
                    
                    # Estimate CPU percentage from load average and core count
                    cores = os.cpu_count() or 1
                    load_1min = float(loadavg[0])
                    cpu_percent = (load_1min / cores) * 100
                    stats["cpu_usage"] = min(cpu_percent, 100)  # Cap at 100%
                    
                    # Uptime
                    with open("/proc/uptime", "r") as f:
                        uptime_seconds = float(f.read().split()[0])
                    
                    stats["uptime"] = {
                        "seconds": uptime_seconds,
                        "days": uptime_seconds / 86400
                    }
            except Exception as e:
                logger.error(f"Error collecting fallback stats: {e}")
        
        return stats
    
    def _extract_value(self, text, prefix):
        """
        Extract a value from text that starts with the given prefix.
        Used for parsing /proc files.
        
        Args:
            text: Text to search in
            prefix: Prefix to look for
            
        Returns:
            Extracted value as string
        """
        for line in text.splitlines():
            if line.startswith(prefix):
                return line.split(":")[1].strip().split()[0]
        return "0"
    
    def collect_inference_stats(self) -> Dict[str, Any]:
        """
        Collect statistics about the inference system.
        
        Returns:
            Dictionary with inference statistics
        """
        stats = {
            "model_loaded": None,
            "images": {
                "normal_count": 0,
                "anomaly_count": 0
            },
            "settings": {}
        }
        
        # Count images
        try:
            normal_dir = config.PATHS.get("NORMAL_DIR")
            if normal_dir and os.path.exists(normal_dir):
                stats["images"]["normal_count"] = len([
                    f for f in os.listdir(normal_dir)
                    if os.path.isfile(os.path.join(normal_dir, f))
                ])
            
            anomaly_dir = config.PATHS.get("ANOMALY_DIR")
            if anomaly_dir and os.path.exists(anomaly_dir):
                stats["images"]["anomaly_count"] = len([
                    f for f in os.listdir(anomaly_dir)
                    if os.path.isfile(os.path.join(anomaly_dir, f))
                ])
        except Exception as e:
            logger.error(f"Error counting images: {e}")
        
        # Get settings
        try:
            settings_file = config.PATHS.get("SETTINGS_FILE")
            if settings_file and os.path.exists(settings_file):
                with open(settings_file, "r") as f:
                    settings = json.load(f)
                
                # Extract key settings
                stats["settings"] = {
                    "threshold": settings.get("threshold", -1),
                    "patch_threshold": settings.get("patch_threshold", -1),
                    "patch_defect_ratio": settings.get("patch_defect_ratio", -1),
                    "good_count": settings.get("good_count", 0),
                    "bad_count": settings.get("bad_count", 0),
                    "image_counter": settings.get("image_counter", 0),
                    "alarm": settings.get("alarm", 0)
                }
        except Exception as e:
            logger.error(f"Error reading settings: {e}")
        
        # Check if model is loaded
        model_file = config.PATHS.get("MODEL_FILE")
        if model_file:
            stats["model_loaded"] = os.path.exists(model_file)
        
        return stats
    
    def save_stats(self, stats):
        """
        Save system statistics to the stats file.
        
        Args:
            stats: Statistics dictionary
        """
        try:
            # Create stats directory if needed
            os.makedirs(os.path.dirname(self.stats_path), exist_ok=True)
            
            # Load existing stats if file exists
            existing_stats = []
            if os.path.exists(self.stats_path):
                try:
                    with open(self.stats_path, "r") as f:
                        existing_stats = json.load(f)
                    
                    # Ensure it's a list
                    if not isinstance(existing_stats, list):
                        existing_stats = []
                except (json.JSONDecodeError, IOError):
                    existing_stats = []
            
            # Add new stats
            existing_stats.append(stats)
            
            # Keep only recent stats (maximum 1000 entries)
            if len(existing_stats) > 1000:
                existing_stats = existing_stats[-1000:]
            
            # Save to file
            with open(self.stats_path, "w") as f:
                json.dump(existing_stats, f, indent=2)
            
            logger.debug(f"Saved system stats to {self.stats_path}")
        except Exception as e:
            logger.error(f"Error saving system stats: {e}")
    
    def backup_stats(self):
        """
        Backup the stats file to a timestamped file.
        """
        if not self.config.get("backup", {}).get("enabled", False):
            return
        
        try:
            if not os.path.exists(self.stats_path):
                return
            
            # Create backup directory
            backup_dir = os.path.join(config.BASE_DIR, "stats_backups")
            os.makedirs(backup_dir, exist_ok=True)
            
            # Create timestamped backup file
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(backup_dir, f"system_stats_{timestamp}.json")
            
            # Copy the stats file
            shutil.copy2(self.stats_path, backup_path)
            logger.info(f"Backed up system stats to {backup_path}")
            
            # Clean up old backups
            self._cleanup_old_backups(backup_dir)
        except Exception as e:
            logger.error(f"Error backing up system stats: {e}")
    
    def _cleanup_old_backups(self, backup_dir):
        """
        Clean up old backup files beyond the max history setting.
        
        Args:
            backup_dir: Directory containing backup files
        """
        max_history = self.config.get("backup", {}).get("max_history", 30)
        
        try:
            # Get all backup files
            backup_files = []
            for f in os.listdir(backup_dir):
                if f.startswith("system_stats_") and f.endswith(".json"):
                    file_path = os.path.join(backup_dir, f)
                    backup_files.append((file_path, os.path.getmtime(file_path)))
            
            # Sort by modification time (newest first)
            backup_files.sort(key=lambda x: x[1], reverse=True)
            
            # Delete older files beyond the limit
            for file_path, _ in backup_files[max_history:]:
                os.remove(file_path)
                logger.debug(f"Deleted old backup file: {file_path}")
        except Exception as e:
            logger.error(f"Error cleaning up old backups: {e}")
    
    def check_thresholds(self, stats):
        """
        Check if any monitored values exceed their thresholds.
        
        Args:
            stats: Current system statistics
            
        Returns:
            List of threshold alerts
        """
        alerts = []
        thresholds = self.config.get("thresholds", {})
        
        # Check disk usage
        if (stats.get("disk_usage") and 
            thresholds.get("disk_usage") and 
            stats["disk_usage"].get("percent", 0) > thresholds["disk_usage"]):
            alerts.append({
                "type": "disk_usage",
                "message": f"Disk usage exceeds threshold: {stats['disk_usage']['percent']:.1f}% (threshold: {thresholds['disk_usage']}%)",
                "value": stats["disk_usage"]["percent"],
                "threshold": thresholds["disk_usage"],
                "timestamp": stats["timestamp"]
            })
        
        # Check memory usage
        if (stats.get("memory_usage") and 
            thresholds.get("memory_usage") and 
            stats["memory_usage"].get("percent", 0) > thresholds["memory_usage"]):
            alerts.append({
                "type": "memory_usage",
                "message": f"Memory usage exceeds threshold: {stats['memory_usage']['percent']:.1f}% (threshold: {thresholds['memory_usage']}%)",
                "value": stats["memory_usage"]["percent"],
                "threshold": thresholds["memory_usage"],
                "timestamp": stats["timestamp"]
            })
        
        # Check CPU usage
        if (stats.get("cpu_usage") is not None and 
            thresholds.get("cpu_usage") and 
            stats["cpu_usage"] > thresholds["cpu_usage"]):
            alerts.append({
                "type": "cpu_usage",
                "message": f"CPU usage exceeds threshold: {stats['cpu_usage']:.1f}% (threshold: {thresholds['cpu_usage']}%)",
                "value": stats["cpu_usage"],
                "threshold": thresholds["cpu_usage"],
                "timestamp": stats["timestamp"]
            })
        
        # Check temperature
        if (stats.get("temperature") is not None and 
            thresholds.get("temperature") and 
            stats["temperature"] > thresholds["temperature"]):
            alerts.append({
                "type": "temperature",
                "message": f"Temperature exceeds threshold: {stats['temperature']:.1f}°C (threshold: {thresholds['temperature']}°C)",
                "value": stats["temperature"],
                "threshold": thresholds["temperature"],
                "timestamp": stats["timestamp"]
            })
        
        # Check log size
        if (stats.get("log_size") and 
            thresholds.get("log_size") and 
            stats["log_size"].get("mb", 0) > thresholds["log_size"]):
            alerts.append({
                "type": "log_size",
                "message": f"Log file size exceeds threshold: {stats['log_size']['mb']:.1f} MB (threshold: {thresholds['log_size']} MB)",
                "value": stats["log_size"]["mb"],
                "threshold": thresholds["log_size"],
                "timestamp": stats["timestamp"]
            })
        
        return alerts
    
    def handle_alerts(self, alerts):
        """
        Handle threshold alerts by sending notifications.
        
        Args:
            alerts: List of threshold alerts
        """
        if not alerts:
            return
        
        # Store alerts for later retrieval
        self.threshold_alerts.extend(alerts)
        
        # Trim alerts list to most recent 100
        if len(self.threshold_alerts) > 100:
            self.threshold_alerts = self.threshold_alerts[-100:]
        
        # Log alerts
        if self.config.get("alerts", {}).get("log", {}).get("enabled", True):
            for alert in alerts:
                logger.warning(f"THRESHOLD ALERT: {alert['message']}")
        
        # Send email alerts if enabled
        if self.config.get("alerts", {}).get("email", {}).get("enabled", False):
            self._send_email_alerts(alerts)
    
    def _send_email_alerts(self, alerts):
        """
        Send email notifications for alerts.
        
        Args:
            alerts: List of threshold alerts
        """
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            email_config = self.config.get("alerts", {}).get("email", {})
            
            # Create message
            msg = MIMEMultipart()
            msg["From"] = email_config.get("from", "monitor@example.com")
            msg["To"] = email_config.get("recipient", "admin@example.com")
            msg["Subject"] = f"System Alert: {len(alerts)} threshold(s) exceeded"
            
            # Build message body
            body = "The following thresholds have been exceeded:\n\n"
            for alert in alerts:
                body += f"- {alert['message']}\n"
            
            body += f"\nSystem: {self.system_info['hostname']}\n"
            body += f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            
            msg.attach(MIMEText(body, "plain"))
            
            # Send email
            server = smtplib.SMTP(
                email_config.get("smtp_server", "smtp.example.com"),
                email_config.get("smtp_port", 587)
            )
            server.starttls()
            
            # Login if credentials provided
            if email_config.get("username") and email_config.get("password"):
                server.login(email_config["username"], email_config["password"])
            
            # Send message
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Sent email alert to {email_config.get('recipient')}")
        except Exception as e:
            logger.error(f"Error sending email alert: {e}")
    
    def start_monitoring(self):
        """Start the monitoring thread."""
        if self.monitoring_active:
            logger.warning("Monitoring is already active")
            return
        
        if not self.config.get("enabled", True):
            logger.info("Monitoring is disabled in configuration")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"System monitoring started (interval: {self.monitor_interval} seconds)")
    
    def stop_monitoring(self):
        """Stop the monitoring thread."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("System monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop (runs in a separate thread)."""
        last_backup_time = time.time()
        backup_interval = self.config.get("backup", {}).get("backup_interval", 86400)
        
        while self.monitoring_active:
            try:
                # Collect and save system stats
                stats = self.collect_system_stats()
                self.save_stats(stats)
                
                # Check thresholds
                alerts = self.check_thresholds(stats)
                if alerts:
                    self.handle_alerts(alerts)
                
                # Backup stats if interval reached
                current_time = time.time()
                if current_time - last_backup_time >= backup_interval:
                    self.backup_stats()
                    last_backup_time = current_time
                
                # Sleep until next check
                for _ in range(self.monitor_interval):
                    if not self.monitoring_active:
                        break
                    time.sleep(1)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Sleep before retrying on error
    
    def get_recent_stats(self, count=10):
        """
        Get recent system statistics.
        
        Args:
            count: Number of recent stat entries to return
            
        Returns:
            List of recent stat entries
        """
        if not os.path.exists(self.stats_path):
            return []
        
        try:
            with open(self.stats_path, "r") as f:
                stats = json.load(f)
            
            # Ensure it's a list
            if not isinstance(stats, list):
                return []
            
            # Return most recent entries
            return stats[-count:]
        except Exception as e:
            logger.error(f"Error reading system stats: {e}")
            return []
    
    def get_recent_alerts(self, count=10):
        """
        Get recent threshold alerts.
        
        Args:
            count: Number of recent alerts to return
            
        Returns:
            List of recent alerts
        """
        return self.threshold_alerts[-count:]
    
    def get_system_report(self):
        """
        Generate a system health report.
        
        Returns:
            Report as a formatted string
        """
        # Collect current stats
        stats = self.collect_system_stats()
        
        # Build report
        report = []
        report.append("SYSTEM HEALTH REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Hostname: {self.system_info['hostname']}")
        report.append(f"Platform: {self.system_info['platform']}")
        report.append(f"Python: {self.system_info['python_version']}")
        
        # Add hardware info
        report.append("\nHARDWARE INFORMATION")
        report.append("-" * 50)
        report.append(f"Processor: {self.system_info['processor']}")
        report.append(f"Cores: {self.system_info.get('cores', 'N/A')}")
        
        if self.system_info.get('total_memory'):
            memory_gb = self.system_info['total_memory'] / (1024 ** 3)
            report.append(f"Memory: {memory_gb:.2f} GB")
        
        if self.system_info.get('total_disk'):
            disk_gb = self.system_info['total_disk'] / (1024 ** 3)
            report.append(f"Disk: {disk_gb:.2f} GB")
        
        # Add current usage
        report.append("\nCURRENT USAGE")
        report.append("-" * 50)
        
        if stats.get("cpu_usage") is not None:
            report.append(f"CPU Usage: {stats['cpu_usage']:.1f}%")
        
        if stats.get("memory_usage"):
            report.append(f"Memory Usage: {stats['memory_usage']['percent']:.1f}%")
            memory_used_gb = stats['memory_usage']['used'] / (1024 ** 3)
            memory_total_gb = stats['memory_usage']['total'] / (1024 ** 3)
            report.append(f"Memory Used: {memory_used_gb:.2f} GB of {memory_total_gb:.2f} GB")
        
        if stats.get("disk_usage"):
            report.append(f"Disk Usage: {stats['disk_usage']['percent']:.1f}%")
            disk_used_gb = stats['disk_usage']['used'] / (1024 ** 3)
            disk_total_gb = stats['disk_usage']['total'] / (1024 ** 3)
            report.append(f"Disk Used: {disk_used_gb:.2f} GB of {disk_total_gb:.2f} GB")
        
        if stats.get("temperature") is not None:
            report.append(f"Temperature: {stats['temperature']:.1f}°C")
        
        if stats.get("uptime"):
            days = stats['uptime']['days']
            report.append(f"Uptime: {days:.1f} days")
        
        # Add inference stats
        inference_stats = stats.get("inference_stats", {})
        report.append("\nINFERENCE SYSTEM")
        report.append("-" * 50)
        report.append(f"Model Loaded: {'Yes' if inference_stats.get('model_loaded') else 'No'}")
        
        images = inference_stats.get("images", {})
        report.append(f"Normal Images: {images.get('normal_count', 0)}")
        report.append(f"Anomaly Images: {images.get('anomaly_count', 0)}")
        
        settings = inference_stats.get("settings", {})
        if settings:
            report.append(f"Threshold: {settings.get('threshold', 'N/A')}")
            report.append(f"Good Count: {settings.get('good_count', 0)}")
            report.append(f"Bad Count: {settings.get('bad_count', 0)}")
            report.append(f"Alarm State: {'Active' if settings.get('alarm', 0) > 0 else 'Inactive'}")
        
        # Add recent alerts
        alerts = self.get_recent_alerts(5)
        if alerts:
            report.append("\nRECENT ALERTS")
            report.append("-" * 50)
            for alert in alerts:
                report.append(f"- {alert['message']}")
        
        return "\n".join(report)
    
    def clear_old_logs(self, max_size_mb=100):
        """
        Clear old log files if they exceed the maximum size.
        
        Args:
            max_size_mb: Maximum log size in megabytes
            
        Returns:
            Boolean indicating if logs were cleared
        """
        try:
            log_path = config.PATHS.get("LOG_FILE")
            if not log_path or not os.path.exists(log_path):
                return False
            
            # Check log size
            log_size_mb = os.path.getsize(log_path) / (1024 * 1024)
            if log_size_mb <= max_size_mb:
                return False
            
            # Create backup of log file
            backup_dir = os.path.join(config.BASE_DIR, "log_backups")
            os.makedirs(backup_dir, exist_ok=True)
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(backup_dir, f"log_{timestamp}.log")
            
            # Copy log to backup
            shutil.copy2(log_path, backup_path)
            
            # Truncate the log file
            with open(log_path, "w") as f:
                f.write(f"Log truncated at {datetime.datetime.now().isoformat()} (previous size: {log_size_mb:.2f} MB)\n")
            
            logger.info(f"Log file truncated (previous size: {log_size_mb:.2f} MB, backup: {backup_path})")
            return True
        except Exception as e:
            logger.error(f"Error clearing old logs: {e}")
            return False

# For testing
if __name__ == "__main__":
    print("Testing System Monitor")
    
    # Create monitor
    monitor = SystemMonitor()
    
    # Print system info
    print("\nSystem Information:")
    for key, value in monitor.system_info.items():
        print(f"  {key}: {value}")
    
    # Collect and print current stats
    print("\nCurrent System Stats:")
    stats = monitor.collect_system_stats()
    
    if stats.get("cpu_usage") is not None:
        print(f"  CPU Usage: {stats['cpu_usage']:.1f}%")
    
    if stats.get("memory_usage"):
        print(f"  Memory Usage: {stats['memory_usage']['percent']:.1f}%")
    
    if stats.get("disk_usage"):
        print(f"  Disk Usage: {stats['disk_usage']['percent']:.1f}%")
    
    if stats.get("temperature") is not None:
        print(f"  Temperature: {stats['temperature']:.1f}°C")
    
    # Print inference stats
    print("\nInference System Stats:")
    inference_stats = stats.get("inference_stats", {})
    print(f"  Model Loaded: {'Yes' if inference_stats.get('model_loaded') else 'No'}")
    
    images = inference_stats.get("images", {})
    print(f"  Normal Images: {images.get('normal_count', 0)}")
    print(f"  Anomaly Images: {images.get('anomaly_count', 0)}")
    
    # Generate and print system report
    print("\nSystem Report:")
    report = monitor.get_system_report()
    print(report)
    
    print("\nSystem Monitor tests completed.")