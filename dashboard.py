#!/usr/bin/env python3
"""
Dashboard for Defect Detection System
Provides real-time monitoring and visualization of detection results.
"""

import os
import sys
import argparse
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib
matplotlib.use('Qt5Agg')  # Use Qt5 backend for interactive plots

# Import PyQt5 for UI
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QTabWidget, QGridLayout, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog,
    QSplitter, QFrame, QProgressBar, QDateEdit
)
from PyQt5.QtCore import QTimer, Qt, QDateTime, pyqtSlot
from PyQt5.QtGui import QColor, QPixmap, QIcon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Import our modules
from config import config
from database.database_manager import DatabaseManager
from utils.system_monitor import SystemMonitor
from utils.settings_manager import SettingsManager
from utils.data_visualization import DataVisualizer

class MatplotlibCanvas(FigureCanvas):
    """Matplotlib canvas for embedding plots in Qt widgets."""
    
    def __init__(self, width=5, height=4, dpi=100):
        """Initialize the canvas."""
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MatplotlibCanvas, self).__init__(self.fig)
        self.fig.tight_layout()

class DashboardWindow(QMainWindow):
    """Main dashboard window for the defect detection system."""
    
    def __init__(self, args):
        """Initialize the dashboard window."""
        super().__init__()
        self.args = args
        
        # Initialize components
        self.db_manager = DatabaseManager()
        self.settings_manager = SettingsManager()
        self.system_monitor = SystemMonitor()
        self.data_visualizer = DataVisualizer()
        
        # Start system monitor
        self.system_monitor.start_monitoring()
        
        # Setup UI
        self.setWindowTitle("Defect Detection Dashboard")
        self.setGeometry(100, 100, 1200, 800)
        self.setup_ui()
        
        # Setup update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_dashboard)
        self.update_timer.start(5000)  # Update every 5 seconds
        
        # Initial update
        self.update_dashboard()
    
    def setup_ui(self):
        """Setup the user interface."""
        # Create central widget and main layout
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        tab_widget = QTabWidget()
        
        # Create tabs
        overview_tab = self.create_overview_tab()
        defect_history_tab = self.create_defect_history_tab()
        system_monitor_tab = self.create_system_monitor_tab()
        image_browser_tab = self.create_image_browser_tab()
        analytics_tab = self.create_analytics_tab()
        
        # Add tabs to tab widget
        tab_widget.addTab(overview_tab, "Overview")
        tab_widget.addTab(defect_history_tab, "Defect History")
        tab_widget.addTab(system_monitor_tab, "System Monitor")
        tab_widget.addTab(image_browser_tab, "Image Browser")
        tab_widget.addTab(analytics_tab, "Analytics")
        
        # Add tab widget to main layout
        main_layout.addWidget(tab_widget)
        
        # Create status bar
        self.statusBar().showMessage("Ready")
        
        # Set central widget
        self.setCentralWidget(central_widget)
    
    def create_overview_tab(self):
        """Create the overview tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Top section with key metrics
        metrics_frame = QFrame()
        metrics_frame.setFrameShape(QFrame.StyledPanel)
        metrics_layout = QGridLayout(metrics_frame)
        
        # Add key metrics
        self.total_inspections_label = QLabel("Total Inspections: 0")
        self.defect_rate_label = QLabel("Defect Rate: 0.0%")
        self.current_threshold_label = QLabel("Current Threshold: 0.0")
        self.last_inspection_label = QLabel("Last Inspection: Never")
        
        metrics_layout.addWidget(QLabel("<h3>Key Metrics</h3>"), 0, 0, 1, 2)
        metrics_layout.addWidget(self.total_inspections_label, 1, 0)
        metrics_layout.addWidget(self.defect_rate_label, 1, 1)
        metrics_layout.addWidget(self.current_threshold_label, 2, 0)
        metrics_layout.addWidget(self.last_inspection_label, 2, 1)
        
        # Add defect rate chart
        self.defect_rate_canvas = MatplotlibCanvas(width=8, height=3)
        
        # Add recent defects table
        recent_defects_table = QTableWidget()
        recent_defects_table.setColumnCount(4)
        recent_defects_table.setHorizontalHeaderLabels(["Time", "Filename", "Error", "Method"])
        recent_defects_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.recent_defects_table = recent_defects_table
        
        # Add components to layout
        layout.addWidget(metrics_frame)
        layout.addWidget(self.defect_rate_canvas)
        layout.addWidget(QLabel("<h3>Recent Defects</h3>"))
        layout.addWidget(recent_defects_table)
        
        return tab
    
    def create_defect_history_tab(self):
        """Create the defect history tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Controls section
        controls_layout = QHBoxLayout()
        
        # Time range selector
        controls_layout.addWidget(QLabel("Time Range:"))
        time_range_combo = QComboBox()
        time_range_combo.addItems(["Last 24 Hours", "Last 7 Days", "Last 30 Days", "All Time"])
        time_range_combo.currentIndexChanged.connect(self.update_defect_history)
        self.time_range_combo = time_range_combo
        controls_layout.addWidget(time_range_combo)
        
        # Date range selectors
        controls_layout.addWidget(QLabel("Start Date:"))
        start_date = QDateEdit()
        start_date.setCalendarPopup(True)
        start_date.setDateTime(QDateTime.currentDateTime().addDays(-7))
        self.start_date = start_date
        controls_layout.addWidget(start_date)
        
        controls_layout.addWidget(QLabel("End Date:"))
        end_date = QDateEdit()
        end_date.setCalendarPopup(True)
        end_date.setDateTime(QDateTime.currentDateTime())
        self.end_date = end_date
        controls_layout.addWidget(end_date)
        
        # Apply button
        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self.update_defect_history)
        controls_layout.addWidget(apply_button)
        
        # Export button
        export_button = QPushButton("Export Data")
        export_button.clicked.connect(self.export_defect_history)
        controls_layout.addWidget(export_button)
        
        # Create chart
        self.defect_history_canvas = MatplotlibCanvas(width=10, height=5)
        
        # Create table
        history_table = QTableWidget()
        history_table.setColumnCount(5)
        history_table.setHorizontalHeaderLabels(["Date", "Total", "Defects", "Rate", "Avg Error"])
        history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.history_table = history_table
        
        # Add components to layout
        layout.addLayout(controls_layout)
        layout.addWidget(self.defect_history_canvas)
        layout.addWidget(QLabel("<h3>Daily Summary</h3>"))
        layout.addWidget(history_table)
        
        return tab
    
    def create_system_monitor_tab(self):
        """Create the system monitor tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # System info panel
        info_frame = QFrame()
        info_frame.setFrameShape(QFrame.StyledPanel)
        info_layout = QGridLayout(info_frame)
        
        self.cpu_usage_label = QLabel("CPU Usage: 0%")
        self.memory_usage_label = QLabel("Memory Usage: 0%")
        self.disk_usage_label = QLabel("Disk Usage: 0%")
        self.temperature_label = QLabel("Temperature: N/A")
        self.uptime_label = QLabel("Uptime: N/A")
        
        info_layout.addWidget(QLabel("<h3>System Status</h3>"), 0, 0, 1, 2)
        info_layout.addWidget(self.cpu_usage_label, 1, 0)
        info_layout.addWidget(self.memory_usage_label, 1, 1)
        info_layout.addWidget(self.disk_usage_label, 2, 0)
        info_layout.addWidget(self.temperature_label, 2, 1)
        info_layout.addWidget(self.uptime_label, 3, 0, 1, 2)
        
        # Progress bars for resource usage
        progress_frame = QFrame()
        progress_frame.setFrameShape(QFrame.StyledPanel)
        progress_layout = QGridLayout(progress_frame)
        
        progress_layout.addWidget(QLabel("CPU:"), 0, 0)
        self.cpu_progress = QProgressBar()
        progress_layout.addWidget(self.cpu_progress, 0, 1)
        
        progress_layout.addWidget(QLabel("Memory:"), 1, 0)
        self.memory_progress = QProgressBar()
        progress_layout.addWidget(self.memory_progress, 1, 1)
        
        progress_layout.addWidget(QLabel("Disk:"), 2, 0)
        self.disk_progress = QProgressBar()
        progress_layout.addWidget(self.disk_progress, 2, 1)
        
        # System metrics chart
        self.system_metrics_canvas = MatplotlibCanvas(width=10, height=4)
        
        # Alert log
        self.alert_table = QTableWidget()
        self.alert_table.setColumnCount(3)
        self.alert_table.setHorizontalHeaderLabels(["Time", "Type", "Message"])
        self.alert_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # Add components to layout
        layout.addWidget(info_frame)
        layout.addWidget(progress_frame)
        layout.addWidget(self.system_metrics_canvas)
        layout.addWidget(QLabel("<h3>System Alerts</h3>"))
        layout.addWidget(self.alert_table)
        
        return tab
    
    def create_image_browser_tab(self):
        """Create the image browser tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Controls section
        controls_layout = QHBoxLayout()
        
        # Image type selector
        controls_layout.addWidget(QLabel("Image Type:"))
        image_type_combo = QComboBox()
        image_type_combo.addItems(["Normal", "Defective", "All"])
        image_type_combo.currentIndexChanged.connect(self.update_image_browser)
        self.image_type_combo = image_type_combo
        controls_layout.addWidget(image_type_combo)
        
        # Sort selector
        controls_layout.addWidget(QLabel("Sort By:"))
        sort_combo = QComboBox()
        sort_combo.addItems(["Newest First", "Oldest First", "Error (High to Low)", "Error (Low to High)"])
        sort_combo.currentIndexChanged.connect(self.update_image_browser)
        self.sort_combo = sort_combo
        controls_layout.addWidget(sort_combo)
        
        # Refresh button
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.update_image_browser)
        controls_layout.addWidget(refresh_button)
        
        # Image grid
        # This would typically be implemented as a custom widget
        # For simplicity, we'll use a table
        image_table = QTableWidget()
        image_table.setColumnCount(3)
        image_table.setHorizontalHeaderLabels(["Thumbnail", "Filename", "Details"])
        image_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.image_table = image_table
        
        # Add components to layout
        layout.addLayout(controls_layout)
        layout.addWidget(self.image_table)
        
        return tab
    
    def create_analytics_tab(self):
        """Create the analytics tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Error distribution chart
        layout.addWidget(QLabel("<h3>Error Distribution</h3>"))
        self.error_dist_canvas = MatplotlibCanvas(width=10, height=4)
        layout.addWidget(self.error_dist_canvas)
        
        # Metrics over time
        layout.addWidget(QLabel("<h3>Metrics Over Time</h3>"))
        self.metrics_canvas = MatplotlibCanvas(width=10, height=4)
        layout.addWidget(self.metrics_canvas)
        
        return tab
    
    def update_dashboard(self):
        """Update all dashboard components."""
        try:
            # Update overview tab
            self.update_overview()
            
            # Update system monitor tab
            self.update_system_monitor()
            
            # Update status bar
            self.statusBar().showMessage(f"Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        except Exception as e:
            self.statusBar().showMessage(f"Error updating dashboard: {str(e)}")
    
    def update_overview(self):
        """Update the overview tab."""
        try:
            # Get current settings
            settings = self.settings_manager.get_settings()
            
            # Update key metrics
            good_count = settings.get("good_count", 0)
            bad_count = settings.get("bad_count", 0)
            total_count = good_count + bad_count
            
            self.total_inspections_label.setText(f"Total Inspections: {total_count}")
            
            if total_count > 0:
                defect_rate = (bad_count / total_count) * 100
                self.defect_rate_label.setText(f"Defect Rate: {defect_rate:.2f}%")
            
            self.current_threshold_label.setText(f"Current Threshold: {settings.get('threshold', 0.0):.4f}")
            
            # Get recent detections
            recent = self.db_manager.get_recent_detections(10)
            
            # Update recent defects table
            self.recent_defects_table.setRowCount(0)
            for i, record in enumerate(recent):
                if record.get("is_defect", 0) == 1:
                    self.recent_defects_table.insertRow(i)
                    
                    # Parse timestamp
                    timestamp = datetime.datetime.fromisoformat(record["timestamp"])
                    time_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                    
                    self.recent_defects_table.setItem(i, 0, QTableWidgetItem(time_str))
                    self.recent_defects_table.setItem(i, 1, QTableWidgetItem(record.get("filename", "")))
                    self.recent_defects_table.setItem(i, 2, QTableWidgetItem(f"{record.get('global_error', 0.0):.4f}"))
                    self.recent_defects_table.setItem(i, 3, QTableWidgetItem(record.get("detection_method", "")))
            
            # Update defect rate chart
            self.update_defect_rate_chart()
            
        except Exception as e:
            print(f"Error updating overview: {e}")
    
    def update_defect_rate_chart(self):
        """Update the defect rate chart."""
        try:
            # Get defect rate data
            defect_rate_data = self.db_manager.get_defect_rate_by_date(30)
            
            # Clear chart
            ax = self.defect_rate_canvas.axes
            ax.clear()
            
            if not defect_rate_data:
                ax.text(0.5, 0.5, "No data available", ha='center', va='center')
                self.defect_rate_canvas.draw()
                return
            
            # Prepare data
            dates = []
            rates = []
            
            for record in defect_rate_data:
                date = datetime.datetime.fromisoformat(record["date"]).date()
                dates.append(date)
                
                if record["total_inspections"] > 0:
                    rate = (record["defects_found"] / record["total_inspections"]) * 100
                else:
                    rate = 0
                rates.append(rate)
            
            # Plot data
            ax.bar(dates, rates, color='red', alpha=0.7)
            ax.set_ylabel("Defect Rate (%)")
            ax.set_title("Daily Defect Rate (Last 30 Days)")
            
            # Format x-axis
            ax.xaxis.set_major_formatter(DateFormatter("%m-%d"))
            if len(dates) > 10:
                # Show every nth label to avoid crowding
                n = len(dates) // 10 + 1
                for i, label in enumerate(ax.get_xticklabels()):
                    if i % n != 0:
                        label.set_visible(False)
            
            ax.grid(True, alpha=0.3)
            self.defect_rate_canvas.fig.tight_layout()
            self.defect_rate_canvas.draw()
            
        except Exception as e:
            print(f"Error updating defect rate chart: {e}")
    
    def update_defect_history(self):
        """Update the defect history tab."""
        try:
            # Get selected time range
            time_range = self.time_range_combo.currentText()
            
            # Calculate date range
            end_date = datetime.datetime.now().date()
            
            if time_range == "Last 24 Hours":
                start_date = end_date - datetime.timedelta(days=1)
            elif time_range == "Last 7 Days":
                start_date = end_date - datetime.timedelta(days=7)
            elif time_range == "Last 30 Days":
                start_date = end_date - datetime.timedelta(days=30)
            else:  # All Time or custom
                # Use date range from UI controls
                start_date = self.start_date.date().toPyDate()
                end_date = self.end_date.date().toPyDate()
            
            # Format dates for database query
            start_date_str = start_date.isoformat()
            end_date_str = end_date.isoformat()
            
            # Get defect history data
            defect_data = self.db_manager.get_detections_by_date_range(start_date_str, end_date_str)
            
            # Update chart and table
            self.plot_defect_history(defect_data, start_date, end_date)
            self.update_history_table(defect_data)
            
        except Exception as e:
            print(f"Error updating defect history: {e}")
    
    def plot_defect_history(self, defect_data, start_date, end_date):
        """Plot defect history chart."""
        try:
            # Clear chart
            ax = self.defect_history_canvas.axes
            ax.clear()
            
            if not defect_data:
                ax.text(0.5, 0.5, "No data available for selected time range", ha='center', va='center')
                self.defect_history_canvas.draw()
                return
            
            # Group data by date
            date_groups = {}
            for record in defect_data:
                # Parse date from timestamp
                timestamp = datetime.datetime.fromisoformat(record["timestamp"])
                date = timestamp.date()
                
                if date not in date_groups:
                    date_groups[date] = {
                        "total": 0,
                        "defects": 0,
                        "error_sum": 0
                    }
                
                date_groups[date]["total"] += 1
                date_groups[date]["error_sum"] += record.get("global_error", 0)
                
                if record.get("is_defect", 0) == 1:
                    date_groups[date]["defects"] += 1
            
            # Prepare data for plotting
            dates = sorted(date_groups.keys())
            totals = [date_groups[date]["total"] for date in dates]
            defects = [date_groups[date]["defects"] for date in dates]
            
            # Calculate defect rates
            rates = []
            for date in dates:
                if date_groups[date]["total"] > 0:
                    rate = (date_groups[date]["defects"] / date_groups[date]["total"]) * 100
                else:
                    rate = 0
                rates.append(rate)
            
            # Create figure with two y-axes
            ax2 = ax.twinx()
            
            # Plot bars for totals and defects
            bar_width = 0.35
            ax.bar([d - bar_width/2 for d in range(len(dates))], totals, bar_width, 
                  color='blue', alpha=0.5, label='Total')
            ax.bar([d + bar_width/2 for d in range(len(dates))], defects, bar_width, 
                  color='red', alpha=0.7, label='Defects')
            
            # Plot line for defect rate
            ax2.plot(range(len(dates)), rates, color='black', marker='o', linestyle='-', 
                    linewidth=2, label='Defect Rate (%)')
            
            # Set labels and title
            ax.set_xlabel("Date")
            ax.set_ylabel("Count")
            ax2.set_ylabel("Defect Rate (%)")
            ax.set_title(f"Defect History ({start_date} to {end_date})")
            
            # Set x-tick labels to dates
            ax.set_xticks(range(len(dates)))
            ax.set_xticklabels([d.strftime("%m-%d") for d in dates], rotation=45)
            
            # Add legends
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Adjust layout
            self.defect_history_canvas.fig.tight_layout()
            self.defect_history_canvas.draw()
            
        except Exception as e:
            print(f"Error plotting defect history: {e}")
    
    def update_history_table(self, defect_data):
        """Update history table with daily summary."""
        try:
            # Group data by date
            date_groups = {}
            for record in defect_data:
                # Parse date from timestamp
                timestamp = datetime.datetime.fromisoformat(record["timestamp"])
                date = timestamp.date()
                
                if date not in date_groups:
                    date_groups[date] = {
                        "total": 0,
                        "defects": 0,
                        "error_sum": 0
                    }
                
                date_groups[date]["total"] += 1
                date_groups[date]["error_sum"] += record.get("global_error", 0)
                
                if record.get("is_defect", 0) == 1:
                    date_groups[date]["defects"] += 1
            
            # Calculate averages and rates
            for date in date_groups:
                if date_groups[date]["total"] > 0:
                    date_groups[date]["avg_error"] = date_groups[date]["error_sum"] / date_groups[date]["total"]
                    date_groups[date]["defect_rate"] = (date_groups[date]["defects"] / date_groups[date]["total"]) * 100
                else:
                    date_groups[date]["avg_error"] = 0
                    date_groups[date]["defect_rate"] = 0
            
            # Sort dates
            dates = sorted(date_groups.keys(), reverse=True)
            
            # Update table
            self.history_table.setRowCount(len(dates))
            
            for i, date in enumerate(dates):
                data = date_groups[date]
                
                self.history_table.setItem(i, 0, QTableWidgetItem(date.strftime("%Y-%m-%d")))
                self.history_table.setItem(i, 1, QTableWidgetItem(str(data["total"])))
                self.history_table.setItem(i, 2, QTableWidgetItem(str(data["defects"])))
                self.history_table.setItem(i, 3, QTableWidgetItem(f"{data['defect_rate']:.2f}%"))
                self.history_table.setItem(i, 4, QTableWidgetItem(f"{data['avg_error']:.4f}"))
                
        except Exception as e:
            print(f"Error updating history table: {e}")
    
    def export_defect_history(self):
        """Export defect history data to CSV."""
        try:
            # Ask user for save location
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Defect History", "", "CSV Files (*.csv)"
            )
            
            if not file_path:
                return
            
            # Get selected time range
            time_range = self.time_range_combo.currentText()
            
            # Calculate date range
            end_date = datetime.datetime.now().date()
            
            if time_range == "Last 24 Hours":
                start_date = end_date - datetime.timedelta(days=1)
            elif time_range == "Last 7 Days":
                start_date = end_date - datetime.timedelta(days=7)
            elif time_range == "Last 30 Days":
                start_date = end_date - datetime.timedelta(days=30)
            else:  # All Time or custom
                # Use date range from UI controls
                start_date = self.start_date.date().toPyDate()
                end_date = self.end_date.date().toPyDate()
            
            # Format dates for database query
            start_date_str = start_date.isoformat()
            end_date_str = end_date.isoformat()
            
            # Export data to CSV
            self.db_manager.export_to_csv(file_path, start_date_str, end_date_str)
            
            self.statusBar().showMessage(f"Data exported to {file_path}")
        except Exception as e:
            self.statusBar().showMessage(f"Error exporting data: {str(e)}")
    
    def update_system_monitor(self):
        """Update system monitor tab."""
        try:
            # Get system stats
            stats = self.system_monitor.collect_system_stats()
            
            # Update labels and progress bars
            if stats.get("cpu_usage") is not None:
                self.cpu_usage_label.setText(f"CPU Usage: {stats['cpu_usage']:.1f}%")
                self.cpu_progress.setValue(min(100, int(stats['cpu_usage'])))
            
            if stats.get("memory_usage"):
                memory_percent = stats["memory_usage"]["percent"]
                self.memory_usage_label.setText(f"Memory Usage: {memory_percent:.1f}%")
                self.memory_progress.setValue(min(100, int(memory_percent)))
            
            if stats.get("disk_usage"):
                disk_percent = stats["disk_usage"]["percent"]
                self.disk_usage_label.setText(f"Disk Usage: {disk_percent:.1f}%")
                self.disk_progress.setValue(min(100, int(disk_percent)))
            
            if stats.get("temperature") is not None:
                self.temperature_label.setText(f"Temperature: {stats['temperature']:.1f}°C")
            
            if stats.get("uptime"):
                days = stats["uptime"]["days"]
                self.uptime_label.setText(f"Uptime: {days:.1f} days")
            
            # Update system metrics chart
            self.update_system_metrics_chart()
            
            # Update alerts table
            self.update_alerts_table()
            
        except Exception as e:
            print(f"Error updating system monitor: {e}")
    
    def update_system_metrics_chart(self):
        """Update system metrics chart."""
        try:
            # Get recent system stats
            recent_stats = self.system_monitor.get_recent_stats(60)  # Last 60 data points
            
            if not recent_stats:
                return
            
            # Prepare data
            timestamps = []
            cpu_values = []
            memory_values = []
            disk_values = []
            
            for stat in recent_stats:
                # Parse timestamp
                timestamp = datetime.datetime.fromisoformat(stat["timestamp"])
                timestamps.append(timestamp)
                
                # Get values
                if stat.get("cpu_usage") is not None:
                    cpu_values.append(stat["cpu_usage"])
                else:
                    cpu_values.append(None)
                
                if stat.get("memory_usage") and "percent" in stat["memory_usage"]:
                    memory_values.append(stat["memory_usage"]["percent"])
                else:
                    memory_values.append(None)
                
                if stat.get("disk_usage") and "percent" in stat["disk_usage"]:
                    disk_values.append(stat["disk_usage"]["percent"])
                else:
                    disk_values.append(None)
            
            # Clear chart
            ax = self.system_metrics_canvas.axes
            ax.clear()
            
            # Plot data
            if cpu_values and any(v is not None for v in cpu_values):
                ax.plot(timestamps, cpu_values, 'r-', label='CPU Usage (%)')
            
            if memory_values and any(v is not None for v in memory_values):
                ax.plot(timestamps, memory_values, 'g-', label='Memory Usage (%)')
            
            if disk_values and any(v is not None for v in disk_values):
                ax.plot(timestamps, disk_values, 'b-', label='Disk Usage (%)')
            
            # Set labels and title
            ax.set_xlabel("Time")
            ax.set_ylabel("Usage (%)")
            ax.set_title("System Resource Usage Over Time")
            
            # Format x-axis
            ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
            
            # Add legend and grid
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # Adjust layout
            self.system_metrics_canvas.fig.tight_layout()
            self.system_metrics_canvas.draw()