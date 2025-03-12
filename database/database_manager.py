#!/usr/bin/env python3
"""
Database Manager for Defect Detection System
Provides a reliable storage system for detection results with querying capabilities.
"""

import os
import sqlite3
import csv
import datetime
import threading
from contextlib import contextmanager
from typing import Dict, List, Tuple, Any, Optional, Union

# Import centralized configuration
from config import config

# Set up logger
logger = config.setup_logger("database_manager")

class DatabaseManager:
    """
    Manages SQLite database operations for storing detection results.
    Provides thread-safe access and migration from CSV files.
    """
    
    def __init__(self, db_path=None):
        """
        Initialize the database manager.
        
        Args:
            db_path: Path to SQLite database file (defaults to config)
        """
        self.db_path = db_path or os.path.join(config.BASE_DIR, "detection_results.db")
        self.lock = threading.RLock()
        
        # Ensure database directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema if it doesn't exist"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Create detection results table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS detection_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                filename TEXT NOT NULL,
                global_threshold REAL NOT NULL,
                global_error REAL NOT NULL,
                patch_threshold REAL,
                patch_value REAL,
                patch_defect_ratio_threshold REAL,
                patch_defect_ratio_value REAL,
                is_defect INTEGER NOT NULL,
                detection_method TEXT,
                image_path TEXT,
                heatmap_path TEXT
            )
            ''')
            
            # Create index on timestamp for faster queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON detection_results(timestamp)')
            
            # Create stats table for aggregated data
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS detection_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                total_inspections INTEGER NOT NULL,
                defects_found INTEGER NOT NULL,
                average_global_error REAL NOT NULL,
                max_global_error REAL NOT NULL,
                min_global_error REAL NOT NULL,
                UNIQUE(date)
            )
            ''')
            
            conn.commit()
    
    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections.
        Ensures proper handling of connections and transactions.
        """
        with self.lock:  # Ensure thread safety
            connection = None
            try:
                connection = sqlite3.connect(self.db_path)
                # Enable foreign keys
                connection.execute("PRAGMA foreign_keys = ON")
                # Return rows as dictionaries
                connection.row_factory = sqlite3.Row
                yield connection
                connection.commit()
            except sqlite3.Error as e:
                if connection:
                    connection.rollback()
                logger.error(f"Database error: {e}")
                raise
            finally:
                if connection:
                    connection.close()
    
    def store_detection_result(self, result_data: Dict[str, Any]) -> int:
        """
        Store a detection result in the database.
        
        Args:
            result_data: Dictionary containing detection result data
        
        Returns:
            ID of the inserted record
        """
        # Ensure timestamp is present
        if 'timestamp' not in result_data:
            result_data['timestamp'] = datetime.datetime.now().isoformat()
        
        # Convert boolean to integer for SQLite
        if 'is_defect' in result_data and isinstance(result_data['is_defect'], bool):
            result_data['is_defect'] = 1 if result_data['is_defect'] else 0
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Prepare query dynamically based on available fields
            fields = []
            placeholders = []
            values = []
            
            for key, value in result_data.items():
                fields.append(key)
                placeholders.append('?')
                values.append(value)
            
            query = f'''
            INSERT INTO detection_results ({','.join(fields)})
            VALUES ({','.join(placeholders)})
            '''
            
            cursor.execute(query, values)
            record_id = cursor.lastrowid
            
            # Update daily stats
            self._update_daily_stats(conn)
            
            return record_id
    
    def _update_daily_stats(self, conn) -> None:
        """
        Update daily statistics based on detection results.
        
        Args:
            conn: Active database connection
        """
        today = datetime.date.today().isoformat()
        cursor = conn.cursor()
        
        # Get today's stats
        cursor.execute('''
        SELECT 
            COUNT(*) as total_inspections,
            SUM(CASE WHEN is_defect = 1 THEN 1 ELSE 0 END) as defects_found,
            AVG(global_error) as average_global_error,
            MAX(global_error) as max_global_error,
            MIN(global_error) as min_global_error
        FROM detection_results
        WHERE date(timestamp) = date(?)
        ''', (today,))
        
        result = cursor.fetchone()
        
        if result and result['total_inspections'] > 0:
            # Insert or update stats
            cursor.execute('''
            INSERT OR REPLACE INTO detection_stats 
            (date, total_inspections, defects_found, average_global_error, max_global_error, min_global_error)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                today,
                result['total_inspections'],
                result['defects_found'],
                result['average_global_error'],
                result['max_global_error'],
                result['min_global_error']
            ))
    
    def get_recent_detections(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent detection results.
        
        Args:
            limit: Maximum number of results to return
        
        Returns:
            List of detection result dictionaries
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT * FROM detection_results
            ORDER BY timestamp DESC
            LIMIT ?
            ''', (limit,))
            
            # Convert row objects to dictionaries
            return [dict(row) for row in cursor.fetchall()]
    
    def get_detection_by_id(self, detection_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a detection result by ID.
        
        Args:
            detection_id: ID of the detection result
        
        Returns:
            Detection result dictionary or None if not found
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT * FROM detection_results
            WHERE id = ?
            ''', (detection_id,))
            
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_detections_by_date_range(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """
        Get detection results within a date range.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        
        Returns:
            List of detection result dictionaries
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT * FROM detection_results
            WHERE date(timestamp) BETWEEN date(?) AND date(?)
            ORDER BY timestamp DESC
            ''', (start_date, end_date))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_defect_rate_by_date(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get daily defect rates for the specified number of days.
        
        Args:
            days: Number of days to include
        
        Returns:
            List of daily stats dictionaries
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT 
                date,
                total_inspections,
                defects_found,
                CAST(defects_found AS REAL) / total_inspections * 100 as defect_rate,
                average_global_error
            FROM detection_stats
            WHERE date >= date('now', ?)
            ORDER BY date DESC
            ''', (f'-{days} days',))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def migrate_from_csv(self, csv_path: str) -> int:
        """
        Migrate detection results from CSV to the database.
        
        Args:
            csv_path: Path to the CSV file
        
        Returns:
            Number of records migrated
        """
        if not os.path.exists(csv_path):
            logger.warning(f"CSV file not found: {csv_path}")
            return 0
        
        migrated_count = 0
        
        try:
            with open(csv_path, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    
                    for row in reader:
                        # Convert CSV fields to match database schema
                        result_data = {
                            'timestamp': datetime.datetime.now().isoformat(),  # Use current time as fallback
                            'filename': row.get('filename', ''),
                            'global_threshold': float(row.get('global_threshold', 0)),
                            'global_error': float(row.get('global_error', 0)),
                            'is_defect': 1 if row.get('is_defect', '').lower() == 'yes' else 0,
                        }
                        
                        # Add optional fields if present
                        if 'patch_threshold' in row and row['patch_threshold']:
                            result_data['patch_threshold'] = float(row['patch_threshold'])
                        
                        if 'mean_patch_error' in row and row['mean_patch_error']:
                            result_data['patch_value'] = float(row['mean_patch_error'])
                        
                        if 'patch_defect_ratio_threshold' in row and row['patch_defect_ratio_threshold']:
                            result_data['patch_defect_ratio_threshold'] = float(row['patch_defect_ratio_threshold'])
                        
                        if 'patch_defect_ratio_value' in row and row['patch_defect_ratio_value']:
                            result_data['patch_defect_ratio_value'] = float(row['patch_defect_ratio_value'])
                        
                        if 'detection_method' in row:
                            result_data['detection_method'] = row['detection_method']
                        
                        # Insert the record
                        fields = []
                        placeholders = []
                        values = []
                        
                        for key, value in result_data.items():
                            fields.append(key)
                            placeholders.append('?')
                            values.append(value)
                        
                        query = f'''
                        INSERT INTO detection_results ({','.join(fields)})
                        VALUES ({','.join(placeholders)})
                        '''
                        
                        cursor.execute(query, values)
                        migrated_count += 1
                    
                    # Update stats
                    self._update_daily_stats(conn)
            
            logger.info(f"Successfully migrated {migrated_count} records from CSV: {csv_path}")
            return migrated_count
        
        except Exception as e:
            logger.error(f"Error migrating from CSV: {e}")
            return 0
    
    def export_to_csv(self, csv_path: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> int:
        """
        Export detection results to CSV.
        
        Args:
            csv_path: Path to save the CSV file
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)
        
        Returns:
            Number of records exported
        """
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Build query based on date range
                query = "SELECT * FROM detection_results"
                params = []
                
                if start_date or end_date:
                    query += " WHERE "
                    if start_date:
                        query += "date(timestamp) >= date(?)"
                        params.append(start_date)
                        if end_date:
                            query += " AND "
                    
                    if end_date:
                        query += "date(timestamp) <= date(?)"
                        params.append(end_date)
                
                query += " ORDER BY timestamp DESC"
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                if not rows:
                    logger.warning("No data to export")
                    return 0
                
                # Get column names
                columns = [column[0] for column in cursor.description]
                
                # Write to CSV
                with open(csv_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(columns)  # Write header
                    
                    for row in rows:
                        writer.writerow(row)
                
                logger.info(f"Successfully exported {len(rows)} records to CSV: {csv_path}")
                return len(rows)
        
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return 0
    
    def purge_old_data(self, days_to_keep: int = 90) -> int:
        """
        Purge detection results older than the specified number of days.
        
        Args:
            days_to_keep: Number of days of data to keep
        
        Returns:
            Number of records purged
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Backup data to CSV before purging
            cutoff_date = (datetime.datetime.now() - datetime.datetime.timedelta(days=days_to_keep)).date().isoformat()
            backup_path = os.path.join(config.PATHS["BACKUP_CSV_DIR"], f"purged_data_{datetime.datetime.now().strftime('%Y%m%d')}.csv")
            
            # Get count of records to be purged
            cursor.execute('''
            SELECT COUNT(*) FROM detection_results
            WHERE date(timestamp) < date(?)
            ''', (cutoff_date,))
            
            purge_count = cursor.fetchone()[0]
            
            if purge_count > 0:
                # Export data to be purged
                self.export_to_csv(backup_path, None, cutoff_date)
                
                # Delete old records
                cursor.execute('''
                DELETE FROM detection_results
                WHERE date(timestamp) < date(?)
                ''', (cutoff_date,))
                
                # Also purge old stats
                cursor.execute('''
                DELETE FROM detection_stats
                WHERE date < date(?)
                ''', (cutoff_date,))
                
                logger.info(f"Purged {purge_count} old records (backed up to {backup_path})")
            else:
                logger.info("No old records to purge")
            
            return purge_count

# For testing
if __name__ == "__main__":
    print("Testing Database Manager")
    
    # Create an instance
    db_manager = DatabaseManager("/tmp/defect_test.db")
    
    # Test storing a result
    print("\nTesting result storage:")
    result_id = db_manager.store_detection_result({
        "filename": "test_image.jpg",
        "global_threshold": 0.85,
        "global_error": 0.92,
        "patch_threshold": 0.7,
        "patch_value": 0.73,
        "patch_defect_ratio_threshold": 0.45,
        "patch_defect_ratio_value": 0.51,
        "is_defect": True,
        "detection_method": "Global",
        "image_path": "/tmp/test_image.jpg"
    })
    print(f"Stored result with ID: {result_id}")
    
    # Test retrieval
    print("\nTesting result retrieval:")
    result = db_manager.get_detection_by_id(result_id)
    print(f"Retrieved result: {result}")
    
    # Test CSV migration
    if os.path.exists(config.PATHS["CSV_FILENAME"]):
        print("\nTesting CSV migration:")
        migrated = db_manager.migrate_from_csv(config.PATHS["CSV_FILENAME"])
        print(f"Migrated {migrated} records from CSV")
    
    # Test fetching recent results
    print("\nTesting recent results:")
    recent = db_manager.get_recent_detections(5)
    for item in recent:
        print(f"  - {item['timestamp']}: {'Defect' if item['is_defect'] else 'Normal'} (Error: {item['global_error']})")
    
    print("\nDatabase Manager tests completed.")