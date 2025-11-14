"""
Storage module for traffic data

Handles SQLite database and CSV file storage
"""

import csv
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


logger = logging.getLogger(__name__)


class StorageManager:
    """Manage traffic data storage in SQLite and CSV"""

    def __init__(
        self,
        db_path: str = "./data/traffic.db",
        csv_dir: str = "./data/csv",
        retention_days: int = 90
    ):
        """
        Initialize storage manager

        Args:
            db_path: Path to SQLite database file
            csv_dir: Directory for CSV files
            retention_days: Number of days to keep data
        """
        self.db_path = Path(db_path)
        self.csv_dir = Path(csv_dir)
        self.retention_days = retention_days

        # Create directories
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_dir.mkdir(parents=True, exist_ok=True)

        self.conn = None
        self._init_database()

        logger.info(f"StorageManager initialized:")
        logger.info(f"  Database: {self.db_path}")
        logger.info(f"  CSV directory: {self.csv_dir}")
        logger.info(f"  Retention: {retention_days} days")

    def _init_database(self):
        """Initialize SQLite database and create tables"""
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        cursor = self.conn.cursor()

        # Create detections table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                track_id INTEGER NOT NULL,
                object_type VARCHAR(20) NOT NULL,
                speed_kmh REAL NOT NULL,
                direction VARCHAR(10),
                confidence REAL,
                distance_meters REAL,
                time_seconds REAL,
                trajectory_length INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON detections(timestamp)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_object_type
            ON detections(object_type)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_track_id
            ON detections(track_id)
        """)

        # Create hourly_stats table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hourly_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                hour INTEGER NOT NULL,
                object_type VARCHAR(20) NOT NULL,
                count INTEGER NOT NULL,
                avg_speed REAL,
                max_speed REAL,
                min_speed REAL,
                speeding_count INTEGER,
                UNIQUE(date, hour, object_type)
            )
        """)

        # Create daily_stats table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL UNIQUE,
                total_vehicles INTEGER,
                avg_speed REAL,
                max_speed REAL,
                min_speed REAL,
                speeding_count INTEGER,
                speeding_rate REAL,
                peak_hour INTEGER
            )
        """)

        self.conn.commit()
        logger.info("✓ Database initialized")

    def save_detection(self, speed_estimate) -> int:
        """
        Save speed estimate to database

        Args:
            speed_estimate: SpeedEstimate object

        Returns:
            ID of inserted row
        """
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO detections (
                timestamp, track_id, object_type, speed_kmh,
                direction, confidence, distance_meters,
                time_seconds, trajectory_length
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            speed_estimate.timestamp,
            speed_estimate.track_id,
            speed_estimate.object_type,
            speed_estimate.speed_kmh,
            speed_estimate.direction,
            speed_estimate.confidence,
            speed_estimate.distance_meters,
            speed_estimate.time_seconds,
            speed_estimate.trajectory_length
        ))

        self.conn.commit()
        return cursor.lastrowid

    def save_csv(self, speed_estimate, speed_limit_kmh: float = 30.0):
        """
        Append speed estimate to daily CSV file

        Args:
            speed_estimate: SpeedEstimate object
            speed_limit_kmh: Speed limit for speeding flag
        """
        # Get date for filename
        date_str = speed_estimate.timestamp.strftime('%Y-%m-%d')
        csv_file = self.csv_dir / f"traffic_{date_str}.csv"

        # Check if file exists (to write header)
        file_exists = csv_file.exists()

        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)

            # Write header if new file
            if not file_exists:
                writer.writerow([
                    'timestamp',
                    'track_id',
                    'object_type',
                    'speed_kmh',
                    'direction',
                    'confidence',
                    'distance_meters',
                    'time_seconds',
                    'trajectory_length',
                    'speeding'
                ])

            # Write data
            writer.writerow([
                speed_estimate.timestamp.isoformat(),
                speed_estimate.track_id,
                speed_estimate.object_type,
                f"{speed_estimate.speed_kmh:.2f}",
                speed_estimate.direction,
                f"{speed_estimate.confidence:.2f}",
                f"{speed_estimate.distance_meters:.2f}",
                f"{speed_estimate.time_seconds:.2f}",
                speed_estimate.trajectory_length,
                speed_estimate.is_speeding(speed_limit_kmh)
            ])

    def get_detections(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        object_type: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get detections from database

        Args:
            start_date: Start date filter
            end_date: End date filter
            object_type: Object type filter

        Returns:
            DataFrame with detections
        """
        query = "SELECT * FROM detections WHERE 1=1"
        params = []

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)

        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)

        if object_type:
            query += " AND object_type = ?"
            params.append(object_type)

        query += " ORDER BY timestamp"

        return pd.read_sql_query(query, self.conn, params=params)

    def calculate_hourly_stats(self, date: datetime, speed_limit_kmh: float = 30.0):
        """
        Calculate and save hourly statistics for a date

        Args:
            date: Date to calculate stats for
            speed_limit_kmh: Speed limit for speeding calculation
        """
        cursor = self.conn.cursor()

        date_str = date.strftime('%Y-%m-%d')

        # Calculate stats for each hour and object type
        cursor.execute("""
            SELECT
                DATE(timestamp) as date,
                CAST(strftime('%H', timestamp) AS INTEGER) as hour,
                object_type,
                COUNT(*) as count,
                AVG(speed_kmh) as avg_speed,
                MAX(speed_kmh) as max_speed,
                MIN(speed_kmh) as min_speed,
                SUM(CASE WHEN speed_kmh > ? THEN 1 ELSE 0 END) as speeding_count
            FROM detections
            WHERE DATE(timestamp) = ?
            GROUP BY date, hour, object_type
        """, (speed_limit_kmh, date_str))

        rows = cursor.fetchall()

        for row in rows:
            cursor.execute("""
                INSERT OR REPLACE INTO hourly_stats (
                    date, hour, object_type, count,
                    avg_speed, max_speed, min_speed, speeding_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row[0], row[1], row[2], row[3],
                row[4], row[5], row[6], row[7]
            ))

        self.conn.commit()
        logger.info(f"Calculated hourly stats for {date_str}")

    def calculate_daily_stats(self, date: datetime, speed_limit_kmh: float = 30.0):
        """
        Calculate and save daily statistics

        Args:
            date: Date to calculate stats for
            speed_limit_kmh: Speed limit for speeding calculation
        """
        cursor = self.conn.cursor()

        date_str = date.strftime('%Y-%m-%d')

        # Calculate daily stats
        cursor.execute("""
            SELECT
                COUNT(*) as total_vehicles,
                AVG(speed_kmh) as avg_speed,
                MAX(speed_kmh) as max_speed,
                MIN(speed_kmh) as min_speed,
                SUM(CASE WHEN speed_kmh > ? THEN 1 ELSE 0 END) as speeding_count
            FROM detections
            WHERE DATE(timestamp) = ?
        """, (speed_limit_kmh, date_str))

        row = cursor.fetchone()

        if row and row[0] > 0:  # If there are detections
            total = row[0]
            speeding_count = row[4]
            speeding_rate = (speeding_count / total) * 100 if total > 0 else 0

            # Get peak hour
            cursor.execute("""
                SELECT hour, SUM(count) as total
                FROM hourly_stats
                WHERE date = ?
                GROUP BY hour
                ORDER BY total DESC
                LIMIT 1
            """, (date_str,))

            peak_row = cursor.fetchone()
            peak_hour = peak_row[0] if peak_row else None

            # Insert daily stats
            cursor.execute("""
                INSERT OR REPLACE INTO daily_stats (
                    date, total_vehicles, avg_speed, max_speed, min_speed,
                    speeding_count, speeding_rate, peak_hour
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                date_str, row[0], row[1], row[2], row[3],
                row[4], speeding_rate, peak_hour
            ))

            self.conn.commit()
            logger.info(f"Calculated daily stats for {date_str}")

    def get_daily_summary(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict:
        """
        Get summary statistics for date range

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            Dictionary with summary statistics
        """
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT
                SUM(total_vehicles) as total,
                AVG(avg_speed) as avg_speed,
                MAX(max_speed) as max_speed,
                SUM(speeding_count) as speeding_count,
                AVG(speeding_rate) as avg_speeding_rate
            FROM daily_stats
            WHERE date >= ? AND date <= ?
        """, (
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        ))

        row = cursor.fetchone()

        if row:
            return {
                'total_vehicles': row[0] or 0,
                'avg_speed': row[1] or 0,
                'max_speed': row[2] or 0,
                'speeding_count': row[3] or 0,
                'avg_speeding_rate': row[4] or 0
            }

        return {}

    def cleanup_old_data(self):
        """Remove data older than retention period"""
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        cutoff_str = cutoff_date.strftime('%Y-%m-%d')

        cursor = self.conn.cursor()

        # Delete old detections
        cursor.execute("""
            DELETE FROM detections WHERE DATE(timestamp) < ?
        """, (cutoff_str,))

        deleted_detections = cursor.rowcount

        # Delete old stats
        cursor.execute("""
            DELETE FROM hourly_stats WHERE date < ?
        """, (cutoff_str,))

        cursor.execute("""
            DELETE FROM daily_stats WHERE date < ?
        """, (cutoff_str,))

        self.conn.commit()

        if deleted_detections > 0:
            logger.info(f"Cleaned up {deleted_detections} old detection records")

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
