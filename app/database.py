import sqlite3
import os
from datetime import datetime

DB_PATH = "crop_diagnoses.db"

def init_db():
    """Initializes the local SQLite database for logging scan history."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS scan_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            disease_detected TEXT NOT NULL,
            confidence_score TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            field_acres REAL NOT NULL,
            risk_score TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def save_scan_record(disease: str, confidence: float, lat: float, lon: float, acres: float, risk: str):
    """Saves a new diagnostic record into SQLite."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO scan_history (timestamp, disease_detected, confidence_score, latitude, longitude, field_acres, risk_score)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (now_str, disease, f"{confidence}%", lat, lon, acres, risk))
    conn.commit()
    conn.close()

def fetch_recent_scans(limit: int = 5):
    """Retrieves the latest scan records."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT timestamp, disease_detected, confidence_score, field_acres, risk_score 
        FROM scan_history ORDER BY id DESC LIMIT ?
    ''', (limit,))
    rows = cursor.fetchall()
    conn.close()
    
    history = []
    for row in rows:
        history.append({
            "timestamp": row[0],
            "disease": row[1],
            "confidence": row[2],
            "acres": row[3],
            "risk": row[4]
        })
    return history

init_db()