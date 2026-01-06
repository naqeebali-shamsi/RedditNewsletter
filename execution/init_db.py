#!/usr/bin/env python3
"""
Initialize the Reddit Content Database.
Run this script once to create the SQLite database with the required schema.
"""

import sqlite3
import os
from pathlib import Path

# Database path (at project root)
DB_PATH = Path(__file__).parent.parent / "reddit_content.db"
SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def init_database():
    """Initialize the database with the schema."""
    
    # Check if schema file exists
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Schema file not found: {SCHEMA_PATH}")
    
    # Read schema
    with open(SCHEMA_PATH, 'r') as f:
        schema_sql = f.read()
    
    # Create/connect to database
    print(f"Initializing database at: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Execute schema
        cursor.executescript(schema_sql)
        conn.commit()
        
        # Verify tables were created
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        print(f"✓ Database initialized successfully!")
        print(f"✓ Created tables: {', '.join([t[0] for t in tables])}")
        
        # Print table counts
        for table in ['posts', 'evaluations', 'drafts']:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"  - {table}: {count} rows")
        
    except sqlite3.Error as e:
        print(f"✗ Database error: {e}")
        conn.rollback()
        raise
    
    finally:
        conn.close()


if __name__ == "__main__":
    init_database()
