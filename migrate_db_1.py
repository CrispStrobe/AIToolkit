# migrate_database.py - RUN THIS ONCE
import os
import sqlite3

DB_PATH = "akademie_suite.db"

def migrate_database():
    """Add UserSettings table if it doesn't exist"""
    
    if not os.path.exists(DB_PATH):
        print("Database doesn't exist yet. Will be created on first run.")
        return
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Check if user_settings table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='user_settings'
        """)
        
        if cursor.fetchone():
            print("✅ user_settings table already exists")
        else:
            print("Creating user_settings table...")
            
            cursor.execute("""
                CREATE TABLE user_settings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL UNIQUE,
                    auto_chunk_enabled INTEGER DEFAULT 1,
                    chunk_size INTEGER DEFAULT 4000,
                    chunk_overlap INTEGER DEFAULT 200,
                    auto_truncate_history INTEGER DEFAULT 1,
                    show_truncation_warning INTEGER DEFAULT 1,
                    show_token_counts INTEGER DEFAULT 0,
                    compact_mode INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            """)
            
            conn.commit()
            print("✅ user_settings table created successfully")
        
    except Exception as e:
        print(f"🔥 Migration error: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    print("=" * 50)
    print("DATABASE MIGRATION")
    print("=" * 50)
    migrate_database()
    print("\nComplete!")
