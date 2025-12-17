#!/usr/bin/env python3
"""
Standalone Database Migration Script
Does NOT import from app.py to avoid circular dependencies
"""

import sys
import os
import logging
import json
import base64

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup (standalone, no app.py import)
from sqlalchemy import create_engine, text, inspect, Column, Integer, String, Text, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime

DATABASE_URL = "sqlite:///akademie_suite.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Import crypto (this is safe - doesn't run startup code)
sys.path.insert(0, '/var/www/transkript_app')
from crypto_utils import crypto, key_wrapper

# Define minimal models (just enough for migration)
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    salt = Column(String)
    encrypted_master_key = Column(Text)
    email = Column(String, unique=True, index=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class Transcription(Base):
    __tablename__ = "transcriptions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    provider = Column(String, nullable=False)
    model = Column(String)
    original_text = Column(Text, nullable=False)
    translated_text = Column(Text)
    language = Column(String)
    filename = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    title = Column(String)
    is_encrypted = Column(Boolean, default=False)
    encryption_metadata = Column(Text)

class ChatHistory(Base):
    __tablename__ = "chat_history"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    provider = Column(String, nullable=False)
    model = Column(String, nullable=False)
    messages = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    title = Column(String)
    is_encrypted = Column(Boolean, default=False)
    encryption_metadata = Column(Text)

class VisionResult(Base):
    __tablename__ = "vision_results"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    provider = Column(String, nullable=False)
    model = Column(String, nullable=False)
    prompt = Column(Text, nullable=False)
    result = Column(Text, nullable=False)
    image_path = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    is_encrypted = Column(Boolean, default=False)
    encryption_metadata = Column(Text)

class GeneratedImage(Base):
    __tablename__ = "generated_images"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    provider = Column(String, nullable=False)
    model = Column(String, nullable=False)
    prompt = Column(Text, nullable=False)
    image_path = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    is_encrypted = Column(Boolean, default=False)
    encryption_metadata = Column(Text)
    encrypted_path = Column(String)

# Helper functions
def check_column_exists(table_name, column_name):
    """Check if a column exists in a table"""
    inspector = inspect(engine)
    columns = [col['name'] for col in inspector.get_columns(table_name)]
    return column_name in columns

def add_user_key_columns():
    """Add salt and encrypted_master_key to users table"""
    with engine.begin() as conn:
        try:
            inspector = inspect(engine)
            cols = [c['name'] for c in inspector.get_columns("users")]
            
            if "salt" not in cols:
                logger.info("➕ Adding 'salt' column to users...")
                conn.execute(text("ALTER TABLE users ADD COLUMN salt TEXT"))
            else:
                logger.info("ℹ️ Column 'salt' already exists")
            
            if "encrypted_master_key" not in cols:
                logger.info("➕ Adding 'encrypted_master_key' column to users...")
                conn.execute(text("ALTER TABLE users ADD COLUMN encrypted_master_key TEXT"))
            else:
                logger.info("ℹ️ Column 'encrypted_master_key' already exists")
                
            logger.info("✅ User table schema updated")
        except Exception as e:
            logger.error(f"❌ Failed to update user schema: {e}")
            raise

def add_encryption_columns():
    """Add encryption columns to data tables"""
    tables_to_update = [
        "transcriptions",
        "chat_history", 
        "vision_results",
        "generated_images"
    ]
    
    with engine.begin() as conn:
        for table in tables_to_update:
            try:
                result = conn.execute(text(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'"))
                if not result.fetchone():
                    logger.warning(f"⚠️ Table {table} does not exist, skipping...")
                    continue
                
                if not check_column_exists(table, "is_encrypted"):
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN is_encrypted BOOLEAN DEFAULT 0"))
                    logger.info(f"✅ Added is_encrypted to {table}")
                else:
                    logger.info(f"ℹ️ Column is_encrypted already exists in {table}")
                
                if not check_column_exists(table, "encryption_metadata"):
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN encryption_metadata TEXT"))
                    logger.info(f"✅ Added encryption_metadata to {table}")
                else:
                    logger.info(f"ℹ️ Column encryption_metadata already exists in {table}")
                
                if table == "generated_images" and not check_column_exists(table, "encrypted_path"):
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN encrypted_path TEXT"))
                    logger.info(f"✅ Added encrypted_path to {table}")
                    
            except Exception as e:
                logger.error(f"❌ Error updating {table}: {e}")
                raise

def encrypt_existing_data():
    """Encrypt all plaintext data with global key"""
    db = SessionLocal()
    
    try:
        logger.info("🔐 Starting data encryption...")
        
        if not crypto.master_key:
            logger.error("❌ Crypto not initialized!")
            return False
        
        # Count
        trans_count = db.query(Transcription).filter(Transcription.is_encrypted == False).count()
        chat_count = db.query(ChatHistory).filter(ChatHistory.is_encrypted == False).count()
        vision_count = db.query(VisionResult).filter(VisionResult.is_encrypted == False).count()
        
        logger.info(f"Found: {trans_count} transcriptions, {chat_count} chats, {vision_count} vision results")
        
        # Encrypt transcriptions
        if trans_count > 0:
            logger.info(f"Encrypting {trans_count} transcriptions...")
            for i, t in enumerate(db.query(Transcription).filter(Transcription.is_encrypted == False).all(), 1):
                try:
                    if t.original_text and t.original_text.strip():
                        t.original_text = crypto.encrypt_text(t.original_text)
                    if t.translated_text and t.translated_text.strip():
                        t.translated_text = crypto.encrypt_text(t.translated_text)
                    
                    t.is_encrypted = True
                    t.encryption_metadata = json.dumps({"algorithm": "AES-256-GCM", "version": 1})
                    
                    if i % 10 == 0:
                        logger.info(f"  Progress: {i}/{trans_count}")
                        db.commit()
                except Exception as e:
                    logger.error(f"Failed: {e}")
                    continue
            
            db.commit()
            logger.info(f"✅ Encrypted {trans_count} transcriptions")
        
        # Encrypt chats
        if chat_count > 0:
            logger.info(f"Encrypting {chat_count} chats...")
            for i, c in enumerate(db.query(ChatHistory).filter(ChatHistory.is_encrypted == False).all(), 1):
                try:
                    if c.messages and c.messages.strip():
                        c.messages = crypto.encrypt_text(c.messages)
                    
                    c.is_encrypted = True
                    c.encryption_metadata = json.dumps({"algorithm": "AES-256-GCM", "version": 1})
                    
                    if i % 10 == 0:
                        logger.info(f"  Progress: {i}/{chat_count}")
                        db.commit()
                except Exception as e:
                    logger.error(f"Failed: {e}")
                    continue
            
            db.commit()
            logger.info(f"✅ Encrypted {chat_count} chats")
        
        # Encrypt vision
        if vision_count > 0:
            logger.info(f"Encrypting {vision_count} vision results...")
            for i, v in enumerate(db.query(VisionResult).filter(VisionResult.is_encrypted == False).all(), 1):
                try:
                    if v.result and v.result.strip():
                        v.result = crypto.encrypt_text(v.result)
                    if v.prompt and v.prompt.strip():
                        v.prompt = crypto.encrypt_text(v.prompt)
                    
                    v.is_encrypted = True
                    v.encryption_metadata = json.dumps({"algorithm": "AES-256-GCM", "version": 1})
                    
                    if i % 10 == 0:
                        logger.info(f"  Progress: {i}/{vision_count}")
                        db.commit()
                except Exception as e:
                    logger.error(f"Failed: {e}")
                    continue
            
            db.commit()
            logger.info(f"✅ Encrypted {vision_count} vision results")
        
        logger.info("✅ Data encryption complete!")
        return True
        
    except Exception as e:
        db.rollback()
        logger.exception(f"🔥 Encryption failed: {e}")
        return False
    finally:
        db.close()

def main():
    """Main migration workflow"""
    
    print("=" * 60)
    print("🔐 STANDALONE DATABASE MIGRATION")
    print("=" * 60)
    print()
    
    response = input("⚠️  This will modify your database. Continue? (type 'YES'): ")
    if response != 'YES':
        print("❌ Migration cancelled")
        return
    
    print()
    logger.info("Starting migration...")
    print()
    
    try:
        # Step 1: Add columns
        logger.info("Step 1/3: Adding encryption columns...")
        add_user_key_columns()
        add_encryption_columns()
        print()
        
        # Step 2: Encrypt data
        logger.info("Step 2/3: Encrypting existing data...")
        success = encrypt_existing_data()
        print()
        
        if not success:
            logger.error("❌ Migration failed!")
            return
        
        # Step 3: Done
        logger.info("Step 3/3: Complete!")
        print()
        print("=" * 60)
        print("✅ MIGRATION SUCCESSFUL!")
        print("=" * 60)
        print()
        print("Next steps:")
        print("1. Restart app: sudo systemctl restart transkript.service")
        print("2. Existing users will be migrated on next login")
        print("3. No password change required!")
        print()
        
    except Exception as e:
        logger.exception(f"🔥 Migration failed: {e}")

if __name__ == "__main__":
    main()