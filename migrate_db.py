#!/usr/bin/env python3
"""
Database Migration Script - Run ONCE to add encryption to existing database
"""

import sys
import os

# Add app directory to path
sys.path.insert(0, '/var/www/transkript_app')

from sqlalchemy import text, inspect
from app import User, engine, SessionLocal, Base, Transcription, ChatHistory, VisionResult, GeneratedImage
from crypto_utils import crypto, key_wrapper
import logging
import json
import base64  

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_column_exists(table_name, column_name):
    """Check if a column exists in a table"""
    inspector = inspect(engine)
    columns = [col['name'] for col in inspector.get_columns(table_name)]
    return column_name in columns

def add_encryption_columns():
    """Add is_encrypted and encryption_metadata columns to existing tables"""
    
    tables_to_update = [
        "transcriptions",
        "chat_history", 
        "vision_results",
        "generated_images"
    ]
    
    with engine.begin() as conn:
        for table in tables_to_update:
            try:
                # Check if table exists
                result = conn.execute(text(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'"))
                if not result.fetchone():
                    logger.warning(f"⚠️ Table {table} does not exist, skipping...")
                    continue
                
                # Check and add is_encrypted
                if not check_column_exists(table, "is_encrypted"):
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN is_encrypted BOOLEAN DEFAULT 0"))
                    logger.info(f"✅ Added is_encrypted to {table}")
                else:
                    logger.info(f"ℹ️ Column is_encrypted already exists in {table}")
                
                # Check and add encryption_metadata
                if not check_column_exists(table, "encryption_metadata"):
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN encryption_metadata TEXT"))
                    logger.info(f"✅ Added encryption_metadata to {table}")
                else:
                    logger.info(f"ℹ️ Column encryption_metadata already exists in {table}")
                
                # Add encrypted_path for generated_images (ADDED THIS)
                if table == "generated_images" and not check_column_exists(table, "encrypted_path"):
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN encrypted_path TEXT"))
                    logger.info(f"✅ Added encrypted_path to {table}")
                    
            except Exception as e:
                logger.error(f"❌ Error updating {table}: {e}")
                raise

def create_new_tables():
    """Create new tables that don't exist yet"""
    try:
        # This creates tables like FileUploadMetadata, DSFARecord, UserConfirmation, UserSettings, etc.
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Created new tables (if any)")
    except Exception as e:
        logger.error(f"❌ Error creating new tables: {e}")
        raise

def encrypt_existing_data():
    """Encrypt all existing plaintext data"""
    db = SessionLocal()
    
    try:
        logger.info("🔐 Starting data encryption...")
        
        # Check if crypto is initialized
        if not crypto.master_key:
            logger.error("❌ Crypto not initialized! Aborting.")
            return False
        
        # Count items to encrypt
        trans_count = db.query(Transcription).filter(Transcription.is_encrypted == False).count()
        chat_count = db.query(ChatHistory).filter(ChatHistory.is_encrypted == False).count()
        vision_count = db.query(VisionResult).filter(VisionResult.is_encrypted == False).count()
        
        logger.info(f"Found: {trans_count} transcriptions, {chat_count} chats, {vision_count} vision results to encrypt")
        
        # 1. Encrypt Transcriptions
        if trans_count > 0:
            logger.info(f"Encrypting {trans_count} transcriptions...")
            trans_list = db.query(Transcription).filter(Transcription.is_encrypted == False).all()
            
            for i, t in enumerate(trans_list, 1):
                try:
                    if t.original_text and t.original_text.strip():
                        t.original_text = crypto.encrypt_text(t.original_text)
                    if t.translated_text and t.translated_text.strip():
                        t.translated_text = crypto.encrypt_text(t.translated_text)
                    
                    t.is_encrypted = True
                    t.encryption_metadata = json.dumps({"algorithm": "AES-256-GCM", "version": 1})
                    
                    if i % 10 == 0:
                        logger.info(f"  Progress: {i}/{trans_count}")
                        db.commit()  # Commit in batches
                        
                except Exception as e:
                    logger.error(f"Failed to encrypt transcription {t.id}: {e}")
                    continue
            
            db.commit()
            logger.info(f"✅ Encrypted {trans_count} transcriptions")
        
        # 2. Encrypt Chat History
        if chat_count > 0:
            logger.info(f"Encrypting {chat_count} chats...")
            chat_list = db.query(ChatHistory).filter(ChatHistory.is_encrypted == False).all()
            
            for i, c in enumerate(chat_list, 1):
                try:
                    if c.messages and c.messages.strip():
                        c.messages = crypto.encrypt_text(c.messages)
                    
                    c.is_encrypted = True
                    c.encryption_metadata = json.dumps({"algorithm": "AES-256-GCM", "version": 1})
                    
                    if i % 10 == 0:
                        logger.info(f"  Progress: {i}/{chat_count}")
                        db.commit()
                        
                except Exception as e:
                    logger.error(f"Failed to encrypt chat {c.id}: {e}")
                    continue
            
            db.commit()
            logger.info(f"✅ Encrypted {chat_count} chats")
        
        # 3. Encrypt Vision Results
        if vision_count > 0:
            logger.info(f"Encrypting {vision_count} vision results...")
            vision_list = db.query(VisionResult).filter(VisionResult.is_encrypted == False).all()
            
            for i, v in enumerate(vision_list, 1):
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
                    logger.error(f"Failed to encrypt vision {v.id}: {e}")
                    continue
            
            db.commit()
            logger.info(f"✅ Encrypted {vision_count} vision results")
        
        # 4. Note about images (skipped - too large)
        img_count = db.query(GeneratedImage).filter(GeneratedImage.is_encrypted == False).count()
        if img_count > 0:
            logger.info(f"ℹ️ Skipping {img_count} generated images (file encryption on demand)")
        
        logger.info("✅ Data encryption complete!")
        return True
        
    except Exception as e:
        db.rollback()
        logger.exception(f"🔥 Encryption failed: {e}")
        return False
    finally:
        db.close()

def verify_encryption():
    """Verify that data is actually encrypted"""
    db = SessionLocal()
    try:
        # Check a sample transcription
        trans = db.query(Transcription).filter(Transcription.is_encrypted == True).first()
        if trans:
            # Encrypted data should be base64 (only contains A-Za-z0-9+/=)
            import re
            if trans.original_text:
                # Check if it's valid base64
                is_base64 = bool(re.match(r'^[A-Za-z0-9+/]*={0,2}$', trans.original_text))
                
                # Check if it's NOT readable German/English text
                has_readable_words = any(word in trans.original_text.lower() for word in ['der', 'die', 'das', 'the', 'and', 'ist', 'haben'])
                
                if is_base64 and not has_readable_words:
                    logger.info("✅ Sample data appears encrypted (base64 format, no readable words)")
                    return True
                else:
                    logger.warning("⚠️ Sample data does not look encrypted!")
                    logger.warning(f"Preview: {trans.original_text[:100]}")
                    return False
            else:
                logger.info("ℹ️ Sample has no text content")
                return True
        else:
            logger.info("ℹ️ No encrypted transcriptions found (might be empty database)")
            return True
    finally:
        db.close()

def migrate_existing_users_to_keychains():
    """
    Mark users for migration but DON'T create keychains yet.
    Keychains will be created when they next log in.
    """
    db = SessionLocal()
    
    try:
        users = db.query(User).filter(
            (User.salt == None) | (User.encrypted_master_key == None)
        ).all()
        
        if not users:
            logger.info("ℹ️ All users already have keychains")
            return True
        
        logger.info(f"🔑 Marking {len(users)} users for keychain migration...")
        
        # Just ensure columns exist - don't fill them yet
        for user in users:
            if not user.salt:
                user.salt = None  # Explicit NULL
            if not user.encrypted_master_key:
                user.encrypted_master_key = None  # Explicit NULL
        
        db.commit()
        
        logger.info(f"✅ Users marked for migration. Keychains will be created on next login.")
        return True
        
    except Exception as e:
        db.rollback()
        logger.exception(f"🔥 Migration failed: {e}")
        return False
    finally:
        db.close()

def add_user_key_columns():
    """Add salt and encrypted_master_key to users table"""
    with engine.begin() as conn:
        try:
            # Check columns
            inspector = inspect(engine)
            cols = [c['name'] for c in inspector.get_columns("users")]
            
            if "salt" not in cols:
                logger.info("➕ Adding 'salt' column to users...")
                conn.execute(text("ALTER TABLE users ADD COLUMN salt TEXT"))
            
            if "encrypted_master_key" not in cols:
                logger.info("➕ Adding 'encrypted_master_key' column to users...")
                conn.execute(text("ALTER TABLE users ADD COLUMN encrypted_master_key TEXT"))
                
            logger.info("✅ User table schema updated")
        except Exception as e:
            logger.error(f"❌ Failed to update user schema: {e}")

def main():
    """Main migration workflow"""
    
    print("=" * 60)
    print("🔐 DATABASE ENCRYPTION MIGRATION V2")
    print("=" * 60)
    print()
    
    # Safety check
    response = input("⚠️  This will modify your database. Continue? (type 'YES'): ")
    if response != 'YES':
        print("❌ Migration cancelled")
        return
    
    print()
    logger.info("Starting migration...")
    print()
    
    try:
        # Step 1: Add columns
        logger.info("Step 1/5: Adding encryption columns...")
        add_encryption_columns()
        add_user_key_columns()
        print()
        
        # Step 2: Create new tables
        logger.info("Step 2/5: Creating new tables...")
        create_new_tables()
        print()
        
        # Step 3: Encrypt data (with global key for now)
        logger.info("Step 3/5: Encrypting existing data...")
        success = encrypt_existing_data()
        print()
        
        if not success:
            logger.error("❌ Migration failed during encryption!")
            return
        
        # Step 4: CREATE KEYCHAINS FOR USERS (NEW!)
        logger.info("Step 4/5: Migrating users to per-user encryption...")
        if not migrate_existing_users_to_keychains():
            logger.error("❌ User migration failed!")
            return
        print()
        
        # Step 5: Verify
        logger.info("Step 5/5: Verifying encryption...")
        if verify_encryption():
            print()
            print("=" * 60)
            print("✅ MIGRATION COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print()
            print("Next steps:")
            print("1. Restart the application: sudo systemctl restart transkript.service")
            print("2. All existing users MUST change passwords on next login")
            print("3. New users will automatically use secure per-user encryption")
            print()
        else:
            logger.warning("⚠️ Migration completed but verification failed")
            print("Please check logs and verify manually")
        
    except Exception as e:
        logger.exception(f"🔥 Migration failed: {e}")
        print()
        print("=" * 60)
        print("❌ MIGRATION FAILED")
        print("=" * 60)