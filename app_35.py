# Copyright (C) 2025 CrispStrobe
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# /var/www/transkript_app/app.py
import gradio as gr
import os

# --- FORCE FFMPEG PATH ---
# Explicitly tell Python where to find the tools
os.environ["PATH"] += os.pathsep + "/usr/bin" + os.pathsep + "/usr/local/bin"

try:
    from pydub import AudioSegment
    AudioSegment.converter = "/usr/bin/ffmpeg"
    AudioSegment.ffprobe   = "/usr/bin/ffprobe"
except ImportError:
    print ("pydub cannot be imported")
    pass

import time
import json
import logging
import sys
import base64
import requests
import tempfile
from io import BytesIO
import asyncio
import re
import hashlib

logging.basicConfig(
    level=logging.INFO,  # Global level INFO to stop library spam
    format='%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/var/www/transkript_app/app.log')
    ]
)
# Only set this app to debug
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Set Gradio file upload limits
os.environ['GRADIO_TEMP_DIR'] = '/tmp/gradio'
os.makedirs('/tmp/gradio', exist_ok=True)

try:
    import fastapi_poe as fp
    HAS_POE = True
except ImportError:
    logger.warning("fastapi_poe not installed. Poe API will be unavailable.")
    HAS_POE = False

# Increase max file size
gr.set_static_paths(paths=["/var/www/transkript_app/static"])

# --- IMPORTS & SETUP ---
try:
    from PIL import Image
    import openai
except ImportError:
    print("⚠️ WARNUNG: Bitte 'pip install openai pillow' ausführen.")

# ==========================================
# 🗄️ DATABASE SETUP
# ==========================================

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, Boolean, UniqueConstraint
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import bcrypt

# Database setup
DATABASE_URL = "sqlite:///akademie_suite.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ==========================================
# 📊 DATABASE MODELS
# ==========================================

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    email = Column(String, unique=True, index=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    chat_history = relationship("ChatHistory", back_populates="user", cascade="all, delete-orphan")
    transcriptions = relationship("Transcription", back_populates="user", cascade="all, delete-orphan")
    vision_results = relationship("VisionResult", back_populates="user", cascade="all, delete-orphan")
    generated_images = relationship("GeneratedImage", back_populates="user", cascade="all, delete-orphan")
    custom_prompts = relationship("CustomPrompt", back_populates="user", cascade="all, delete-orphan")
    model_preferences = relationship("UserModelPreference", back_populates="user", cascade="all, delete-orphan")
    
class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    provider = Column(String, nullable=False)
    model = Column(String, nullable=False)
    messages = Column(Text, nullable=False)  # JSON string
    timestamp = Column(DateTime, default=datetime.utcnow)
    title = Column(String)  # Auto-generated summary

    user = relationship("User", back_populates="chat_history")

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
    title = Column(String)  # User-defined or auto-generated

    user = relationship("User", back_populates="transcriptions")

class VisionResult(Base):
    __tablename__ = "vision_results"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    provider = Column(String, nullable=False)
    model = Column(String, nullable=False)
    prompt = Column(Text, nullable=False)
    result = Column(Text, nullable=False)
    image_path = Column(String)  # Store path to uploaded image
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="vision_results")

class GeneratedImage(Base):
    __tablename__ = "generated_images"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    provider = Column(String, nullable=False)
    model = Column(String, nullable=False)
    prompt = Column(Text, nullable=False)
    image_path = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="generated_images")

class CustomPrompt(Base):
    __tablename__ = "custom_prompts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String, nullable=False)
    category = Column(String)  # "chat", "transcription", etc.
    prompt_text = Column(Text, nullable=False)
    is_shared = Column(Boolean, default=False)  # Share with other users
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="custom_prompts")

class UserModelPreference(Base):
    __tablename__ = "user_model_preferences"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    provider = Column(String, nullable=False)
    model_id = Column(String, nullable=False)  # The actual model ID used by API
    display_name = Column(String)  # Human-readable name
    display_order = Column(Integer, default=0)  # Lower = higher priority
    is_visible = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    user = relationship("User", back_populates="model_preferences")
    
    # Unique constraint: one entry per user+provider+model
    __table_args__ = (
        UniqueConstraint('user_id', 'provider', 'model_id', name='_user_provider_model_uc'),
    )
    
# Create tables
Base.metadata.create_all(bind=engine)

# ==========================================
# 🔐 DATABASE HELPER FUNCTIONS
# ==========================================

def get_db():
    """Get database session"""
    return SessionLocal()

def hash_password(password: str) -> str:
    """Hash a password using bcrypt"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its hash"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def authenticate_user(username: str, password: str):
    """Authenticate user and return user object"""
    db = get_db()
    user = db.query(User).filter(User.username == username).first()
    if user and verify_password(password, user.password_hash):
        db.close()
        return user
    db.close()
    return None

def create_default_users():
    """Create default users if they don't exist"""
    db = SessionLocal()
    try:
        # Check if users exist
        if db.query(User).count() == 0:
            # Create admin user
            admin = User(
                username="admin123",
                password_hash=hash_password("ÄndereDasSofort!"),
                email="adminmail@yourdomain.de",
                is_admin=True
            )
            db.add(admin)

            # Create regular user
            user = User(
                username="user123",
                password_hash=hash_password("ÄndereDasAuchGleich!"),
                email="usermail@yourdomain.de",
                is_admin=False
            )
            db.add(user)

            db.commit()
            logger.info("✅ Default users created")
        else:
            logger.info("Users already exist, skipping creation")
    except Exception as e:
        db.rollback()
        logger.exception(f"Error creating default users: {str(e)}")
    finally:
        db.close()

def save_chat_history(user_id: int, provider: str, model: str, messages: list, title: str = None):
    """Save chat conversation to database"""
    db = SessionLocal()
    try:
        chat = ChatHistory(
            user_id=user_id,
            provider=provider,
            model=model,
            messages=json.dumps(messages),
            title=title or f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        db.add(chat)
        db.commit()
        db.refresh(chat)  # Refresh to get the ID
        chat_id = chat.id  # Get ID BEFORE closing session
        return chat_id
    except Exception as e:
        db.rollback()
        logger.exception(f"Error saving chat history: {str(e)}")
        raise
    finally:
        db.close()

def save_transcription(user_id: int, provider: str, model: str, original: str,
                      translated: str = None, language: str = None, filename: str = None, title: str = None):
    """Save transcription to database"""
    db = SessionLocal()
    try:
        trans = Transcription(
            user_id=user_id,
            provider=provider,
            model=model or "N/A",
            original_text=original,
            translated_text=translated,
            language=language,
            filename=filename,
            title=title or f"Transkript {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        db.add(trans)
        db.commit()
        db.refresh(trans)  # Refresh to get the ID
        trans_id = trans.id  # Get ID BEFORE closing session
        return trans_id
    except Exception as e:
        db.rollback()
        logger.exception(f"Error saving transcription: {str(e)}")
        raise
    finally:
        db.close()

def save_vision_result(user_id: int, provider: str, model: str, prompt: str, result: str, image_path: str = None):
    """Save vision analysis to database"""
    db = SessionLocal()
    try:
        vision = VisionResult(
            user_id=user_id,
            provider=provider,
            model=model,
            prompt=prompt,
            result=result,
            image_path=image_path
        )
        db.add(vision)
        db.commit()
        db.refresh(vision)
        vision_id = vision.id
        return vision_id
    except Exception as e:
        db.rollback()
        logger.exception(f"Error saving vision result: {str(e)}")
        raise
    finally:
        db.close()

def save_generated_image(user_id: int, provider: str, model: str, prompt: str, image_path: str):
    """Save generated image to database"""
    db = SessionLocal()
    try:
        img = GeneratedImage(
            user_id=user_id,
            provider=provider,
            model=model,
            prompt=prompt,
            image_path=image_path
        )
        db.add(img)
        db.commit()
        db.refresh(img)
        img_id = img.id
        return img_id
    except Exception as e:
        db.rollback()
        logger.exception(f"Error saving generated image: {str(e)}")
        raise
    finally:
        db.close()

def get_user_transcriptions(user_id: int, limit: int = 50):
    """Get user's transcription history"""
    db = get_db()
    results = db.query(Transcription).filter(
        Transcription.user_id == user_id
    ).order_by(Transcription.timestamp.desc()).limit(limit).all()
    db.close()
    return results

def get_user_custom_prompts(user_id: int, category: str = None):
    """Get user's custom prompts"""
    db = get_db()
    query = db.query(CustomPrompt).filter(CustomPrompt.user_id == user_id)
    if category:
        query = query.filter(CustomPrompt.category == category)
    results = query.order_by(CustomPrompt.timestamp.desc()).all()
    db.close()
    return results

def save_custom_prompt(user_id: int, name: str, prompt_text: str, category: str = "general", is_shared: bool = False):
    """Save a custom prompt template"""
    db = SessionLocal()
    try:
        prompt = CustomPrompt(
            user_id=user_id,
            name=name,
            category=category,
            prompt_text=prompt_text,
            is_shared=is_shared
        )
        db.add(prompt)
        db.commit()
        db.refresh(prompt)
        prompt_id = prompt.id
        return prompt_id
    except Exception as e:
        db.rollback()
        logger.exception(f"Error saving custom prompt: {str(e)}")
        raise
    finally:
        db.close()

# Add after existing database functions

def delete_transcription(trans_id: int, user_id: int):
    """Delete a transcription safely"""
    db = get_db()
    try:
        trans = db.query(Transcription).filter(Transcription.id == trans_id, Transcription.user_id == user_id).first()
        if trans:
            db.delete(trans)
            db.commit()
            return True
        return False
    except Exception as e:
        db.rollback()
        logger.exception(f"DB Error: {e}")
        return False
    finally:
        db.close()

def delete_chat_history(chat_id: int, user_id: int):
    """Delete a chat history safely"""
    db = get_db()
    try:
        chat = db.query(ChatHistory).filter(ChatHistory.id == chat_id, ChatHistory.user_id == user_id).first()
        if chat:
            db.delete(chat)
            db.commit()
            return True
        return False
    except Exception as e:
        db.rollback()
        logger.exception(f"DB Error: {e}")
        return False
    finally:
        db.close()

def delete_vision_result(vision_id: int, user_id: int):
    """Delete a vision result safely"""
    db = get_db()
    try:
        vision = db.query(VisionResult).filter(VisionResult.id == vision_id, VisionResult.user_id == user_id).first()
        if vision:
            db.delete(vision)
            db.commit()
            return True
        return False
    except Exception as e:
        db.rollback()
        logger.exception(f"DB Error: {e}")
        return False
    finally:
        db.close()

def delete_generated_image(img_id: int, user_id: int):
    """Delete a generated image safely"""
    db = get_db()
    try:
        img = db.query(GeneratedImage).filter(GeneratedImage.id == img_id, GeneratedImage.user_id == user_id).first()
        if img:
            if img.image_path and os.path.exists(img.image_path):
                try: os.remove(img.image_path)
                except: pass
            db.delete(img)
            db.commit()
            return True
        return False
    except Exception as e:
        db.rollback()
        logger.exception(f"DB Error: {e}")
        return False
    finally:
        db.close()

def delete_custom_prompt(prompt_id: int, user_id: int):
    """Delete a custom prompt"""
    db = get_db()
    prompt = db.query(CustomPrompt).filter(
        CustomPrompt.id == prompt_id,
        CustomPrompt.user_id == user_id
    ).first()
    if prompt:
        db.delete(prompt)
        db.commit()
        db.close()
        return True
    db.close()
    return False

def get_user_chat_history(user_id: int, limit: int = 50):
    """Get user's chat history"""
    db = get_db()
    results = db.query(ChatHistory).filter(
        ChatHistory.user_id == user_id
    ).order_by(ChatHistory.timestamp.desc()).limit(limit).all()
    db.close()
    return results

def get_single_chat(chat_id: int, user_id: int):
    """Get a single chat conversation"""
    db = get_db()
    chat = db.query(ChatHistory).filter(
        ChatHistory.id == chat_id,
        ChatHistory.user_id == user_id
    ).first()
    db.close()
    return chat

def get_user_vision_results(user_id: int, limit: int = 50):
    """Get user's vision results"""
    db = get_db()
    results = db.query(VisionResult).filter(
        VisionResult.user_id == user_id
    ).order_by(VisionResult.timestamp.desc()).limit(limit).all()
    db.close()
    return results

def get_user_generated_images(user_id: int, limit: int = 50):
    """Get user's generated images"""
    db = get_db()
    results = db.query(GeneratedImage).filter(
        GeneratedImage.user_id == user_id
    ).order_by(GeneratedImage.timestamp.desc()).limit(limit).all()
    db.close()
    return results

# Initialize default users
create_default_users()


# ==========================================
# 📦 STORAGE BOX HELPER
# ==========================================
STORAGE_MOUNT_POINT = "/mnt/akademie_storage"

def ensure_user_storage_dirs(username):
    """Creates /shared and /username folders on the Storage Box"""
    if not os.path.exists(STORAGE_MOUNT_POINT):
        return False # Mount not active
        
    shared_path = os.path.join(STORAGE_MOUNT_POINT, "shared")
    user_path = os.path.join(STORAGE_MOUNT_POINT, username)
    
    os.makedirs(shared_path, exist_ok=True)
    os.makedirs(user_path, exist_ok=True)
    return True

def copy_storage_file_to_temp(file_path):
    """
    Copies a file from Storage Box to local temp for processing.
    Returns path to local temp file.
    """
    if not file_path: return None
    
    # Security check: Ensure path is within mount point
    abs_path = os.path.abspath(file_path)
    if not abs_path.startswith(os.path.abspath(STORAGE_MOUNT_POINT)):
        raise ValueError("Zugriff verweigert: Datei liegt außerhalb der Storage Box")
        
    if not os.path.exists(abs_path):
        raise ValueError("Datei existiert nicht")
        
    # Copy to temp
    filename = os.path.basename(abs_path)
    temp_dir = tempfile.gettempdir()
    local_dest = os.path.join(temp_dir, f"sb_{int(time.time())}_{filename}")
    
    import shutil
    shutil.copy2(abs_path, local_dest)
    return local_dest

def get_storage_root(user_state=None):
    """Returns valid root for FileExplorer based on login"""
    if user_state and user_state.get("id"):
        # Ensure folders exist
        ensure_user_storage_dirs(user_state.get("username"))
        return STORAGE_MOUNT_POINT
    return None

# ==========================================
# AUDIO HELPERS
# ==========================================

import shutil

JOB_STATE_DIR = "/var/www/transkript_app/jobs"
os.makedirs(JOB_STATE_DIR, exist_ok=True)

def create_job_manifest(job_id, audio_path, provider, model, chunks, lang, prompt, temp):
    """Save job state to disk"""
    manifest = {
        "job_id": job_id,
        "audio_path": audio_path,
        "provider": provider,
        "model": model,
        "lang": lang,
        "prompt": prompt,
        "temp": temp,
        "created_at": time.time(),
        "chunks": [{"path": p, "status": "pending"} for p in chunks],
        "transcript_parts": [""] * len(chunks)
    }
    with open(os.path.join(JOB_STATE_DIR, f"{job_id}.json"), "w") as f:
        json.dump(manifest, f)
    return manifest

def update_job_chunk_status(job_id, chunk_index, status, text=None):
    """Update status of a specific chunk"""
    path = os.path.join(JOB_STATE_DIR, f"{job_id}.json")
    if not os.path.exists(path): return
    
    with open(path, "r") as f:
        manifest = json.load(f)
    
    manifest["chunks"][chunk_index]["status"] = status
    if text is not None:
        manifest["transcript_parts"][chunk_index] = text
        
    with open(path, "w") as f:
        json.dump(manifest, f)

def get_failed_jobs():
    """List incomplete jobs"""
    jobs = []
    if not os.path.exists(JOB_STATE_DIR): return []
    
    for f in os.listdir(JOB_STATE_DIR):
        if f.endswith(".json"):
            try:
                with open(os.path.join(JOB_STATE_DIR, f), "r") as jf:
                    data = json.load(jf)
                    # Check if any chunk is not 'done'
                    pending = sum(1 for c in data["chunks"] if c["status"] != "done")
                    if pending > 0:
                        jobs.append([
                            data["job_id"], 
                            datetime.fromtimestamp(data["created_at"]).strftime("%Y-%m-%d %H:%M"),
                            f"{pending}/{len(data['chunks'])} offen",
                            data["provider"]
                        ])
            except: pass
    return jobs

def get_file_hash(filepath):
    """Generate SHA256 hash for file to track progress"""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def cleanup_chunks(chunk_dir):
    """Safely remove temporary chunk directory"""
    try:
        if chunk_dir and os.path.exists(chunk_dir):
            shutil.rmtree(chunk_dir)
            logger.info(f"Cleaned up chunk directory: {chunk_dir}")
    except Exception as e:
        logger.warning(f"Failed to cleanup chunks: {e}")

def split_audio_into_chunks(audio_path, chunk_minutes=10):
    """
    Splits audio into segments to bypass API size/duration limits.
    Returns: (List of file paths, temp_directory_path)
    """
    try:
        # --- FIX: Ensure imports exist ---
        import math 
        from pydub import AudioSegment
        
        # Create unique temp dir
        job_id = int(time.time())
        chunk_dir = os.path.join(tempfile.gettempdir(), f"transcription_chunks_{job_id}")
        os.makedirs(chunk_dir, exist_ok=True)
        
        logger.info(f"Loading audio for splitting: {audio_path}")
        audio = AudioSegment.from_file(audio_path)
        
        duration_ms = len(audio)
        chunk_length_ms = chunk_minutes * 60 * 1000
        total_chunks = math.ceil(duration_ms / chunk_length_ms)
        
        logger.info(f"Audio duration: {duration_ms/1000/60:.2f} min. Splitting into {total_chunks} chunks.")
        
        chunk_paths = []
        
        for i in range(total_chunks):
            start_ms = i * chunk_length_ms
            end_ms = min((i + 1) * chunk_length_ms, duration_ms)
            
            chunk = audio[start_ms:end_ms]
            
            # Export with low compression to keep size down but quality up
            chunk_filename = os.path.join(chunk_dir, f"chunk_{i:03d}.mp3")
            chunk.export(chunk_filename, format="mp3", bitrate="128k")
            chunk_paths.append(chunk_filename)
            
        return chunk_paths, chunk_dir
        
    except Exception as e:
        logger.exception(f"Error splitting audio: {e}")
        return None, None
    
def split_audio(filepath, chunk_length_ms=600000): # 10 minutes default (safe for Mistral 15m limit)
    """Split audio into chunks and return list of temp file paths"""
    try:
        audio = AudioSegment.from_file(filepath)
        chunks = []
        duration_ms = len(audio)
        
        # Create temp dir for chunks
        chunk_dir = os.path.join(tempfile.gettempdir(), "audio_chunks")
        os.makedirs(chunk_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        
        for i, start_ms in enumerate(range(0, duration_ms, chunk_length_ms)):
            end_ms = min(start_ms + chunk_length_ms, duration_ms)
            chunk = audio[start_ms:end_ms]
            
            # Export chunk
            chunk_path = os.path.join(chunk_dir, f"{base_name}_part_{i}.mp3")
            chunk.export(chunk_path, format="mp3")
            chunks.append(chunk_path)
            
        return chunks
    except Exception as e:
        logger.error(f"Error splitting audio: {e}")
        return None
    
# ==========================================
# 👥 USER MANAGEMENT FUNCTIONS
# ==========================================

def create_user(username, password, email, is_admin=False):
    """Create a new user"""
    db = SessionLocal()
    try:
        # Check if username already exists
        existing_user = db.query(User).filter(User.username == username).first()
        if existing_user:
            return False, "❌ Benutzername existiert bereits"
        
        # Check if email already exists
        if email:
            existing_email = db.query(User).filter(User.email == email).first()
            if existing_email:
                return False, "❌ E-Mail wird bereits verwendet"
        
        # Create new user
        new_user = User(
            username=username,
            password_hash=hash_password(password),
            email=email,
            is_admin=is_admin
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        logger.info(f"User created: {username} (Admin: {is_admin})")
        return True, f"✅ Benutzer '{username}' erfolgreich erstellt"
        
    except Exception as e:
        db.rollback()
        logger.exception(f"Error creating user: {str(e)}")
        return False, f"🔥 Fehler: {str(e)}"
    finally:
        db.close()

def delete_user(user_id, current_user_id):
    """Delete a user (cannot delete self)"""
    db = SessionLocal()
    try:
        if user_id == current_user_id:
            return False, "❌ Du kannst dich nicht selbst löschen"
        
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return False, "❌ Benutzer nicht gefunden"
        
        username = user.username
        db.delete(user)
        db.commit()
        
        logger.info(f"User deleted: {username} (ID: {user_id})")
        return True, f"✅ Benutzer '{username}' gelöscht"
    except Exception as e:
        db.rollback()
        logger.exception(f"Error deleting user: {str(e)}")
        return False, f"🔥 Fehler: {str(e)}"
    finally:
        db.close()

def rename_user(user_id, new_username):
    """Rename a user"""
    db = SessionLocal()
    try:
        # Check if new username already exists
        existing = db.query(User).filter(User.username == new_username).first()
        if existing and existing.id != user_id:
            return False, "❌ Benutzername existiert bereits"
        
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return False, "❌ Benutzer nicht gefunden"
        
        old_username = user.username
        user.username = new_username
        db.commit()
        
        logger.info(f"User renamed: {old_username} → {new_username}")
        return True, f"✅ Benutzer umbenannt: '{old_username}' → '{new_username}'"
        
    except Exception as e:
        db.rollback()
        logger.exception(f"Error renaming user: {str(e)}")
        return False, f"🔥 Fehler: {str(e)}"
    finally:
        db.close()

def reset_user_password(user_id, new_password):
    """Reset a user's password"""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return False, "❌ Benutzer nicht gefunden"
        
        user.password_hash = hash_password(new_password)
        db.commit()
        
        logger.info(f"Password reset for user: {user.username}")
        return True, f"✅ Passwort für '{user.username}' zurückgesetzt"
        
    except Exception as e:
        db.rollback()
        logger.exception(f"Error resetting password: {str(e)}")
        return False, f"🔥 Fehler: {str(e)}"
    finally:
        db.close()

def toggle_admin_status(user_id, current_user_id):
    """Toggle admin status for a user (cannot change own status)"""
    db = SessionLocal()
    try:
        # Prevent changing own admin status
        if user_id == current_user_id:
            return False, "❌ Du kannst deinen eigenen Admin-Status nicht ändern"
        
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return False, "❌ Benutzer nicht gefunden"
        
        # Toggle admin status
        user.is_admin = not user.is_admin
        db.commit()
        
        status = "Admin" if user.is_admin else "Normaler Benutzer"
        logger.info(f"Admin status changed for {user.username}: {status}")
        return True, f"✅ '{user.username}' ist jetzt: {status}"
        
    except Exception as e:
        db.rollback()
        logger.exception(f"Error toggling admin status: {str(e)}")
        return False, f"🔥 Fehler: {str(e)}"
    finally:
        db.close()

def get_all_users():
    """Get all users for admin panel"""
    db = SessionLocal()
    try:
        users = db.query(User).order_by(User.created_at.desc()).all()
        data = []
        for u in users:
            data.append([
                u.id,
                u.username,
                u.email or "—",
                "✅ Admin" if u.is_admin else "👤 User",
                u.created_at.strftime("%Y-%m-%d %H:%M")
            ])
        return data
    except Exception as e:
        logger.exception(f"Error getting users: {str(e)}")
        return []
    finally:
        db.close()

def update_user_email(user_id, new_email):
    """Update user's email address"""
    db = SessionLocal()
    try:
        # Check if email already exists
        if new_email:
            existing = db.query(User).filter(User.email == new_email).first()
            if existing and existing.id != user_id:
                return False, "❌ E-Mail wird bereits verwendet"
        
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return False, "❌ Benutzer nicht gefunden"
        
        user.email = new_email
        db.commit()
        
        logger.info(f"Email updated for user: {user.username}")
        return True, f"✅ E-Mail für '{user.username}' aktualisiert"
        
    except Exception as e:
        db.rollback()
        logger.exception(f"Error updating email: {str(e)}")
        return False, f"🔥 Fehler: {str(e)}"
    finally:
        db.close()


# ==========================================
# 🎯 MODEL PREFERENCES MANAGEMENT
# ==========================================

def fetch_available_models(provider, api_key=None):
    """
    Ruft verfügbare Modelle ab. 
    Für Poe: Nutzt die offizielle API und sortiert nach Modalitäten (Text, Bild, etc.).
    """
    try:
        if provider == "Poe":
            api_key = api_key or API_KEYS.get("POE")
            if not api_key:
                return None, "API Key erforderlich"

            try:
                # Direkter Request an die Poe API, da wir die Metadaten brauchen
                headers = {"Authorization": f"Bearer {api_key}"}
                response = requests.get("https://api.poe.com/v1/models", headers=headers, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    model_list = []
                    
                    # Wir kategorisieren die Modelle für die Rückgabe
                    # (Standardmäßig geben wir alle zurück, aber fügen Infos hinzu)
                    for model in data.get("data", []):
                        arch = model.get("architecture", {})
                        inputs = arch.get("input_modalities", [])
                        outputs = arch.get("output_modalities", [])
                        
                        # Bestimme den Typ für die Anzeige
                        type_label = "Chat"
                        if "image" in outputs:
                            type_label = "Bild-Gen"
                        elif "video" in outputs:
                            type_label = "Video-Gen"
                        elif "audio" in outputs:
                            type_label = "Audio-Gen"
                        elif "image" in inputs and "text" in outputs:
                            type_label = "Vision/Chat"
                        
                        model_list.append({
                            "id": model["id"],
                            "name": f"{model.get('metadata', {}).get('display_name', model['id'])} ({type_label})",
                            "type": type_label, # Hilfreich für Filterung später
                            "context_length": model.get("context_length", 0)
                        })
                    
                    # Sortieren nach Name
                    model_list.sort(key=lambda x: x["name"])
                    return model_list, None
                else:
                    return None, f"Poe API Fehler: {response.status_code} - {response.text}"
            except Exception as e:
                return None, f"Poe Verbindungsfehler: {str(e)}"

        elif provider == "Scaleway":
            # Scaleway uses OpenAI-compatible API
            api_key = api_key or API_KEYS.get("SCALEWAY")
            if not api_key:
                return None, "API Key erforderlich"
            
            try:
                client = openai.OpenAI(base_url="https://api.scaleway.ai/v1", api_key=api_key)
                models = client.models.list()
                model_list = []
                for model in models.data:
                    model_list.append({
                        "id": model.id,
                        "name": model.id,
                        "owned_by": getattr(model, 'owned_by', 'scaleway')
                    })
                return model_list, None
            except Exception as e:
                return None, f"Fehler: {str(e)}"
        
        elif provider == "Nebius":
            # Nebius uses OpenAI-compatible API
            api_key = api_key or API_KEYS.get("NEBIUS")
            if not api_key:
                return None, "API Key erforderlich"
            
            try:
                client = openai.OpenAI(base_url="https://api.tokenfactory.nebius.com/v1", api_key=api_key)
                models = client.models.list()
                model_list = []
                for model in models.data:
                    model_list.append({
                        "id": model.id,
                        "name": model.id,
                        "owned_by": getattr(model, 'owned_by', 'nebius')
                    })
                return model_list, None
            except Exception as e:
                return None, f"Fehler: {str(e)}"
        
        elif provider == "Mistral":
            # Mistral API
            api_key = api_key or API_KEYS.get("MISTRAL")
            if not api_key:
                return None, "API Key erforderlich"
            
            try:
                client = openai.OpenAI(base_url="https://api.mistral.ai/v1", api_key=api_key)
                models = client.models.list()
                model_list = []
                for model in models.data:
                    model_list.append({
                        "id": model.id,
                        "name": model.id,
                        "owned_by": getattr(model, 'owned_by', 'mistral')
                    })
                return model_list, None
            except Exception as e:
                return None, f"Fehler: {str(e)}"
        
        elif provider == "OpenRouter":
            # OpenRouter models list
            try:
                response = requests.get("https://openrouter.ai/api/v1/models", timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    model_list = []
                    for model in data.get("data", []):
                        model_list.append({
                            "id": model["id"],
                            "name": model.get("name", model["id"]),
                            "context_length": model.get("context_length", 0)
                        })
                    return model_list, None
                return None, f"Fehler: Status {response.status_code}"
            except Exception as e:
                return None, f"Fehler: {str(e)}"
        
        elif provider == "Groq":
            # Groq API - use REST endpoint instead of client library
            api_key = api_key or API_KEYS.get("GROQ")
            if not api_key:
                return None, "API Key erforderlich"
            
            try:
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                response = requests.get("https://api.groq.com/openai/v1/models", headers=headers, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    model_list = []
                    for model in data.get("data", []):
                        model_list.append({
                            "id": model["id"],
                            "name": model["id"],
                            "owned_by": model.get("owned_by", "groq")
                        })
                    return model_list, None
                return None, f"Fehler: Status {response.status_code}"
            except Exception as e:
                return None, f"Fehler: {str(e)}"
        
        else:
            # For providers without /models endpoint, return configured models
            provider_config = PROVIDERS.get(provider, {})
            chat_models = provider_config.get("chat_models", [])
            model_list = [{"id": m, "name": m} for m in chat_models]
            return model_list, None
    
    except Exception as e:
        logger.exception(f"Error fetching models for {provider}: {str(e)}")
        return None, f"Fehler: {str(e)}"

def get_user_model_preferences(user_id, provider):
    """Get user's model preferences for a provider"""
    db = SessionLocal()
    try:
        prefs = db.query(UserModelPreference).filter(
            UserModelPreference.user_id == user_id,
            UserModelPreference.provider == provider
        ).order_by(UserModelPreference.display_order).all()
        
        return prefs
    except Exception as e:
        logger.exception(f"Error getting user model preferences: {str(e)}")
        return []
    finally:
        db.close()

def save_user_model_preferences(user_id, provider, model_configs):
    """
    Save user's model preferences
    model_configs: list of dicts with keys: model_id, display_name, is_visible, display_order
    """
    db = SessionLocal()
    try:
        # Delete existing preferences for this provider
        db.query(UserModelPreference).filter(
            UserModelPreference.user_id == user_id,
            UserModelPreference.provider == provider
        ).delete()
        
        # Add new preferences
        for config in model_configs:
            pref = UserModelPreference(
                user_id=user_id,
                provider=provider,
                model_id=config["model_id"],
                display_name=config.get("display_name", config["model_id"]),
                display_order=config.get("display_order", 0),
                is_visible=config.get("is_visible", True)
            )
            db.add(pref)
        
        db.commit()
        logger.info(f"Saved {len(model_configs)} model preferences for user {user_id}, provider {provider}")
        return True, "✅ Einstellungen gespeichert"
        
    except Exception as e:
        db.rollback()
        logger.exception(f"Error saving model preferences: {str(e)}")
        return False, f"🔥 Fehler: {str(e)}"
    finally:
        db.close()

def get_user_visible_models(user_id, provider):
    """Get list of visible models for user, in order"""
    prefs = get_user_model_preferences(user_id, provider)
    
    if not prefs:
        # No preferences set, return all models from provider config
        provider_config = PROVIDERS.get(provider, {})
        return provider_config.get("chat_models", [])
    
    # Return visible models in order
    visible_models = [p.model_id for p in prefs if p.is_visible]
    return visible_models if visible_models else [prefs[0].model_id] if prefs else []

def get_default_model_for_user(user_id, provider):
    """Get user's default model (first in order)"""
    visible = get_user_visible_models(user_id, provider)
    return visible[0] if visible else None
        
# ==========================================
# ⚙️ ZENTRALE KONFIGURATION
# ==========================================

# API Keys (aus Environment oder direkt hier eintragen)
API_KEYS = {
    "SCALEWAY": os.environ.get("SCALEWAY_API_KEY", "your_key"),
    "NEBIUS": os.environ.get("NEBIUS_API_KEY", "your_key"),
    "MISTRAL": os.environ.get("MISTRAL_API_KEY", "your_key"),
    "GLADIA": os.environ.get("GLADIA_API_KEY", "your_key"),
    "OPENROUTER": os.environ.get("OPENROUTER_API_KEY", "your_key"),
    "GROQ": os.environ.get("GROQ_API_KEY", "your_key"),
    "POE": os.environ.get("POE_API_KEY", "your_poe_key_here"),
    "DEEPGRAM": os.environ.get("DEEPGRAM_API_KEY", "your_key"), 
    "ASSEMBLYAI": os.environ.get("ASSEMBLYAI_API_KEY", "your_key"), 
}

# Provider-Datenbank (Modelle, Endpoints, Compliance)
PROVIDERS = {
    "Scaleway": {
        "base_url": "https://api.scaleway.ai/v1",
        "key_name": "SCALEWAY",
        "badge": "🇫🇷 <span style='color:green'><b>DSGVO-Konform</b> (Frankreich)</span>",
        "chat_models": ["gpt-oss-120b", "mistral-small-3.2-24b-instruct-2506", "gemma-3-27b-it", "qwen3-235b-a22b-instruct-2507", "llama-3.3-70b-instruct", "deepseek-r1-distill-llama-70b"],
        "vision_models": ["pixtral-12b-2409", "mistral-small-3.1-24b-instruct-2503"],
        "audio_models": ["whisper-large-v3"],
        "image_models": ["pixtral-12b-2409"],
    },
    "Nebius": {
        "base_url": "https://api.tokenfactory.nebius.com/v1",
        "key_name": "NEBIUS",
        "badge": "🇪🇺 <span style='color:green'><b>DSGVO-Konform</b> (EU-Rechenzentren)</span>",
        "chat_models": ["deepseek-ai/DeepSeek-R1-0528", "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1", "openai/gpt-oss-120b", "moonshotai/Kimi-K2-Instruct", "moonshotai/Kimi-K2-Thinking", "zai-org/GLM-4.5", "meta-llama/Llama-3.3-70B-Instruct"],
        "image_models": ["black-forest-labs/flux-schnell", "black-forest-labs/flux-dev"]
    },
    "Mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "key_name": "MISTRAL",
        "badge": "🇫🇷 <span style='color:green'><b>DSGVO-Konform</b> (Frankreich)</span>",
        "chat_models": ["mistral-large-latest", "mistral-medium-2508", "magistral-medium-2509", "open-mistral-nemo-2407"],
        "vision_models": ["pixtral-large-2411", "pixtral-12b-2409", "mistral-ocr-latest"],
        "audio_models": ["voxtral-mini-latest"]
    },
    "Deepgram": {
        "base_url": "https://api.eu.deepgram.com/v1",
        "key_name": "DEEPGRAM",
        "badge": "🇪🇺 <span style='color:green'><b>DSGVO-Konform</b> (EU-Rechenzentren)</span>",
        "audio_models": ["nova-2-general", "nova-3-general", "nova-2"], 
    },
    "AssemblyAI": {
        "base_url": "https://api.eu.assemblyai.com/v2",
        "key_name": "ASSEMBLYAI",
        "badge": "🇪🇺 <span style='color:green'><b>DSGVO-Konform</b> (EU-Rechenzentren)</span>",
        "audio_models": ["universal", "slam-1"],
    },
    "OpenRouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_name": "OPENROUTER",
        "badge": "🇺🇸 <span style='color:orange'><b>US-Server</b> (Daten verlassen EU)</span>",
        "chat_models": ["z-ai/glm-4.5-air:free", "tngtech/deepseek-r1t2-chimera:free", "qwen/qwen3-235b-a22b:free", "x-ai/grok-4.1-fast:free", "google/gemini-3-pro-preview"],
        "vision_models": ["x-ai/grok-4.1-fast:free", "amazon/nova-2-lite-v1:free", "nvidia/nemotron-nano-12b-v2-vl:free", "google/gemma-3-27b-it:free", "google/gemini-2.0-flash-exp:free", "google/gemma-3-27b-it:free"],
        "audio_models": ["google/gemini-2.0-flash-lite-001", "mistralai/voxtral-small-24b-2507", "mistralai/voxtral-small-24b-2507", "google/gemini-2.5-flash-lite"],
        "image_models": ["google/gemini-2.5-flash-image", "openai/gpt-5-image-mini", "google/gemini-3-pro-image-preview", "black-forest-labs/flux.2-pro", "black-forest-labs/flux.2-flex"]
    },
    "Groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_name": "GROQ",
        "badge": "🇺🇸 <span style='color:orange'><b>US-Server</b> (Daten verlassen EU)</span>",
        "chat_models": ["openai/gpt-oss-120b", "moonshotai/kimi-k2-instruct-0905", "meta-llama/llama-4-scout-17b-16e-instruct", "llama-3.3-70b-versatile", "qwen/qwen3-32b"],
        "audio_models": ["whisper-large-v3-turbo"],
        "vision_models": ["meta-llama/llama-4-scout-17b-16e-instruct", "meta-llama/llama-4-maverick-17b-128e-instruct"]
    },
    "Poe": {
        "base_url": "https://api.poe.com/v1",
        "key_name": "POE",
        "badge": "🌐 <span style='color:blue'><b>Poe Official API</b> (Universal)</span>",
        "chat_models": [
            "gpt-5.1-instant",
            "claude-sonnet-4.5",
            "gemini-3-pro",
            "gpt-5.1",
            "gpt-4o",
            "claude-3.5-sonnet",
            "deepseek-r1",
            "grok-4"
        ],
        "vision_models": [
            "claude-sonnet-4.5",
            "gpt-5.1",
            "gemini-3-pro",
            "gpt-4o",
            "claude-3.5-sonnet"
        ],
        "image_models": [
            "gpt-image-1",
            "flux-pro-1.1-ultra",
            "ideogram-v3",
            "dall-e-3",
            "playground-v3"
        ],
        "audio_models": [
            "elevenlabs-v3",
            "sonic-3.0"
        ],
        "video_models": [ # Neu für Poe!
            "kling-2.5-turbo-pro",
            "runway-gen-4-turbo",
            "veo-3.1"
        ],
        "supports_system": True,
        "supports_streaming": True
    }
}

def get_compliance_html(provider):
    """Gibt nur den Text/Icon zurück. Styling erfolgt im UI-Update."""
    badges = {
        "Scaleway": "🇫🇷 <b>DSGVO-Konform</b> (Frankreich)",
        "Nebius": "🇪🇺 <b>DSGVO-Konform</b> (EU-Rechenzentren)",
        "Mistral": "🇫🇷 <b>DSGVO-Konform</b> (Frankreich)",
        "Gladia": "🇫🇷 <b>DSGVO-Konform</b> (Frankreich)",
        "OpenRouter": "🇺🇸 <b>US-Server</b> (Nicht DSGVO, z.T. kostenlos)",
        "Groq": "🇺🇸 <b>US-Server</b> (Nicht DSGVO, Schnell, z.T. kostenlos)",
        "Deepgram": "🇪🇺 <b>DSGVO-Konform</b> (EU-Rechenzentren)", 
        "AssemblyAI": "🇪🇺 <b>DSGVO-Konform</b> (EU-Rechenzentren)", 
    }
    # Fix: Return raw string only. No <div> wrapper here.
    return badges.get(provider, "❓ Unbekannt")

# Gladia Spezial-Config
GLADIA_CONFIG = {
    "url": "https://api.gladia.io/v2",
    "vocab": [
        "Christian Ströbele", "Konstanze Jüngling", "Fabian Jaskolla", "Jesus Christus", "Amen", "Halleluja",
        "Evangelium", "Predigt", "Liturgie", "Gottesdienst", "Pfarrei",
        "Diözese", "Kirchenvorstand", "Fürbitten", "Akademie",
        "Tagungshaus", "Compliance", "Synode", "Ökumene"
    ]
}

def login_user(username, password):
    """Login function that returns a session dict"""
    user = authenticate_user(username, password)
    if user:
        # Create the session data dictionary
        new_state = {
            "id": user.id,
            "username": user.username,
            "is_admin": user.is_admin
        }
        
        welcome_msg = f"✅ Willkommen, {user.username}!"
        # Show app, Hide login, Update State
        return (
            True,
            welcome_msg, 
            gr.update(visible=True), 
            gr.update(visible=False), 
            new_state 
        )
    
    # Login failed: Return empty state
    return (
        False, # <--- ADDED Success Flag
        "❌ Ungültige Anmeldedaten", 
        gr.update(visible=False), 
        gr.update(visible=True), 
        {"id": None, "username": None, "is_admin": False}
    )

def logout_user():
    """Logout function returns empty state"""
    # Return empty dictionary to reset session_state
    empty_state = {"id": None, "username": None, "is_admin": False}
    return f"👋 Auf Wiedersehen!", gr.update(visible=False), gr.update(visible=True), empty_state

# ==========================================
# 🛠️ HELPER FUNCTIONS
# ==========================================

def get_client(provider_name, api_key_override=None):
    """Factory: Erstellt einen OpenAI-Client für JEDEN Provider"""
    conf = PROVIDERS.get(provider_name)
    if not conf: raise ValueError(f"Unbekannter Provider: {provider_name}")

    key = api_key_override if api_key_override else API_KEYS.get(conf["key_name"])
    if not key: raise ValueError(f"Kein API Key für {provider_name} gefunden.")

    return openai.OpenAI(base_url=conf["base_url"], api_key=key)

def call_poe_sync(messages, model, api_key):
    """
    Synchronously call Poe API by running async code
    """
    import asyncio
    
    async def call_poe_async():
        poe_messages = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            # Skip system messages - Poe doesn't support them
            if role == "system":
                continue
            
            # Convert 'assistant' to 'bot' for Poe
            if role == "assistant":
                role = "bot"
            
            # Handle multimodal content
            if isinstance(content, list):
                text_parts = [item.get("text", "") for item in content if item.get("type") == "text"]
                content = " ".join(text_parts)
            
            # Convert to Poe ProtocolMessage
            poe_messages.append(fp.ProtocolMessage(role=role, content=content))
        
        # Call Poe API and collect response
        full_response = ""
        try:
            async for partial in fp.get_bot_response(
                messages=poe_messages,
                bot_name=model,
                api_key=api_key
            ):
                full_response += partial.text
        except Exception as e:
            raise Exception(f"Poe API error: {str(e)}")
        
        return full_response
    
    # Run the async function synchronously
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(call_poe_async())
        loop.close()
        return result
    except Exception as e:
        raise Exception(f"Poe Fehler: {str(e)}")

def encode_image(image_path):
    if not image_path: return None
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def format_duration(seconds):
    if seconds is None: return "00:00"
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{int(s):02d}"

def run_deepgram_transcription(audio_path, model, lang, diar, key):
    """
    Handle Deepgram transcription using the EU endpoint.
    Uses a single request for simplicity, relying on Deepgram's direct support.
    """
    import requests
    import json
    
    API_URL = "https://api.eu.deepgram.com/v1/listen"
    
    # KORRIGIERTE LOGIK: Verwende UI-Key oder falle auf systemweit konfigurierten Key zurück
    api_key = key if key else API_KEYS.get("DEEPGRAM")

    if not api_key or api_key == "your_key":
        yield "❌ Kein Deepgram Key gefunden oder konfiguriert.", ""
        return

    logs = "🚀 Starte Deepgram Upload (EU Endpoint)..."
    yield logs, ""
    
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/octet-stream"
    }

    # Deepgram parameters: use Nova-2/3 by default
    query_params = {
        "model": model or "nova-2-general", 
        "smart_format": "true",
        "language": lang if lang and lang != "auto" else "de", # Default to German
        "punctuate": "true",
        "diarize": "true" if diar else "false",
        "paragraphs": "true" if diar else "false", # Paragraphs/Utterances is usually required for diarization output
        "utterances": "true" if diar else "false",
    }
    
    try:
        with open(audio_path, "rb") as audio_file:
            response = requests.post(
                API_URL, 
                headers=headers, 
                params=query_params, 
                data=audio_file,
                timeout=300 # 5 minutes for upload and transcription
            )
        
        response.raise_for_status()
        result = response.json()
        
        logs += "\n✅ Upload & Transkription fertig."
        yield logs, ""
        
        # --- Parsing Results ---
        final_text = ""
        if diar and result.get('results', {}).get('utterances'):
            utterances = result['results']['utterances']
            for utt in utterances:
                speaker = utt.get('speaker', 0)
                transcript = utt.get('transcript', '').strip()
                start_sec = utt.get('start', 0) # Seconds
                
                time_code = format_duration(start_sec)
                final_text += f"[{time_code}] Sprecher {speaker}: {transcript}\n"
            
        else:
            # Standard Transcript
            transcript = result.get('results', {}).get('channels', [{}])[0].get('alternatives', [{}])[0].get('transcript', '')
            final_text = transcript
            
        if not final_text.strip():
            final_text = "❌ Transkript ist leer! Bitte Sprache prüfen."

        yield logs + "\n🎉 Fertig!", final_text.strip()
        
    except requests.exceptions.RequestException as e:
        error_msg = f"Deepgram API Fehler: {e}"
        if 'response' in locals() and response.status_code == 400:
            try:
                error_msg += f"\nDetails: {response.json().get('err_msg', response.text)}"
            except: pass
        
        logger.exception(f"Deepgram Error: {e}")
        yield logs + f"\n🔥 Abbruch: {error_msg}", ""
    except Exception as e:
        logger.exception(f"Deepgram Error: {e}")
        yield logs + f"\n🔥 Abbruch: {str(e)}", ""

def run_assemblyai_transcription(audio_path, model, lang, diar, key):
    """
    Handle AssemblyAI transcription using the EU endpoint.
    Requires: Upload -> Job Submission -> Polling (Generator Pattern).
    """
    import requests
    import time
    import json
    
    EU_BASE_URL = "https://api.eu.assemblyai.com/v2"

    # KORRIGIERTE LOGIK: Verwende UI-Key oder falle auf systemweit konfigurierten Key zurück
    api_key = key if key else API_KEYS.get("ASSEMBLYAI")
    
    if not api_key or api_key == "your_key":
        yield "❌ Kein AssemblyAI Key gefunden oder konfiguriert.", ""
        return

    logs = "🚀 Starte AssemblyAI Upload (EU Endpoint)..."
    yield logs, ""
    
    headers = {"Authorization": api_key}
    
    try:
        # A. Upload
        upload_url = f"{EU_BASE_URL}/upload"
        upload_headers = headers.copy()
        upload_headers["Content-Type"] = "application/octet-stream"

        with open(audio_path, "rb") as audio_file:
            response = requests.post(upload_url, headers=upload_headers, data=audio_file, timeout=300)
        response.raise_for_status()
        upload_result = response.json()
        final_audio_url = upload_result["upload_url"]
        
        logs += "\n✅ Upload erfolgreich. Starte Job..."
        yield logs, ""
        
        # B. Job Submission
        submit_url = f"{EU_BASE_URL}/transcript"
        
        # Use recommended defaults for general, accurate transcription
        request_data = {
            "audio_url": final_audio_url,
            "language_code": lang if lang and lang != "auto" else "de",
            "punctuate": True,
            "format_text": True,
            "speaker_labels": diar, # Diarization
            "speech_models": [model or "universal"], # Use selected model
        }

        response = requests.post(submit_url, headers=headers, json=request_data)
        response.raise_for_status()
        submit_result = response.json()
        transcript_id = submit_result["id"]
        
        logs += f"\n✅ Job ID: {transcript_id}. Starte Polling..."
        yield logs, ""
        
        # C. Polling
        get_url = f"{EU_BASE_URL}/transcript/{transcript_id}"
        poll_count = 0
        
        while True:
            time.sleep(5)
            poll_count += 1
            
            poll_response = requests.get(get_url, headers=headers)
            poll_response.raise_for_status()
            poll_result = poll_response.json()
            
            status = poll_result.get("status")
            
            if status == "completed":
                # Extract Transcript
                transcript = poll_result.get('text', '')
                final_text = ""
                
                if diar and poll_result.get('utterances'):
                    # Process diarized utterances
                    for utt in poll_result['utterances']:
                        speaker = utt.get('speaker') # AssemblyAI returns "A", "B", etc.
                        text = utt.get('text', '').strip()
                        start_sec = utt.get('start', 0) / 1000 # Convert ms to seconds
                        
                        time_code = format_duration(start_sec)
                        final_text += f"[{time_code}] Sprecher {speaker}: {text}\n"
                else:
                    final_text = transcript
                    
                if not final_text.strip():
                    final_text = "❌ Transkript ist leer! Bitte Sprache/Modell prüfen."

                logs += f"\n✅ Status: COMPLETED nach {poll_count * 5} Sekunden."
                yield logs + "\n🎉 Fertig!", final_text.strip()
                return

            elif status == "error":
                raise Exception(f"AssemblyAI Error: {poll_result.get('error')}")

            if poll_count % 3 == 0:
                logs += f"\n⏳ Status: {status}... Polling {poll_count}..."
                yield logs, ""
                
    except requests.exceptions.RequestException as e:
        error_msg = f"AssemblyAI API Fehler: {e}"
        try:
            if 'response' in locals():
                error_msg += f"\nDetails: {response.text}"
        except: pass
        logger.exception(f"AssemblyAI Error: {e}")
        yield logs + f"\n🔥 Abbruch: {error_msg}", ""
    except Exception as e:
        logger.exception(f"AssemblyAI Error: {e}")
        yield logs + f"\n🔥 Abbruch: {str(e)}", ""

# ==========================================
# 1. CHAT LOGIK
# ==========================================

def run_chat(message, history, provider, model, temp, system_prompt, key, r_effort, r_tokens, user_state):
    # --- SECURITY CHECK ---
    if not user_state or not user_state.get("id"):
        yield "⛔ Nicht autorisiert. Bitte neu anmelden."
        return
    
    user_id = user_state["id"]
    # ----------------------

    import re
    import json
    
    try:
        client = get_client(provider, key)
        
        # 1. Build Messages
        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": str(system_prompt)})
            
        for msg in history:
            content = str(msg["content"]) if msg.get("content") else ""
            messages.append({"role": msg["role"], "content": content})
            
        messages.append({"role": "user", "content": str(message)})
        
        # 2. Base Parameters
        params = {
            "model": model,
            "messages": messages,
            "stream": True
        }
        
        # 3. Provider-Specific Reasoning Logic
        extra_body = {}
        
        if provider == "OpenRouter":
            reasoning_config = {}
            if r_tokens > 0:
                reasoning_config["max_tokens"] = int(r_tokens)
            elif r_effort != "default":
                reasoning_config["effort"] = r_effort
            
            if reasoning_config:
                extra_body["reasoning"] = reasoning_config
                extra_body["include_reasoning"] = True

        elif provider == "Scaleway":
            if r_effort and r_effort != "default":
                params["reasoning_effort"] = r_effort
            if r_tokens > 0:
                params["max_completion_tokens"] = int(r_tokens)
            params["temperature"] = float(temp)

        elif provider == "Mistral":
            if r_tokens > 0:
                params["max_tokens"] = int(r_tokens)
            params["temperature"] = float(temp)

        else:
            params["temperature"] = float(temp)
            if r_tokens > 0:
                params["max_tokens"] = int(r_tokens)

        if extra_body:
            params["extra_body"] = extra_body

        # 4. Execute
        stream = client.chat.completions.create(**params)
        
        full_response = ""
        reasoning_buffer = ""
        is_thinking = False
        
        for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                
                # --- A. Capture Explicit Reasoning ---
                new_reasoning = ""
                if hasattr(delta, 'reasoning') and delta.reasoning:
                    new_reasoning = delta.reasoning
                elif hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                    new_reasoning = delta.reasoning_content
                elif hasattr(delta, 'reasoning_details') and delta.reasoning_details:
                    if isinstance(delta.reasoning_details, str):
                        new_reasoning = delta.reasoning_details
                    elif isinstance(delta.reasoning_details, list):
                        for detail in delta.reasoning_details:
                            if detail.get('type') == 'reasoning.text':
                                new_reasoning += detail.get('text', '')

                if new_reasoning:
                    reasoning_buffer += new_reasoning
                    display_thought = f"<details open><summary>💭 Gedankengang ({len(reasoning_buffer)} zeichen)</summary>\n\n{reasoning_buffer}\n</details>\n\n"
                    yield display_thought + full_response
                    continue

                # --- B. Capture Content ---
                if delta.content:
                    val = delta.content
                    if "<think>" in val:
                        is_thinking = True
                        val = val.replace("<think>", "")
                    elif "</think>" in val:
                        is_thinking = False
                        val = val.replace("</think>", "")
                    
                    if is_thinking:
                        reasoning_buffer += val
                        display_thought = f"<details open><summary>💭 Gedankengang ({len(reasoning_buffer)} zeichen)</summary>\n\n{reasoning_buffer}\n</details>\n\n"
                        yield display_thought + full_response
                    else:
                        full_response += val
                        if reasoning_buffer:
                            display_thought = f"<details><summary>💭 Gedankengang ({len(reasoning_buffer)} zeichen)</summary>\n\n{reasoning_buffer}\n</details>\n\n"
                            yield display_thought + full_response
                        else:
                            yield full_response
                
    except Exception as e:
        # Catch specific errors to prevent connection drop
        err_str = str(e)
        if "unexpected tokens" in err_str: # Scaleway fix
             try:
                 match = re.search(r'\[.*\]', err_str)
                 if match:
                     leaked = json.loads(match.group(0))
                     reasoning_buffer += " ".join(leaked) + " [⚠️ Limit]"
                     yield f"<details open><summary>💭 Recovered</summary>{reasoning_buffer}</details>"
                     return
             except: pass
        
        logger.exception(f"Chat error with {provider}: {str(e)}")
        yield f"🔥 Fehler ({provider}): {str(e)}"
        
# ==========================================
# 2. VISION LOGIK
# ==========================================

def run_vision(image, prompt, provider, model, key, user_state):
    # --- SECURITY CHECK ---
    if not user_state or not user_state.get("id"):
        return "⛔ Nicht autorisiert. Bitte anmelden."
    # ----------------------
    if not image: return "❌ Bitte Bild hochladen."
    
    try:
        client = get_client(provider, key)
        b64_img = encode_image(image)
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
            ]
        }]
        
        # Standard OpenAI Vision call works for Poe now too
        response = client.chat.completions.create(
            model=model, messages=messages, max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"🔥 Vision Fehler: {str(e)}"

# ==========================================
# 3. TRANSKRIPTION
# ==========================================

def smart_format(utterances, show_tc, smart_merge, group_para, show_spk):
    """V6 Formatierungs-Logik für Gladia"""
    if not utterances: return ""
    out = ""; buf = ""; buf_start = 0; buf_spk = -1; last_end = 0

    for turn in utterances:
        txt = turn.get("text", "").strip()
        start = turn.get("start", 0); end = turn.get("end", 0); spk = turn.get("speaker", 0)

        if buf == "": buf_start = start; buf_spk = spk

        change = (spk != buf_spk)
        pause = start - last_end
        new_para = group_para and (pause > 1.5)

        if change or (new_para and buf):
            prefix = ""
            if show_tc: prefix += f"[{format_duration(buf_start)}] "
            if show_spk: prefix += f"Sprecher {buf_spk}: "
            out += f"{prefix}{buf}\n"
            if new_para: out += "\n"
            buf = txt; buf_start = start; buf_spk = spk
        else:
            if smart_merge: buf += " " + txt if buf else txt
            else:
                prefix = ""
                if show_tc: prefix += f"[{format_duration(start)}] "
                if show_spk: prefix += f"Sprecher {spk}: "
                out += f"{prefix}{txt}\n"; buf = ""
        last_end = end

    if buf:
        prefix = ""
        if show_tc: prefix += f"[{format_duration(buf_start)}] "
        if show_spk: prefix += f"Sprecher {buf_spk}: "
        out += f"{prefix}{buf}\n"
    return out

def run_mistral_transcription(audio_path, model, lang, api_key):
    """
    Handle Mistral transcription with chunking and state tracking.
    """
    import os
    
    client = openai.OpenAI(
        base_url="https://api.mistral.ai/v1", 
        api_key=api_key
    )
    
    # 1. Hash file to check/create progress
    file_hash = get_file_hash(audio_path)
    logger.info(f"Processing file hash: {file_hash}")
    
    # 2. Split file (Mistral limit ~15 mins)
    # We use 10 min chunks to be safe and allow for overhead
    logger.info("Splitting audio for Mistral (limit ~15m)...")
    chunks = split_audio(audio_path, chunk_length_ms=10 * 60 * 1000)
    
    if not chunks:
        yield "❌ Fehler beim Aufteilen der Audiodatei."
        return

    full_transcript = ""
    total_chunks = len(chunks)
    
    yield f"📂 Datei in {total_chunks} Teile zerlegt. Starte Upload & Transkription..."

    # 3. Process Chunks
    for i, chunk_path in enumerate(chunks):
        try:
            current_step = i + 1
            yield f"⏳ Verarbeite Teil {current_step}/{total_chunks}..."
            
            # Open file in binary mode
            with open(chunk_path, "rb") as f:
                # Mistral uses the standard OpenAI-compatible audio endpoint structure
                # Note: 'timestamp_granularities' is unique to Mistral but passed via extra_body or standard params depending on SDK version.
                # Since we use the OpenAI client for compatibility, we use standard params.
                
                params = {
                    "model": model or "voxtral-mini-latest",
                    "file": f,
                    "response_format": "json"
                }
                
                # Mistral specific: language param
                if lang and lang != "auto":
                    params["language"] = lang

                transcript = client.audio.transcriptions.create(**params)
                
                text_part = transcript.text
                full_transcript += text_part + " "
                
                logger.info(f"Chunk {current_step} complete.")
                yield f"✅ Teil {current_step}/{total_chunks} fertig."

        except Exception as e:
            logger.exception(f"Error processing chunk {i}: {e}")
            yield f"🔥 Fehler bei Teil {current_step}: {str(e)}"
            # We don't abort, try next chunk? Or abort? Usually better to abort to avoid broken texts.
            return 
        finally:
            # Cleanup chunk
            if os.path.exists(chunk_path):
                os.remove(chunk_path)

    yield full_transcript.strip()

def run_chunked_api_transcription(client, model, chunk_paths, lang, prompt, temp, job_id=None):
    """
    Iterates through chunks with State Tracking.
    """
    import time
    
    # Load manifest if resuming
    start_index = 0
    full_transcript_parts = [""] * len(chunk_paths)
    
    if job_id:
        try:
            with open(os.path.join(JOB_STATE_DIR, f"{job_id}.json"), "r") as f:
                manifest = json.load(f)
                # Find first non-done chunk
                for i, c in enumerate(manifest["chunks"]):
                    if c["status"] == "done":
                        full_transcript_parts[i] = manifest["transcript_parts"][i]
                    else:
                        if start_index == 0 and i > 0: start_index = i
        except: pass

    total = len(chunk_paths)
    MAX_RETRIES = 3
    BASE_DELAY = 10 
    
    for i in range(start_index, total):
        chunk_path = chunk_paths[i]
        step = i + 1
        chunk_success = False
        retry_count = 0
        
        # If resuming, check if file still exists (tmp files might be gone if server restarted)
        if not os.path.exists(chunk_path):
            yield f"❌ Fehler: Chunk-Datei {step} nicht gefunden. Job kann nicht fortgesetzt werden."
            return

        while not chunk_success and retry_count < MAX_RETRIES:
            try:
                yield f"⏳ Verarbeite Teil {step}/{total}..."
                
                with open(chunk_path, "rb") as f:
                    args = {
                        "model": model,
                        "file": f,
                        "response_format": "json"
                    }
                    if lang and lang != "auto": args["language"] = lang
                    if temp is not None: args["temperature"] = float(temp)
                    
                    # Context handling
                    if i == 0 and prompt:
                        args["prompt"] = prompt
                    elif i > 0:
                        # Construct context from previous successful parts
                        prev_text = "".join(full_transcript_parts[:i])
                        if prev_text: args["prompt"] = prev_text[-220:]

                    resp = client.audio.transcriptions.create(**args)
                    text_part = resp.text if hasattr(resp, 'text') else str(resp)
                    
                    # Update State
                    full_transcript_parts[i] = text_part + " "
                    if job_id:
                        update_job_chunk_status(job_id, i, "done", text_part + " ")
                    
                    yield f"✅ Teil {step}/{total} erledigt."
                    chunk_success = True
            
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate_limit" in error_str.lower():
                    retry_count += 1
                    wait = BASE_DELAY * (2 ** retry_count)
                    yield f"⚠️ Rate Limit (429). Warte {wait}s..."
                    time.sleep(wait)
                else:
                    logger.error(f"Critical error in chunk {step}: {e}")
                    full_transcript_parts[i] = f" [Fehler in Teil {step}] "
                    if job_id: update_job_chunk_status(job_id, i, "failed")
                    break 
        
        if not chunk_success:
            full_transcript_parts[i] = f" [Timeout Teil {step}] "
            if job_id: update_job_chunk_status(job_id, i, "failed")

    yield "".join(full_transcript_parts).strip()
        
def run_transcription(audio, provider, model, lang, whisper_temp, whisper_prompt, diar, trans, target, key, chunk_opt=True, chunk_len=10):
    """
    Unified transcription router.
    - Gladia: Uses Native V2 API (Upload -> URL -> Poll).
    - Deepgram/AssemblyAI: Uses Native Async/Sync API.
    - Others: Uses Local Chunking (Optional) -> OpenAI API.
    """
    logger.info("=" * 50)
    logger.info(f"TRANSCRIPTION START: {provider} | Model: {model} | File: {audio}")

    # --- 1. Validation ---
    if not audio:
        yield "❌ Keine Datei.", "", ""
        return

    if not os.path.exists(audio):
        yield f"❌ Datei nicht gefunden: {audio}", "", ""
        return

    try:
        file_size = os.path.getsize(audio)
        if file_size == 0:
            yield "❌ Datei ist leer (0 Bytes).", "", ""
            return
    except Exception as e:
        logger.error(f"File check error: {e}")

    # --- 2. BRANCH A: GLADIA (Native Long-File Support) ---
    if provider == "Gladia":
        logger.info("Using Gladia Native V2 Flow")
        
        api_key = key if key else API_KEYS.get("GLADIA", "")
        if not api_key:
            yield "❌ Kein Gladia Key gefunden.", "", ""
            return

        logs = "🚀 Start Gladia Upload..."
        yield logs, "", ""

        try:
            # A. Upload
            headers = {"x-gladia-key": api_key, "accept": "application/json"}
            fname = os.path.basename(audio)
            
            with open(audio, 'rb') as f:
                r = requests.post(
                    f"{GLADIA_CONFIG['url']}/upload",
                    headers=headers,
                    files={'audio': (fname, f, 'audio/wav')},
                    timeout=600 
                )
            
            if r.status_code != 200:
                raise Exception(f"Upload failed: {r.text}")
            
            upload_url = r.json().get("audio_url")
            logs += "\n✅ Upload fertig. Starte Job..."
            yield logs, "", ""

            # B. Job Config
            vocab_list = [{"value": w} for w in GLADIA_CONFIG.get('vocab', [])]
            
            payload = {
                "audio_url": upload_url,
                "language_config": {
                    "code_switching": (lang == "auto"),
                    "languages": [] if lang == "auto" else [lang]
                },
                "diarization": diar,
                "diarization_config": {"min_speakers": 1, "max_speakers": 10} if diar else None,
                "custom_vocabulary": True,
                "custom_vocabulary_config": {"vocabulary": vocab_list},
                "translation": trans,
                "translation_config": {
                    "target_languages": [target],
                    "model": "base",
                    "match_original_utterances": True
                } if trans else None
            }

            # C. Start Job
            r = requests.post(f"{GLADIA_CONFIG['url']}/pre-recorded", headers=headers, json=payload)
            if r.status_code != 201:
                raise Exception(f"Job start failed: {r.text}")
            
            result_url = r.json().get("result_url")
            
            # D. Polling
            poll_count = 0
            start_t = time.time()
            
            while True:
                time.sleep(5)
                poll_count += 1
                elapsed = time.time() - start_t
                
                if poll_count % 2 == 0: 
                    yield f"{logs}\n⏳ Verarbeite... ({format_duration(elapsed)})", "", ""

                if elapsed > 3600:
                    raise Exception("Timeout nach 60 Minuten")

                r = requests.get(result_url, headers=headers)
                if r.status_code != 200: continue
                
                data = r.json()
                status = data.get("status")
                
                if status == "done":
                    break
                elif status == "error":
                    raise Exception(f"Gladia Error: {json.dumps(data)}")

            # E. Process Result
            res = data.get("result", {})
            
            transcription = res.get("transcription", {})
            utterances = transcription.get("utterances", [])
            final_text = smart_format(utterances, True, True, True, diar)
            if not final_text: 
                final_text = transcription.get("full_transcript", "")

            trans_text = ""
            if trans:
                translation = res.get("translation", {})
                t_res = translation.get("results", [])
                if t_res:
                    t_utt = t_res[0].get("utterances", [])
                    trans_text = smart_format(t_utt, True, True, True, diar)
            
            yield f"{logs}\n🎉 Fertig!", final_text, trans_text

        except Exception as e:
            logger.exception(f"Gladia Error: {e}")
            yield f"🔥 Fehler: {str(e)}", "", ""
            
    # --- 2. BRANCH B: DEEPGRAM (Native Single-Shot EU) ---
    elif provider == "Deepgram":
        logger.info("Using Deepgram Native Sync Flow (EU)")
        full_text = ""
        # model, lang, diar aus den UI-Optionen verwenden
        for log, text in run_deepgram_transcription(audio, model, lang, diar, key):
            if text: full_text = text
            yield log, full_text, "(Keine Übersetzung verfügbar)"
        return # Ende der Funktion

    # --- 2. BRANCH C: ASSEMBLYAI (Native Async EU) ---
    elif provider == "AssemblyAI":
        logger.info("Using AssemblyAI Native Async Flow (EU)")
        full_text = ""
        # model, lang, diar aus den UI-Optionen verwenden
        for log, text in run_assemblyai_transcription(audio, model, lang, diar, key):
            if text: full_text = text
            yield log, full_text, "(Keine Übersetzung verfügbar)"
        return # Ende der Funktion

    # --- 3. BRANCH D: GENERIC CHUNKING (Mistral, Scaleway, Groq) ---
    else:
        logger.info(f"Using Generic Provider: {provider}")
        
        try:
            client = get_client(provider, key)
        except Exception as e:
            yield f"🔥 Client Fehler: {str(e)}", "", ""
            return

        if not model:
            conf = PROVIDERS.get(provider, {})
            model = conf.get("audio_models", ["whisper-large-v3"])[0]

        logs = f"🚀 Starte {provider} ({model})..."
        yield logs, "", ""

        chunks = []
        chunk_dir = None # Default if not splitting

        try:
            # --- OPTIONAL CHUNKING LOGIC ---
            if chunk_opt:
                yield f"{logs}\n✂️ Teile Audio (alle {chunk_len} Min)...", "", ""
                chunks, chunk_dir = split_audio_into_chunks(audio, chunk_minutes=int(chunk_len))
                
                if not chunks:
                    yield "❌ Fehler beim Aufteilen der Datei.", "", ""
                    return
                logs += f"\n📂 {len(chunks)} Teile erstellt."
            else:
                # No chunking: Pass original file as single item list
                yield f"{logs}\n⚠️ Chunking deaktiviert. Sende Originaldatei...", "", ""
                if os.path.getsize(audio) > 25 * 1024 * 1024:
                    logger.warning("File > 25MB and chunking disabled. API might fail.")
                    logs += "\n⚠️ WARNUNG: Datei > 25MB. Upload könnte fehlschlagen."
                chunks = [audio]

            # --- CREATE JOB MANIFEST (For Resume Capability) ---
            job_id = int(time.time())
            create_job_manifest(job_id, audio, provider, model, chunks, lang, whisper_prompt, whisper_temp)
            logs += f"\n🆔 Job-ID: {job_id} (Für Resume gespeichert)"
            yield logs, "", ""

            # D. Run Sequential Processing
            full_text = ""
            
            # Pass job_id to allow state saving during processing
            transcriber = run_chunked_api_transcription(
                client, model, chunks, lang, whisper_prompt, whisper_temp, job_id=job_id
            )

            for update in transcriber:
                if len(update) < 300 and (update.startswith("⏳") or update.startswith("✅") or update.startswith("⚠️")):
                    logs += f"\n{update}"
                    yield logs, full_text, ""
                else:
                    full_text = update

            yield logs + "\n🎉 Fertig!", full_text, "(Keine Übersetzung verfügbar)"

        except Exception as e:
            logger.exception(f"Provider Error: {e}")
            yield logs + f"\n🔥 Abbruch: {str(e)}", "", ""
        
        finally:
            # E. Cleanup (Only if we actually created chunks)
            if chunk_dir:
                cleanup_chunks(chunk_dir)

                
def run_and_save_transcription(audio, provider, model, lang, w_temp, w_prompt, diar, trans, target, key, chunk_opt, chunk_len, dg_lang, dg_diar, aa_lang, aa_diar, user_state):
    # --- SECURITY CHECK ---
    if not user_state or not user_state.get("id"):
        yield "⛔ Nicht autorisiert. Bitte anmelden.", "", ""
        return
    user_id = user_state["id"]
    # ----------------------
    
    # 1. Prepare parameters
    final_lang = lang
    final_diar = diar
    final_trans = trans
    final_target = target

    if provider == "Deepgram":
        final_lang = dg_lang
        final_diar = dg_diar
        final_trans = False
        final_target = None
    elif provider == "AssemblyAI":
        final_lang = aa_lang
        final_diar = aa_diar
        final_trans = False
        final_target = None
    
    try:
        logger.info(f"Starting transcription: provider={provider}, model={model}, audio={audio}")

        # Basic File Validation
        if not audio:
            yield "❌ Keine Audiodatei hochgeladen.", "", ""
            return
        if not os.path.exists(audio):
            yield f"❌ Datei nicht gefunden: {audio}", "", ""
            return
        if os.path.getsize(audio) == 0:
            yield "❌ Audiodatei ist leer.", "", ""
            return

        # 2. Run transcription
        result = None
        for result in run_transcription(
            audio, provider, model, 
            final_lang, w_temp, w_prompt, 
            final_diar, final_trans, final_target, 
            key, chunk_opt, chunk_len
        ):
            yield result

        # 3. Save to database after completion
        if user_id and result and len(result) > 1 and result[1]:
            logger.info("Auto-saving transcription to database...")
            filename = os.path.basename(audio) if audio else None
            save_lang = final_lang

            try:
                trans_id = save_transcription(
                    user_id=user_id,
                    provider=provider,
                    model=model or "N/A",
                    original=result[1],
                    translated=result[2] if len(result) > 2 else None,
                    language=save_lang,
                    filename=filename
                )
                updated_log = result[0] + f"\n\n💾 Automatisch gespeichert (ID: {trans_id})"
                yield (updated_log, result[1], result[2] if len(result) > 2 else "")

            except Exception as save_error:
                updated_log = result[0] + f"\n\n⚠️ Speichern fehlgeschlagen: {str(save_error)}"
                yield (updated_log, result[1], result[2] if len(result) > 2 else "")

    except Exception as e:
        logger.exception(f"Critical error in transcription: {str(e)}")
        yield f"🔥 Kritischer Fehler: {str(e)}", "", ""

# ==========================================
# 4. BILDGENERIERUNG 
# ==========================================

def run_image_gen(prompt, provider, model, width, height, steps, key, user_state):
    # --- SECURITY CHECK ---
    if not user_state or not user_state.get("id"):
        return None, "⛔ Nicht autorisiert. Bitte anmelden."
    # ----------------------
    
    import requests 
    import base64
    import tempfile
    import re
    import time
    
    try:
        client = get_client(provider, key)
        
        # --- SPECIAL CASE: POE (Chat-to-Image) ---
        if provider == "Poe":
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=False
            )
            response_text = response.choices[0].message.content
            
            # Extract Image URL
            match = re.search(r'!\[.*?\]\((https?://.*?)\)', response_text)
            if not match:
                match = re.search(r'\[.*?\]\((https?://.*?)\)', response_text)
            if not match:
                match = re.search(r'(https?://[^\s]+)', response_text)
            
            if not match:
                return None, f"❌ Kein Bild gefunden. Antwort: {response_text[:200]}"
            
            image_url = match.group(1).rstrip(".,;)")
            
            # Download with Retry
            for attempt in range(1, 4):
                try:
                    r = requests.get(image_url, timeout=15)
                    if r.status_code == 200:
                        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                        tfile.write(r.content)
                        tfile.close()
                        return tfile.name, "✅ Erfolg (Poe)"
                    elif r.status_code in [403, 404]:
                        time.sleep(2)
                        continue
                    else:
                        return None, f"❌ Download Fehler: {r.status_code}"
                except Exception as e:
                    logger.warning(f"Download attempt {attempt} failed: {e}")
                    time.sleep(2)
            
            return None, "❌ Bild konnte nach 3 Versuchen nicht geladen werden."
        
        # --- SPECIAL CASE: OPENROUTER (Chat Completions with Modalities) ---
        if provider == "OpenRouter":
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                modalities=["image", "text"],
                stream=False
            )
            
            message = response.choices[0].message
            
            if not hasattr(message, 'images') or not message.images:
                content = getattr(message, 'content', '')
                return None, f"❌ Keine Bilder generiert. Antwort: {content[:200]}"
            
            image_data_url = message.images[0].image_url.url
            
            if not image_data_url.startswith('data:image/'):
                return None, f"❌ Ungültiges Bildformat: {image_data_url[:50]}"
            
            base64_data = image_data_url.split('base64,', 1)[1]
            img_data = base64.b64decode(base64_data)
            
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            tfile.write(img_data)
            tfile.close()
            return tfile.name, "✅ Erfolg (OpenRouter)"
        
        # --- SPECIAL CASE: SCALEWAY ---
        if provider == "Scaleway":
            # Scaleway doesn't support standard image generation endpoint
            return None, "❌ Scaleway: Bildgenerierung derzeit nicht unterstützt. Bitte Nebius verwenden."
        
        # --- STANDARD PROVIDER: NEBIUS ---
        params = {
            "model": model,
            "prompt": prompt,
            "response_format": "b64_json",
        }
        
        if provider == "Nebius":
            params["extra_body"] = {
                "width": width, 
                "height": height, 
                "num_inference_steps": steps,
                "response_extension": "jpg"
            }
            
        response = client.images.generate(**params)
        
        img_data = base64.b64decode(response.data[0].b64_json)
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        tfile.write(img_data)
        tfile.close()
        return tfile.name, "✅ Erfolg"
        
    except Exception as e:
        logger.exception(f"Image Gen Error: {str(e)}")
        return None, f"🔥 Fehler: {str(e)}"
    
# ==========================================
# 📋 PREDEFINED PROMPT TEMPLATES
# ==========================================

TRANSCRIPT_PROMPTS = {
    "Veranstaltungsrückblick": """Schreibe auf der Grundlage dieses automatisch erstellten Transkripts einen professionellen Veranstaltungsrückblick für unsere Website.

Berücksichtige dabei:
- Hauptthemen und Kernaussagen
- Wichtige Diskussionspunkte
- Atmosphäre und Teilnehmerfeedback (falls erwähnt)
- Schlussfolgerungen und Ausblicke

Transkript:
{transcript}

Zusätzliche Hinweise:
{notes}""",

    "Zusammenfassung": """Erstelle eine prägnante Zusammenfassung dieses Transkripts.

Gliedere nach:
1. Hauptthema und Zielsetzung
2. Kernaussagen (3-5 Punkte)
3. Wichtigste Erkenntnisse
4. Offene Fragen oder Ausblick

Transkript:
{transcript}

Zusätzliche Hinweise:
{notes}""",

    "Pressemitteilung": """Verfasse eine Pressemitteilung basierend auf diesem Veranstaltungstranskript.

Die Pressemitteilung sollte:
- Einen aufmerksamkeitsstarken Titel haben
- Die 5 W-Fragen beantworten (Wer, Was, Wann, Wo, Warum)
- Zitate von Rednern einbinden
- Maximal 300 Wörter umfassen

Transkript:
{transcript}

Zusätzliche Hinweise:
{notes}""",

    "Social Media Posts": """Erstelle 3 verschiedene Social Media Posts basierend auf diesem Transkript:

1. LinkedIn Post (professionell, 150 Wörter)
2. Twitter/X Thread (3-4 Tweets)
3. Instagram Caption (ansprechend, mit Emojis)

Transkript:
{transcript}

Zusätzliche Hinweise:
{notes}""",

    "Protokoll": """Erstelle ein formelles Protokoll dieser Veranstaltung basierend auf dem Transkript.

Struktur:
1. Datum, Ort, Teilnehmer
2. Tagesordnung/Themen
3. Diskussionspunkte und Beschlüsse
4. Offene Aufgaben und Verantwortliche
5. Nächste Schritte

Transkript:
{transcript}

Zusätzliche Hinweise:
{notes}""",

    "FAQ Generierung": """Analysiere dieses Transkript und erstelle daraus eine FAQ-Liste mit 8-10 häufig gestellten Fragen und detaillierten Antworten.

Die FAQs sollten:
- Klar und verständlich formuliert sein
- Die wichtigsten Informationen abdecken
- Für Website-Besucher nützlich sein

Transkript:
{transcript}

Zusätzliche Hinweise:
{notes}""",

    "Zitate-Sammlung": """Extrahiere die wichtigsten und aussagekräftigsten Zitate aus diesem Transkript.

Für jedes Zitat gib an:
- Sprecher (falls bekannt)
- Kontext
- Warum dieses Zitat bedeutsam ist

Transkript:
{transcript}

Zusätzliche Hinweise:
{notes}""",

    "Blogbeitrag": """Schreibe einen ansprechenden Blogbeitrag basierend auf diesem Veranstaltungstranskript.

Der Blogbeitrag sollte:
- Einen einladenden Einstieg haben
- Die Hauptinhalte verständlich aufbereiten
- Persönliche Eindrücke einbinden
- Mit einem Call-to-Action enden
- Ca. 500-800 Wörter umfassen

Transkript:
{transcript}

Zusätzliche Hinweise:
{notes}""",

    "Eigener Prompt": """{transcript}

{notes}"""
}

# ==========================================
# 📱 PWA & CSS CONFIGURATION
# ==========================================
PWA_HEAD = """
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<meta name="theme-color" content="#1976d2">
<meta name="mobile-web-app-capable" content="yes">
<link rel="manifest" href="/manifest.json" crossorigin="use-credentials">
<link rel="icon" type="image/png" sizes="192x192" href="/static/icon-192.png">
<link rel="stylesheet" href="/static/custom.css">
<script src="/static/pwa.js" defer></script>

<style>
/* 🧠 Reasoning/Thinking Block Styling */
.message-wrap details {
    background-color: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 8px 12px;
    margin-bottom: 12px;
    font-size: 0.9em;
}

.message-wrap summary {
    cursor: pointer;
    font-weight: 600;
    color: #6b7280;
    user-select: none;
    display: flex;
    align-items: center;
    gap: 6px;
    outline: none;
}

.message-wrap details[open] summary {
    margin-bottom: 8px;
    padding-bottom: 8px;
    border-bottom: 1px solid #e5e7eb;
}

body.dark .message-wrap details {
    background-color: #1f2937;
    border-color: #374151;
}
body.dark .message-wrap summary {
    color: #9ca3af;
}

/* 🚫 HIDE FOOTER (Built with Gradio) */
footer {
    display: none !important;
}
</style>
"""

# ==========================================
# UI HELPERS
# ==========================================

# --- CHAT UI UPDATE ---
def update_c_ui(prov, force_all=False, user_state=None):
    p_data = PROVIDERS.get(prov, {})
    badge = p_data.get("badge", "")
    api_key = API_KEYS.get(p_data.get("key_name"))
    
    # 1. Fetch
    models, error = fetch_available_models(prov, api_key)
    choices = []
    
    if models:
        # Filter for Chat types if Poe
        if prov == "Poe":
            choices = [(m["name"], m["id"]) for m in models if "Chat" in m.get("type", "Chat")]
        else:
            choices = [(m["name"], m["id"]) for m in models]
    
    # Fallback
    if not choices:
        choices = [(m, m) for m in p_data.get("chat_models", [])]

    # 2. Filter via Helper
    final_choices = get_filtered_model_choices(prov, choices, force_all, user_state)
    
    default_val = final_choices[0][1] if final_choices else None
    return gr.update(choices=final_choices, value=default_val), badge

# --- VISION UI UPDATE ---
def update_v_ui(prov, force_all=False, user_state=None):
    # 1. Handle Badge Styling (Dark background, White text)
    raw_html = get_compliance_html(prov)
    styled_badge = f"""
    <div style="
        background-color: #374151; 
        color: #ffffff !important; 
        padding: 0 12px; 
        border-radius: 8px; 
        border: 1px solid #4b5563; 
        display: flex; 
        align-items: center; 
        justify-content: center;
        height: 42px; 
        font-size: 0.9em;
        white-space: nowrap;
        overflow: hidden;
    ">
        {raw_html}
    </div>
    """

    p_data = PROVIDERS.get(prov, {})
    api_key = API_KEYS.get(p_data.get("key_name"))
    
    # 2. Fetch Models
    models, error = fetch_available_models(prov, api_key)
    choices = []
    
    if models and prov == "Poe":
        choices = [(m["name"], m["id"]) for m in models if "Vision" in m.get("type", "")]
    elif not models or prov != "Poe":
        choices = [(m, m) for m in p_data.get("vision_models", [])]

    # 3. Filter via Helper
    final_choices = get_filtered_model_choices(prov, choices, force_all, user_state)
    default_val = final_choices[0][1] if final_choices else None
    
    # Return Badge and Model Update
    return styled_badge, gr.update(choices=final_choices, value=default_val)

# --- IMAGE UI UPDATE ---
def update_g_ui(prov, force_all=False, user_state=None):
    # 1. Handle Badge Styling
    raw_html = get_compliance_html(prov)
    styled_badge = f"""
    <div style="
        background-color: #374151; 
        color: #ffffff !important; 
        padding: 0 12px; 
        border-radius: 8px; 
        border: 1px solid #4b5563; 
        display: flex; 
        align-items: center; 
        justify-content: center;
        height: 42px; 
        font-size: 0.9em;
        white-space: nowrap;
        overflow: hidden;
    ">
        {raw_html}
    </div>
    """

    p_data = PROVIDERS.get(prov, {})
    api_key = API_KEYS.get(p_data.get("key_name"))
    
    # 2. Fetch Models
    models, error = fetch_available_models(prov, api_key)
    choices = []
    
    if models and prov == "Poe":
        choices = [(f"{m['name']}", m['id']) for m in models if m.get('type') in ["Bild-Gen", "Video-Gen"]]
    
    if not choices:
        choices = [(m, m) for m in p_data.get("image_models", [])]
        
    # 3. Filter via Helper
    final_choices = get_filtered_model_choices(prov, choices, force_all, user_state)
    default_val = final_choices[0][1] if final_choices else None
    
    return styled_badge, gr.update(choices=final_choices, value=default_val)

# --- TRANSCRIPTION UI UPDATE ---
# Signature: (badge, t_model_update, gladia_vis, whisper_vis, deepgram_vis, assemblyai_vis)
def update_t_ui(prov, force_all=False):
    # 1. Handle Badge Formatting
    raw_html = get_compliance_html(prov)
    
    # CSS: Multi-line support, dark background, centered
    styled_badge = f"""
    <div style="
        background-color: #374151; 
        color: #ffffff !important; 
        padding: 8px 12px; 
        border-radius: 8px; 
        border: 1px solid #4b5563; 
        margin-top: 10px; 
        display: flex; 
        align-items: center; 
        justify-content: center;
        text-align: center;
        font-size: 0.9em;
    ">
        {raw_html}
    </div>
    """

    # 2. Handle Models
    is_whisper = prov in ["Mistral", "Scaleway", "Groq"]
    show_model_dropdown = prov in ["Deepgram", "AssemblyAI"] or is_whisper
    
    choices = []
    default_val = None

    if show_model_dropdown:
        p_data = PROVIDERS.get(prov, {})
        raw_list = p_data.get("audio_models", [])
        choices = [(m, m) for m in raw_list]
        if choices:
            default_val = choices[0][1]

    # Return updates
    return (
        styled_badge,
        gr.update(visible=show_model_dropdown, choices=choices, value=default_val),
        gr.update(visible=is_whisper) 
    )
    
# --- Chat functions ---
def user_msg(msg, hist):
    return "", hist + [{"role": "user", "content": msg}]

def bot_msg(hist, prov, mod, temp, sys, key, r_effort, r_tokens, user_state):
    """Execute chat passing user state for security"""
    if not hist: yield hist; return

    if hist[-1]["role"] != "user": yield hist; return

    last_user_msg = hist[-1]["content"]
    hist.append({"role": "assistant", "content": ""})

    # Prepare Context
    raw_context = hist[:-2]
    clean_context = []
    for m in raw_context:
        role = m["role"]
        if role in ["bot", "model"]: role = "assistant"
        clean_context.append({"role": role, "content": m["content"]})

    try:
        # Pass user_state to run_chat
        generator = run_chat(last_user_msg, clean_context, prov, mod, temp, sys, key, r_effort, r_tokens, user_state)
        
        for chunk in generator:
            hist[-1]["content"] = chunk
            yield hist
            
    except Exception as e:
        hist[-1]["content"] = f"🔥 Wrapper Fehler: {str(e)}"
        yield hist

def save_chat(hist, prov, mod, user_state):
    """Save current chat to database"""
    try:
        if not user_state or not user_state.get("id"):
            logger.warning("Save chat failed: User not logged in")
            return "❌ Bitte anmelden"

        user_id = user_state["id"]
        logger.info(f"Attempting to save chat for user {user_id}")

        if not hist or len(hist) == 0:
            logger.warning("Save chat failed: Empty chat history")
            return "❌ Kein Chat zum Speichern"

        # Generate title
        first_content = hist[0].get("content", "") if isinstance(hist[0], dict) else str(hist[0])
        title = first_content[:50] + "..." if len(first_content) > 50 else first_content

        logger.info(f"Saving chat with title: {title}")
        chat_id = save_chat_history(user_id, prov, mod, hist, title)
        logger.info(f"Chat saved successfully with ID: {chat_id}")

        return f"✅ Chat gespeichert (ID: {chat_id})"

    except Exception as e:
        logger.exception(f"Error saving chat: {str(e)}")
        return f"🔥 Fehler beim Speichern: {str(e)}"
    
def load_chat_list(user_state):
    """Load user's chat history (legacy format)"""
    if not user_state or not user_state.get("id"):
        return [["Bitte anmelden", "", "", ""]]

    chats = get_user_chat_history(user_state["id"])
    data = []
    for chat in chats:
        data.append([
            chat.id,
            chat.timestamp.strftime("%Y-%m-%d %H:%M"),
            chat.title or "Ohne Titel",
            chat.model
        ])
    return data if data else [["Keine Chats vorhanden", "", "", ""]]

def load_chat_list_with_state(user_state=None):
    """Load user's chat history returning both Display Data and State Data"""
    if not user_state or not user_state.get("id"):
        return [["Bitte anmelden", "", "", ""]], []

    chats = get_user_chat_history(user_state["id"])
    
    # Clean data for State (ID, Date, Title, Model)
    state_data = [[c.id, c.timestamp.strftime("%Y-%m-%d %H:%M"), c.title or "Ohne Titel", c.model] for c in chats]
    
    if not state_data:
        return [["Keine Chats vorhanden", "", "", ""]], []
        
    return state_data, state_data

def select_chat_row(evt: gr.SelectData, state_data):
    """Smart Selection for Chat List"""
    try:
        if not state_data:
            return None
            
        row_idx = evt.index[0]
        if row_idx < len(state_data):
            real_row = state_data[row_idx]
            # ID is in column 0
            return int(real_row[0])
    except Exception as e:
        logger.error(f"Selection error: {e}")
    return None

# ==========================================
# 📄 UNIVERSAL CONTENT EXTRACTOR (Enhanced & Debuggable)
# ==========================================
import mimetypes
import chardet
import shutil
import subprocess
import traceback  # NEW: For detailed error logging
from io import StringIO

# Try imports and log immediately what is missing
try:
    import pytesseract
    from pdf2image import convert_from_path
    from docx import Document
    import fitz  # PyMuPDF
    import pandas as pd
    from pypdf import PdfReader # NEW: Pure Python fallback
except ImportError as e:
    logger.warning(f"⚠️ Missing extraction library: {e}. Functionality will be limited.")

class UniversalExtractor:
    @staticmethod
    def extract(filepath):
        """Determines file type and extracts text content using robust methods + fallbacks"""
        if not filepath or not os.path.exists(filepath):
            return "❌ Datei nicht gefunden."
            
        mime_type, _ = mimetypes.guess_type(filepath)
        ext = os.path.splitext(filepath)[1].lower()
        filename = os.path.basename(filepath)
        
        logger.info(f"🔍 Extracting: {filename} (Type: {ext})")
        
        try:
            # 1. Images (OCR)
            if (mime_type and mime_type.startswith('image/')) or ext in ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff']:
                return UniversalExtractor._extract_image(filepath)
            
            # 2. PDF (Chain: PyMuPDF -> PyPDF -> CLI -> OCR)
            elif ext == '.pdf':
                return UniversalExtractor._extract_pdf(filepath)
            
            # 3. Modern Word (.docx)
            elif ext == '.docx':
                try: return UniversalExtractor._extract_docx(filepath)
                except Exception as e:
                    logger.warning(f"Docx-Lib failed: {e}, trying CLI...")
                    return UniversalExtractor._extract_with_cli_tool(filepath)

            # 4. Excel (.xls, .xlsx, .csv)
            elif ext in ['.xls', '.xlsx', '.csv']:
                return UniversalExtractor._extract_excel(filepath)

            # 5. Ebooks & Legacy Docs (.epub, .mobi, .doc, .odt, .rtf)
            elif ext in ['.epub', '.mobi', '.azw', '.azw3', '.fb2', '.doc', '.odt', '.rtf', '.html']:
                return UniversalExtractor._extract_with_cli_tool(filepath)
                
            # 6. Text/Code
            else:
                return UniversalExtractor._extract_plain_text(filepath)
                
        except Exception as e:
            # LOG THE FULL ERROR TRACE
            logger.error(f"🔥 Critical Extraction Error for {filename}: {str(e)}")
            logger.error(traceback.format_exc()) 
            return f"[Systemfehler beim Lesen von {filename}: {str(e)}]"

    # --- INTERNAL HANDLERS ---

    @staticmethod
    def _extract_with_cli_tool(input_path):
        """Uses ebook-converter, calibre, or pandoc"""
        tool = shutil.which("ebook-converter") or shutil.which("ebook-convert") or shutil.which("pandoc")
        
        if not tool:
            logger.warning("No CLI converter tool found (ebook-converter/calibre/pandoc)")
            return "[Kein Konverter installiert]"

        try:
            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
                out_path = tmp.name

            cmd = [tool, input_path, out_path]
            if "pandoc" in tool:
                cmd = [tool, input_path, "-t", "plain", "-o", out_path]

            logger.info(f"Running CLI: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                logger.error(f"CLI stderr: {result.stderr}")
                # Don't return error yet, file might still exist
                
            if os.path.exists(out_path):
                with open(out_path, 'r', encoding='utf-8', errors='replace') as f:
                    text = f.read()
                os.remove(out_path)
                if text.strip(): return text

            return f"[CLI Konvertierung fehlgeschlagen: {result.stderr}]"

        except Exception as e:
            logger.error(f"CLI Tool Error: {e}")
            return f"[CLI Fehler: {str(e)}]"

    @staticmethod
    def _extract_image(path):
        try:
            # Check if tesseract is installed
            if not shutil.which("tesseract"):
                return "[Fehler: 'tesseract' ist nicht installiert. sudo apt install tesseract-ocr]"
                
            text = pytesseract.image_to_string(Image.open(path), lang='deu+eng')
            return f"[OCR-Ergebnis]:\n{text}" if text.strip() else "[Kein Text im Bild erkannt]"
        except Exception as e:
            logger.error(f"OCR Error: {e}")
            return f"[OCR Fehler: {str(e)}]"

    @staticmethod
    def _extract_pdf(path):
        """Chain of Responsibility for PDFs"""
        errors = []
        
        # Method A: PyMuPDF (Fastest & Best Layout)
        try:
            text = ""
            with fitz.open(path) as doc:
                for page in doc: text += page.get_text() + "\n"
            if len(text.strip()) > 50: 
                return text
            else:
                errors.append("PyMuPDF: Text too short (Scanned?)")
        except Exception as e:
            errors.append(f"PyMuPDF Error: {e}")
            logger.warning(f"PyMuPDF failed: {e}")

        # Method B: PyPDF (Pure Python Fallback)
        try:
            reader = PdfReader(path)
            text = ""
            for page in reader.pages:
                text += (page.extract_text() or "") + "\n"
            if len(text.strip()) > 50:
                return f"[Extracted via PyPDF]\n{text}"
            else:
                errors.append("PyPDF: Text too short")
        except Exception as e:
            errors.append(f"PyPDF Error: {e}")

        # Method C: CLI Tool (ebook-converter)
        try:
            text = UniversalExtractor._extract_with_cli_tool(path)
            if "Fehler" not in text and len(text) > 50:
                return f"[Extracted via CLI]\n{text}"
        except Exception as e:
             errors.append(f"CLI Error: {e}")

        # Method D: OCR (Last Resort for Scans)
        logger.info("All PDF text methods failed. Attempting OCR...")
        try:
            return UniversalExtractor._extract_scanned_pdf(path)
        except Exception as e:
            errors.append(f"OCR Error: {e}")
            
        logger.error(f"PDF Extraction completely failed. Log: {errors}")
        return f"[PDF konnte nicht gelesen werden. Errors: {'; '.join(errors)}]"

    @staticmethod
    def _extract_scanned_pdf(path):
        try:
            text = "[HINWEIS: OCR Scan Modus]\n"
            # Only do first 5 pages to prevent server timeout/hang
            images = convert_from_path(path, first_page=1, last_page=5) 
            for i, img in enumerate(images):
                text += f"\n--- Seite {i+1} ---\n"
                text += pytesseract.image_to_string(img, lang='deu+eng')
            return text
        except Exception as e:
            logger.error(f"Scan extraction error: {e}")
            raise e

    @staticmethod
    def _extract_docx(path):
        doc = Document(path)
        return "\n".join([p.text for p in doc.paragraphs])

    @staticmethod
    def _extract_excel(path):
        try:
            if path.endswith('.csv'): df = pd.read_csv(path)
            else: df = pd.read_excel(path)
            return df.to_markdown(index=False)
        except Exception as e:
            return f"[Excel Fehler: {str(e)}]"

    @staticmethod
    def _extract_plain_text(path):
        try:
            with open(path, 'rb') as f:
                raw = f.read(50000)
                enc = chardet.detect(raw)['encoding'] or 'utf-8'
            with open(path, 'r', encoding=enc, errors='replace') as f:
                content = f.read()
            if "\0" in content: # Binary fallback
                return UniversalExtractor._extract_with_cli_tool(path)
            return content
        except Exception as e:
            return f"[Text-Lese-Fehler: {str(e)}]"

def attach_content_to_chat(hist, attach_type, attach_id, custom_text, uploaded_files, sb_files, user_state):
    """Attach multiple files content to chat"""
    if not user_state or not user_state.get("id"):
        return hist, "❌ Bitte anmelden"
    
    user_id = user_state["id"]
    content_to_add = ""
    status_msg = []

    # Helper to process a single file path
    def process_file_path(path, source_label):
        fname = os.path.basename(path)
        extracted = UniversalExtractor.extract(path)
        return f"\n\n=== 📄 Datei: {fname} ({source_label}) ===\n{extracted}\n"

    try:
        # 1. Browser Upload (Multiple)
        if attach_type == "Datei uploaden" and uploaded_files:
            # Ensure it's a list
            files_list = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]
            
            for file_obj in files_list:
                # Gradio passes file paths in .name
                content_to_add += process_file_path(file_obj.name, "Upload")
                status_msg.append(os.path.basename(file_obj.name))

        # 2. Storage Box (Multiple)
        elif attach_type == "Storage Box Datei" and sb_files:
            # Ensure it's a list
            files_list = sb_files if isinstance(sb_files, list) else [sb_files]
            
            for f_path in files_list:
                if not f_path.startswith("/"):
                    f_path = os.path.join(STORAGE_MOUNT_POINT, f_path)
                
                if os.path.exists(f_path):
                    local_temp = copy_storage_file_to_temp(f_path)
                    content_to_add += process_file_path(local_temp, "Cloud")
                    status_msg.append(os.path.basename(f_path))
                    try: os.remove(local_temp)
                    except: pass

        # 3. Transcript
        elif attach_type == "Transkript":
            if not attach_id: return hist, "❌ ID fehlt"
            db = get_db()
            trans = db.query(Transcription).filter(Transcription.id == int(attach_id), Transcription.user_id == user_id).first()
            db.close()
            if trans: 
                content_to_add = f"[Transkript #{trans.id}]\n\n{trans.original_text}"
                status_msg.append(f"Transkript {trans.id}")

        # 4. Vision
        elif attach_type == "Vision-Ergebnis":
            if not attach_id: return hist, "❌ ID fehlt"
            db = get_db()
            vis = db.query(VisionResult).filter(VisionResult.id == int(attach_id), VisionResult.user_id == user_id).first()
            db.close()
            if vis: 
                content_to_add = f"[Vision #{vis.id}]\n\n{vis.result}"
                status_msg.append(f"Vision {vis.id}")

        # 5. Custom Text
        elif attach_type == "Eigener Text":
            if custom_text: 
                content_to_add = custom_text
                status_msg.append("Eigener Text")

    except Exception as e:
        logger.exception(f"Attachment Error: {e}")
        return hist, f"🔥 Fehler: {str(e)}"

    # Append to Chat
    if not hist: hist = []
    
    if content_to_add:
        # Safety truncate if huge
        if len(content_to_add) > 150000:
            content_to_add = content_to_add[:150000] + "\n\n[... Inhalt gekürzt (Max Limit) ...]"
            
        hist.append({"role": "user", "content": content_to_add})
        return hist, f"✅ Angehängt: {', '.join(status_msg)}"
    
    return hist, "❌ Nichts ausgewählt"
def get_user_prompt_choices(user_state):
    """Get list of user's custom prompt names for dropdown"""
    if not user_state or not user_state.get("id"): return []
    prompts = get_user_custom_prompts(user_state["id"])
    return [p.name for p in prompts]

def insert_custom_prompt(prompt_name, current_msg, user_state):
    """Insert selected custom prompt text"""
    if not user_state or not user_state.get("id") or not prompt_name: 
        return current_msg
    
    db = get_db()
    p = db.query(CustomPrompt).filter(CustomPrompt.user_id == user_state["id"], CustomPrompt.name == prompt_name).first()
    db.close()
    
    if p:
        if current_msg:
            return current_msg + "\n\n" + p.prompt_text
        return p.prompt_text
    return current_msg

def load_single_chat(chat_id, user_state=None):
    """Load a specific chat into the UI"""
    if not user_state or not user_state.get("id"):
        return None, "❌ Bitte anmelden"
    
    if not chat_id:
        return None, "❌ Ungültige ID"

    chat = get_single_chat(int(chat_id), user_state["id"])
    if chat:
        try:
            messages = json.loads(chat.messages)
            if isinstance(messages, list) and len(messages) > 0:
                return messages, f"✅ Chat '{chat.title}' geladen"
            else:
                return None, "⚠️ Chat-Format veraltet"
        except Exception as e:
            return None, f"🔥 Ladefehler: {str(e)}"
            
    return None, "❌ Chat nicht gefunden"

def delete_chat(chat_id, user_state=None):
    """Delete a chat and update both list and state"""
    def get_fresh_data():
        return load_chat_list_with_state(user_state)

    if not user_state or not user_state.get("id") or not chat_id:
        d, s = get_fresh_data()
        return "❌ Fehler/Auth", d, s

    if delete_chat_history(int(chat_id), user_state["id"]):
        d, s = get_fresh_data()
        return "✅ Chat gelöscht", d, s
        
    d, s = get_fresh_data()
    return "❌ Fehler beim Löschen", d, s

def clear_chat():
    """Clear current chat"""
    return [], ""

def get_filtered_model_choices(provider, available_models_list, force_all, user_state):
    """
    Helper to filter models based on user preferences.
    """
    # If no login, return all models
    if not user_state or not user_state.get("id"):
        return available_models_list

    # 1. Get User Preferences (Ordered ID list)
    pref_ids = get_user_visible_models(user_state["id"], provider)
    
    # If no preferences set, return everything
    if not pref_ids:
        return available_models_list

    val_to_name = {v: k for k, v in available_models_list}
    
    # 2. Separate into Favorites and Others
    fav_choices = []
    seen_ids = set()
    
    for pid in pref_ids:
        if pid in val_to_name:
            fav_choices.append((val_to_name[pid], pid))
            seen_ids.add(pid)
            
    rest_choices = []
    for name, val in available_models_list:
        if val not in seen_ids:
            rest_choices.append((name, val))
            
    # 3. Decision Logic
    if not force_all and fav_choices:
        return fav_choices
    
    return fav_choices + rest_choices

# --- Functions for Transkription ---

# Send to Chat functionality
def send_transcript_to_chat(transcript, template, notes, custom_prompt, provider, model, api_key, user_state):
    # --- SECURITY CHECK ---
    if not user_state or not user_state.get("id"):
        return "⛔ Nicht autorisiert. Bitte anmelden."
    # ----------------------

    """Process transcript with selected prompt and return result"""
    if not transcript or transcript.strip() == "":
        return "❌ Kein Transkript vorhanden."

    # Build the full prompt
    if template == "Eigener Prompt":
        if not custom_prompt or custom_prompt.strip() == "":
            return "❌ Bitte eigenen Prompt eingeben."
        full_prompt = TRANSCRIPT_PROMPTS[template].format(
            transcript=transcript,
            notes=custom_prompt
        )
    else:
        prompt_template_text = TRANSCRIPT_PROMPTS.get(template, TRANSCRIPT_PROMPTS["Zusammenfassung"])
        full_prompt = prompt_template_text.format(
            transcript=transcript,
            notes=notes if notes else "(keine weiteren Hinweise)"
        )

    # Call chat API
    try:
        client = get_client(provider, api_key)
        status = f"🤖 Verarbeite mit {provider}/{model}...\n"

        messages = [
            {"role": "system", "content": "Du bist ein professioneller Redakteur und Content-Spezialist für kirchliche und akademische Veranstaltungen."},
            {"role": "user", "content": full_prompt}
        ]

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=3000
        )

        result = response.choices[0].message.content
        status += f"✅ Fertig!\n\n---\n\n{result}"
        return status

    except Exception as e:
        return f"🔥 Fehler: {str(e)}"

def generate_and_handle_ui(prompt, provider, model, width, height, steps, key, user_state):

    """Generates image and updates ALL UI components"""
    img_path, status = run_image_gen(prompt, provider, model, width, height, steps, key, user_state)

    if img_path:
        # Copy to a more permanent location for download
        import shutil
        download_dir = "/tmp/gradio_downloads"
        os.makedirs(download_dir, exist_ok=True)
        download_path = os.path.join(download_dir, f"image_{int(time.time())}.jpg")
        shutil.copy2(img_path, download_path)
        
        return (
            img_path,                   # g_out (Preview)
            status,                     # g_stat
            img_path,                   # g_img_path (State)
            download_path,              # g_download_file (Direct file link)
            gr.update(visible=True),    # g_save_btn
            ""                          # g_save_status (Reset)
        )
    else:
        return (
            None, 
            status, 
            None,
            None,                       # g_download_file
            gr.update(visible=False), 
            ""
        )

# --- DB SAVE WRAPPER ---
def process_gallery_save(img_path, provider, prompt, model, user_state):
    """Explicit wrapper to handle DB saving safely"""
    try:
        if not user_state or not user_state.get("id"):
            return "❌ Bitte anmelden", gr.update(visible=True)
        
        user_id = user_state["id"]
        
        if not img_path or not os.path.exists(img_path):
            return "❌ Datei nicht gefunden (Session abgelaufen?)", gr.update(visible=True)

        import shutil
        permanent_dir = "/var/www/transkript_app/generated_images"
        os.makedirs(permanent_dir, exist_ok=True)
        filename = f"img_{int(time.time())}_{os.path.basename(img_path)}"
        permanent_path = os.path.join(permanent_dir, filename)
        shutil.copy2(img_path, permanent_path)

        img_id = save_generated_image(
            user_id=int(user_id), 
            provider=str(provider), 
            model=str(model), 
            prompt=str(prompt), 
            image_path=str(permanent_path)
        )
        return f"✅ Gespeichert (ID: {img_id})", gr.update(visible=False)

    except Exception as e:
        logger.exception(f"Gallery Save Error: {e}")
        return f"🔥 Fehler: {str(e)}", gr.update(visible=True)

def manual_save_transcription(original, translated, provider, model, lang, user_state, filename="manual_save.mp3"):
    """Manually save transcription to database"""
    try:
        if not user_state or not user_state.get("id"):
            return "❌ Bitte anmelden"

        if not original or original.strip() == "":
            return "❌ Kein Transkript zum Speichern"

        trans_id = save_transcription(
            user_id=user_state["id"],
            provider=provider,
            model=model or "N/A",
            original=original,
            translated=translated if translated and translated != "(Whisper: Keine Übersetzung verfügbar)" else None,
            language=lang,
            filename=filename
        )

        return f"✅ Transkript gespeichert (ID: {trans_id})"

    except Exception as e:
        logger.exception(f"Error manually saving transcription: {str(e)}")
        return f"🔥 Fehler: {str(e)}"
    
    
# ==========================================
# 🖥️ GUI BUILDER
# ==========================================

with gr.Blocks(title="Akademie KI Suite", theme=gr.themes.Soft(), head=PWA_HEAD) as demo:
    
    # 1. Define the Session "Backpack" (Stores data per browser tab)
    session_state = gr.State({"id": None, "username": None, "is_admin": False})

    # Set higher file size limits
    gr.set_static_paths(paths=["/var/www/transkript_app/static"])

    # Login/Logout UI
    with gr.Row():
        gr.Markdown("# ⛪ KI Toolkit")
        with gr.Column(scale=1):
            login_status = gr.Markdown("👤 Nicht angemeldet")
            logout_btn = gr.Button("🚪 Abmelden", visible=False, size="sm")

    # Login Screen (shown when not logged in)
    login_screen = gr.Column(visible=True)
    with login_screen:
        gr.Markdown("## 🔐 Anmeldung erforderlich")
        with gr.Row():
            with gr.Column(scale=1):
                pass
            with gr.Column(scale=1):
                login_username = gr.Textbox(label="Benutzername", placeholder="user123")
                login_password = gr.Textbox(label="Passwort", type="password", placeholder="meinpasswort123")
                login_btn = gr.Button("🔓 Anmelden", variant="primary")
                login_message = gr.Markdown("")
            with gr.Column(scale=1):
                pass

    # Main App (shown when logged in)
    main_app = gr.Column(visible=False)
    with main_app:

        with gr.Tabs():

            # --- TAB 1: CHAT ---
            with gr.TabItem("💬 Chat", id="chat_tab") as chat_tab:
                with gr.Row():
                    with gr.Column(scale=3):
                        # Top Bar
                        with gr.Row():
                            chat_providers = [k for k, v in PROVIDERS.items() if "chat_models" in v]
                            
                            c_prov = gr.Dropdown(
                                choices=chat_providers, 
                                value="Scaleway", 
                                label="Anbieter", 
                                scale=2
                            )
                            c_model = gr.Dropdown(
                                choices=PROVIDERS["Scaleway"]["chat_models"], 
                                value=PROVIDERS["Scaleway"]["chat_models"][0], # Ensure this is not None
                                label="Modell", 
                                scale=4
                            )
                            c_model = gr.Dropdown(PROVIDERS["Scaleway"]["chat_models"], value=PROVIDERS["Scaleway"]["chat_models"][0], label="Modell", scale=4)
                            c_load_all = gr.Button("🌍 Alle", scale=0, size="sm", min_width=60)

                        c_badge = gr.HTML(value=PROVIDERS["Scaleway"]["badge"])

                        # UI Updates
                        c_prov.change(
                            lambda p, s: update_c_ui(p, force_all=False, user_state=s), # You will need to update update_c_ui signature too
                            inputs=[c_prov, session_state], 
                            outputs=[c_model, c_badge]
                        )
                        c_load_all.click(
                            lambda p, s: update_c_ui(p, force_all=True, user_state=s), 
                            inputs=[c_prov, session_state], 
                            outputs=[c_model, c_badge]
                        )

                        # Chat Area
                        c_bot = gr.Chatbot(
                            height=500, 
                            type="messages", 
                            show_copy_button=True
                        )
                        c_msg = gr.Textbox(placeholder="Nachricht...", show_label=False, lines=3)

                        with gr.Row():
                            c_btn = gr.Button("📤 Senden", variant="primary", scale=2)
                            c_stop_btn = gr.Button("🛑 Stop", variant="stop", scale=1)
                            c_save_btn = gr.Button("💾 Speichern", scale=1)
                            c_clear_btn = gr.Button("🗑️ Neu", scale=1)

                        c_save_status = gr.Markdown("")

                    # Right Sidebar
                    with gr.Column(scale=1):
                        
                        # 1. Settings (Closed)
                        with gr.Accordion("⚙️ Einstellungen", open=False):
                            c_key = gr.Textbox(label="API Key (Optional)", type="password")
                            c_sys = gr.Textbox(label="System Rolle", value="Du bist ein hilfreicher Assistent.", lines=2)
                            
                            gr.Markdown("**🧠 Reasoning / Thinking Configuration**")
                            with gr.Row():
                                c_reasoning_effort = gr.Dropdown(
                                    choices=["default", "low", "medium", "high"],
                                    value="default",
                                    label="Reasoning Effort",
                                    info="Für OpenAI o-series & Scaleway."
                                )
                                c_reasoning_tokens = gr.Slider(
                                    0, 32000, value=0, step=1024,
                                    label="Reasoning Token Budget",
                                    info="0 = Auto. Für Anthropic/OpenRouter/Mistral."
                                )
                            
                            c_temp = gr.Slider(0, 2, value=0.7, label="Temperatur (Standard)", step=0.1)
                            

                        # 2. History (Closed)
                        with gr.Accordion("📚 Alte Chats laden", open=False):
                            c_history_state = gr.State([]) 
                            
                            with gr.Row():
                                refresh_chats_btn = gr.Button("🔄 Liste aktualisieren", size="sm")
                                delete_chat_btn = gr.Button("🗑️ Löschen", variant="stop", size="sm")
                            
                            old_chats = gr.Dataframe(
                                headers=["ID", "Datum", "Titel", "Modell"], 
                                value=[[None, "", "", ""]], 
                                label="Gespeicherte Chats", 
                                interactive=False, 
                                height=200, 
                                wrap=True
                            )
                            
                            with gr.Row():
                                load_chat_id = gr.Number(label="Chat-ID", precision=0)
                            
                            load_chat_btn = gr.Button("📥 Chat laden", variant="primary")
                            chat_load_status = gr.Markdown("")

                            # --- Events for History ---
                            
                            # 1. Load list on click/tab select
                            chat_tab.select(
                                load_chat_list_with_state, 
                                inputs=[session_state], 
                                outputs=[old_chats, c_history_state]
                            )
                            refresh_chats_btn.click(
                                load_chat_list_with_state, 
                                inputs=[session_state],
                                outputs=[old_chats, c_history_state]
                            )
                            
                            # 2. Select row to get ID (Using State to avoid Index Errors)
                            old_chats.select(
                                select_chat_row, 
                                inputs=[c_history_state], 
                                outputs=[load_chat_id]
                            )
                            
                            # 3. Load Chat Logic
                            load_chat_btn.click(
                                load_single_chat, 
                                inputs=[load_chat_id], 
                                outputs=[c_bot, chat_load_status]
                            )
                            
                            # 4. Delete Logic (Now updates State + DataFrame to prevent 'backend' errors on next click)
                            delete_chat_btn.click(
                                delete_chat, 
                                inputs=[load_chat_id], 
                                outputs=[chat_load_status, old_chats, c_history_state]
                            )

                        # 3. Attachments & Prompts (Closed)
                        with gr.Accordion("📎 Inhalt & Prompts", open=False):
                            
                            # Custom Prompts Section
                            gr.Markdown("**📝 Vorlagen**")
                            with gr.Row():
                                c_prompt_select = gr.Dropdown(choices=[], label="Vorlage wählen", scale=2)
                                c_prompt_refresh = gr.Button("🔄", scale=0, size="sm")
                            c_insert_prompt_btn = gr.Button("⬇️ In Textfeld einfügen", size="sm")
                            
                            gr.Markdown("---")
                            
                            # Content Attachments Section
                            gr.Markdown("**📎 Anhang**")
                            gr.Markdown("_Hinweis: Inhalte werden automatisch extrahiert und zusammengefügt (Max. 150.000 Zeichen)_", visible=True)
                            
                            attach_type = gr.Radio(
                                ["Transkript", "Vision-Ergebnis", "Eigener Text", "Datei uploaden", "Storage Box Datei"],
                                value="Transkript",
                                label="Typ"
                            )
                            
                            # Dynamic inputs
                            attach_id = gr.Number(label="ID (Transkript/Vision)", precision=0, visible=True)
                            attach_custom = gr.Textbox(label="Text einfügen", lines=3, visible=False)
                            
                            # --- FIXED: Multiple File Upload Support ---
                            attach_file = gr.File(
                                label="Dateien wählen (PDF, Bilder, Word, etc.)", 
                                visible=False, 
                                file_count="multiple",  # <--- ALLOWS MULTIPLE FILES
                                type="filepath"
                            )
                            
                            # STORAGE BOX BROWSER FOR CHAT
                            with gr.Group(visible=False) as sb_group:
                                gr.Markdown("Dateien auf Server:")
                                attach_sb_browser = gr.FileExplorer(
                                    root_dir=STORAGE_MOUNT_POINT,
                                    glob="**/*",
                                    height=200,
                                    file_count="multiple" # <--- ALLOWS MULTIPLE SELECTION IN BROWSER
                                )
                                sb_refresh_btn = gr.Button("🔄 Aktualisieren", size="sm")

                            attach_btn = gr.Button("➕ An Chat anhängen", variant="secondary")
                            attach_status = gr.Markdown("")

                            # Toggle visibility logic
                            def toggle_attach_inputs(atype):
                                return (
                                    gr.update(visible=atype in ["Transkript", "Vision-Ergebnis"]),
                                    gr.update(visible=atype == "Eigener Text"),
                                    gr.update(visible=atype == "Datei uploaden"),
                                    gr.update(visible=atype == "Storage Box Datei")
                                )
                            
                            attach_type.change(
                                toggle_attach_inputs, 
                                attach_type, 
                                [attach_id, attach_custom, attach_file, sb_group]
                            )
                            
                            # Refresh Logic for File Browser
                            def refresh_chat_sb():
                                return gr.update(value=None)
                            sb_refresh_btn.click(refresh_chat_sb, outputs=attach_sb_browser)

                # --- EVENT WIRING ---

                # Chat Execution (With Stop)
                submit_event = c_msg.submit(user_msg, [c_msg, c_bot], [c_msg, c_bot], queue=False).then(
                    bot_msg, 
                    # Add session_state to the inputs list
                    [c_bot, c_prov, c_model, c_temp, c_sys, c_key, c_reasoning_effort, c_reasoning_tokens, session_state], 
                    c_bot
                )

                click_event = c_btn.click(user_msg, [c_msg, c_bot], [c_msg, c_bot], queue=False).then(
                    bot_msg, 
                    # Add session_state to the inputs list
                    [c_bot, c_prov, c_model, c_temp, c_sys, c_key, c_reasoning_effort, c_reasoning_tokens, session_state], 
                    c_bot
                )
                
                # Stop Button
                c_stop_btn.click(fn=None, cancels=[submit_event, click_event])

                # Save & Clear
                c_save_btn.click(save_chat, [c_bot, c_prov, c_model, session_state], c_save_status)
                c_clear_btn.click(lambda: ([], ""), outputs=[c_bot, c_save_status])

                # History Logic
                chat_tab.select(load_chat_list_with_state, inputs=[session_state], outputs=[old_chats, c_history_state])
                refresh_chats_btn.click(load_chat_list_with_state, inputs=[session_state], outputs=[old_chats, c_history_state])
                old_chats.select(select_chat_row, inputs=[c_history_state], outputs=[load_chat_id])
                load_chat_btn.click(load_single_chat, inputs=[load_chat_id, session_state], outputs=[c_bot, chat_load_status])
                delete_chat_btn.click(delete_chat, inputs=[load_chat_id, session_state], outputs=[chat_load_status, old_chats, c_history_state])

                # Attachment Logic
                attach_btn.click(
                    attach_content_to_chat, 
                    # Added session_state
                    inputs=[c_bot, attach_type, attach_id, attach_custom, attach_file, attach_sb_browser, session_state], 
                    outputs=[c_bot, attach_status]
                )

                # Prompt Logic
                c_prompt_refresh.click(
                    get_user_prompt_choices, 
                    inputs=[session_state], 
                    outputs=c_prompt_select
                )
                chat_tab.select(
                    get_user_prompt_choices, 
                    inputs=[session_state], 
                    outputs=c_prompt_select
                )

                c_insert_prompt_btn.click(
                    insert_custom_prompt, 
                    inputs=[c_prompt_select, c_msg, session_state], 
                    outputs=[c_msg]
                )
            # --- TAB 2: TRANSKRIPTION ---
            with gr.TabItem("🎙️ Transkription"):
                with gr.Row():
                    with gr.Column():
                        # --- INPUT SELECTION: Upload vs Storage Box ---
                        with gr.Tabs():
                            with gr.TabItem("📤 Upload"):
                                t_audio = gr.Audio(type="filepath", label="Datei hochladen")
                            
                            with gr.TabItem("📦 Storage Box"):
                                gr.Markdown("Wähle Datei aus Cloud-Speicher:")
                                t_storage_browser = gr.FileExplorer(
                                    root_dir=STORAGE_MOUNT_POINT,
                                    glob="**/*", 
                                    height=300,
                                    label="Dateien durchsuchen"
                                )
                                with gr.Row():
                                    t_refresh_sb_btn = gr.Button("🔄 Aktualisieren", size="sm", scale=0)
                                    t_load_sb_btn = gr.Button("✅ Diese verwenden", variant="secondary", scale=1)
                                t_sb_status = gr.Markdown("")

                        # Logic: Storage Box Selection
                        def use_storage_file(selected_files):
                            if not selected_files:
                                return None, "❌ Keine Datei ausgewählt"
                            f_path = selected_files[0] if isinstance(selected_files, list) else selected_files
                            if not f_path.startswith("/"):
                                f_path = os.path.join(STORAGE_MOUNT_POINT, f_path)
                            try:
                                local_temp = copy_storage_file_to_temp(f_path)
                                return local_temp, f"✅ Geladen: {os.path.basename(f_path)}"
                            except Exception as e:
                                return None, f"🔥 Fehler: {str(e)}"

                        def refresh_explorer():
                            return gr.update(value=None) 

                        t_load_sb_btn.click(use_storage_file, inputs=t_storage_browser, outputs=[t_audio, t_sb_status])
                        t_refresh_sb_btn.click(refresh_explorer, outputs=t_storage_browser)

                        # --- MAIN CONTROLS ---
                        gr.Markdown("### 🎛️ Auswahl")
                        with gr.Group():
                            with gr.Row():
                                t_prov = gr.Dropdown(
                                    choices=["Gladia", "Deepgram", "AssemblyAI", "Mistral", "Scaleway", "Groq"], 
                                    value="Gladia", 
                                    label="Engine",
                                    scale=2
                                )
                                t_diar = gr.Checkbox(
                                    value=True, 
                                    label="🎭 Sprecher erkennen",
                                    scale=1,
                                    container=False
                                )
                            
                            # Badge moved to separate row below
                            t_badge = gr.HTML(value=get_compliance_html("Gladia"))

                        # --- SETTINGS ACCORDION ---
                        with gr.Accordion("⚙️ Einstellungen", open=False):
                            
                            # Note: Badge removed from here as it is now in the top row

                            # 1. Language & Model
                            with gr.Row():
                                t_lang = gr.Dropdown(
                                    [("Auto-Erkennung", "auto"), ("Deutsch", "de"), ("Englisch", "en")], 
                                    value="de", 
                                    label="Sprache"
                                )
                                
                                with gr.Row():
                                    t_model = gr.Dropdown(choices=[], value=None, label="Modell", scale=3, visible=False)
                                    t_refresh_models = gr.Button("🔄", size="sm", scale=0, variant="secondary")

                            # 2. Translation Settings
                            with gr.Row():
                                t_trans = gr.Checkbox(False, label="🌍 Übersetzen")
                                t_target = gr.Dropdown(
                                    [("Deutsch", "de"), ("Englisch", "en")], 
                                    value="en", 
                                    label="Zielsprache", 
                                    visible=False
                                )

                            # 3. Whisper Options (Hidden Group)
                            with gr.Group(visible=False) as w_options_group:
                                gr.Markdown("---")
                                gr.Markdown("**Erweiterte Optionen (Whisper)**")
                                with gr.Row():
                                    w_chunk_opt = gr.Checkbox(value=True, label="✂️ Chunking")
                                    w_chunk_len = gr.Number(value=10, label="Minuten", precision=0, minimum=1)
                                    w_temp = gr.Slider(0, 1, value=0, step=0.1, label="Temperatur")
                                w_prompt = gr.Textbox(label="Kontext-Prompt", placeholder="Optionaler Kontext...")

                        # Hidden state for API key
                        t_key = gr.State(value="") 
                        
                        # Start Button
                        t_btn = gr.Button("▶️ Transkription starten", variant="primary", size="lg")
                        t_log = gr.Textbox(label="Status Log", lines=3)

                        # --- DYNAMIC UI LOGIC ---
                        
                        # 1. Provider Change -> Update Badge, Model, and Whisper Options
                        t_prov.change(
                            fn=update_t_ui, 
                            inputs=t_prov, 
                            outputs=[t_badge, t_model, w_options_group]
                        )
                        
                        # 2. Translation Toggle
                        def toggle_translation(chk):
                            return gr.update(visible=chk), gr.update(visible=chk)
                        
                        # 3. Model Refresh Logic (Optional specific handler)
                        t_refresh_models.click(
                            lambda p: update_t_ui(p, force_all=True)[1], # Return only model update
                            inputs=t_prov, 
                            outputs=t_model
                        )

                    with gr.Column():
                        # OUTPUTS
                        t_orig = gr.Textbox(label="📄 Transkript", lines=15, show_copy_button=True)
                        t_trsl = gr.Textbox(label="🌍 Übersetzung", lines=15, show_copy_button=True, visible=False)

                        # Wire visibility toggle for translation output
                        t_trans.change(toggle_translation, inputs=t_trans, outputs=[t_target, t_trsl])

                        with gr.Row():
                            t_save_btn = gr.Button("💾 Transkript speichern", variant="secondary")
                            t_save_status = gr.Markdown("")

                # --- SEND TO CHAT SECTION ---
                with gr.Accordion("💬 An Chat senden", open=False) as send_to_chat_section:
                    gr.Markdown("### Weiterverarbeitung")

                    with gr.Row():
                        with gr.Column(scale=1):
                            prompt_template = gr.Dropdown(
                                choices=list(TRANSCRIPT_PROMPTS.keys()),
                                value="Veranstaltungsrückblick",
                                label="📋 Prompt-Vorlage"
                            )
                        with gr.Column(scale=1):
                            chat_provider = gr.Dropdown(
                                list(PROVIDERS.keys()),
                                value="Scaleway",
                                label="🤖 Chat Provider"
                            )
                        with gr.Column(scale=1):
                            chat_model_for_transcript = gr.Dropdown(
                                PROVIDERS["Scaleway"]["chat_models"],
                                value=PROVIDERS["Scaleway"]["chat_models"][0],
                                label="Modell"
                            )

                    additional_notes = gr.Textbox(
                        label="📝 Zusätzliche Hinweise",
                        placeholder="Erwähne Kooperationspartner, betone ...",
                        lines=2
                    )

                    custom_prompt_input = gr.Textbox(
                        label="✏️ Eigener Prompt",
                        lines=3,
                        visible=False
                    )

                    send_to_chat_btn = gr.Button("💬 An Chat senden", variant="primary")
                    send_status = gr.Markdown("")

                # --- LOGIC WIRING (Backend) ---
                
                # Chat Model Updates
                def update_chat_model_dropdown(prov):
                    ms = PROVIDERS.get(prov, {}).get("chat_models", [])
                    return gr.update(choices=ms, value=ms[0] if ms else "")
                chat_provider.change(update_chat_model_dropdown, chat_provider, chat_model_for_transcript)

                # Custom Prompt Visibility
                prompt_template.change(
                    lambda t: gr.update(visible=(t == "Eigener Prompt")), 
                    prompt_template, 
                    custom_prompt_input
                )

                # Main Execution
                # Mapping global inputs to the function arguments
                t_btn.click(
                    run_and_save_transcription, 
                    inputs=[
                        t_audio, t_prov, t_model, 
                        t_lang, w_temp, w_prompt, 
                        t_diar, t_trans, t_target, t_key,
                        w_chunk_opt, w_chunk_len,
                        t_lang, t_diar, t_lang, t_diar,
                        session_state 
                    ], 
                    outputs=[t_log, t_orig, t_trsl]
                )

                t_save_btn.click(
                    manual_save_transcription,
                    # Added session_state
                    inputs=[t_orig, t_trsl, t_prov, t_model, t_lang, session_state],
                    outputs=t_save_status
                )

                send_to_chat_btn.click(
                    send_transcript_to_chat,
                    inputs=[
                        t_orig,
                        prompt_template,
                        additional_notes,
                        custom_prompt_input,
                        chat_provider,
                        chat_model_for_transcript,
                        t_key,
                        session_state
                    ],
                    outputs=send_status
                )

            # --- TAB 3: VISION ---
            with gr.TabItem("👁️ Vision"):
                with gr.Row():
                    with gr.Column():
                        
                        # --- INPUT SELECTION: Upload vs Storage Box ---
                        with gr.Tabs():
                            with gr.TabItem("📤 Upload"):
                                v_img = gr.Image(type="filepath", label="Bild hochladen", height=300)
                            
                            with gr.TabItem("📦 Storage Box"):
                                gr.Markdown("Wähle ein Bild aus dem Cloud-Speicher:")
                                v_storage_browser = gr.FileExplorer(
                                    root_dir=STORAGE_MOUNT_POINT,
                                    glob="**/*.{png,jpg,jpeg,webp}", 
                                    height=300,
                                    label="Bilder durchsuchen"
                                )
                                with gr.Row():
                                    v_refresh_sb_btn = gr.Button("🔄 Aktualisieren", size="sm", scale=0)
                                    v_load_sb_btn = gr.Button("✅ Dieses Bild verwenden", variant="secondary", scale=1)
                                v_sb_status = gr.Markdown("")

                        # Logic: Storage Box Selection (Vision)
                        def use_storage_image(selected_files):
                            if not selected_files:
                                return None, "❌ Kein Bild ausgewählt"
                            f_path = selected_files[0] if isinstance(selected_files, list) else selected_files
                            if not f_path.startswith("/"):
                                f_path = os.path.join(STORAGE_MOUNT_POINT, f_path)
                            try:
                                local_temp = copy_storage_file_to_temp(f_path)
                                return local_temp, f"✅ Geladen: {os.path.basename(f_path)}"
                            except Exception as e:
                                return None, f"🔥 Fehler: {str(e)}"

                        def refresh_v_explorer():
                            return gr.update(value=None)

                        v_load_sb_btn.click(use_storage_image, inputs=v_storage_browser, outputs=[v_img, v_sb_status])
                        v_refresh_sb_btn.click(refresh_v_explorer, outputs=v_storage_browser)


                        # --- SELECTION ROW ---
                        gr.Markdown("### 🎛️ Auswahl")
                        with gr.Group():
                            with gr.Row(equal_height=True):
                                v_prov = gr.Dropdown(
                                    ["Scaleway", "Mistral", "Nebius", "OpenRouter", "Poe"], 
                                    value="Scaleway", 
                                    label="Provider", 
                                    scale=1
                                )
                                v_model = gr.Dropdown(
                                    PROVIDERS["Scaleway"]["vision_models"], 
                                    value="pixtral-12b-2409", 
                                    label="Modell", 
                                    allow_custom_value=True,
                                    scale=2
                                )
                                v_load_all = gr.Button("🔄", scale=0, size="sm", variant="secondary")
                            
                            # Badge Row
                            v_badge = gr.HTML(value=get_compliance_html("Scaleway"))

                        # Inputs
                        v_prompt = gr.Textbox(label="Frage", value="Beschreibe dieses Bild detailliert.", lines=2)
                        
                        # Hidden Key (Pass empty string or handle in backend)
                        v_key = gr.State(value="") 
                        
                        v_btn = gr.Button("👁️ Analysieren", variant="primary", size="lg")
                    
                    with gr.Column():
                        v_out = gr.Markdown(label="Ergebnis")
                        
                # UI Updates
                v_prov.change(
                    lambda p: update_v_ui(p, force_all=False), 
                    inputs=v_prov, 
                    outputs=[v_badge, v_model]
                )
                v_load_all.click(
                    lambda p: update_v_ui(p, force_all=True), 
                    inputs=v_prov, 
                    outputs=[v_badge, v_model]
                )
                
                # Execution
                v_btn.click(run_vision, [v_img, v_prompt, v_prov, v_model, v_key, session_state], v_out)
            

            # --- TAB 4: BILDERZEUGUNG ---
            with gr.TabItem("🎨 Bilderzeugung"):
                with gr.Row():
                    with gr.Column():
                        # Prompt Input
                        g_prompt = gr.Textbox(
                            label="Prompt", 
                            placeholder="Eine futuristische Kathedrale aus Glas und Licht...", 
                            lines=3
                        )
                        
                        # --- SELECTION ROW ---
                        gr.Markdown("### 🎛️ Auswahl")
                        with gr.Group():
                            with gr.Row(equal_height=True):
                                g_provider = gr.Dropdown(
                                    ["Nebius", "Scaleway", "OpenRouter", "Poe"], 
                                    value="Nebius", 
                                    label="Provider", 
                                    scale=1
                                )
                                g_model = gr.Dropdown(
                                    PROVIDERS["Nebius"]["image_models"], 
                                    value="black-forest-labs/flux-schnell", 
                                    label="Modell",
                                    scale=2
                                )
                                g_load_all = gr.Button("🔄", scale=0, size="sm", variant="secondary")

                            # Badge Row
                            g_badge = gr.HTML(value=get_compliance_html("Nebius"))

                        # --- SETTINGS ACCORDION ---
                        with gr.Accordion("⚙️ Einstellungen", open=False):
                            with gr.Row():
                                g_w = gr.Slider(256, 1440, value=1024, step=64, label="Breite")
                                g_h = gr.Slider(256, 1440, value=768, step=64, label="Höhe")
                            g_steps = gr.Slider(1, 50, value=4, step=1, label="Schritte")

                        # Hidden Key
                        g_key = gr.State(value="")

                        g_btn = gr.Button("🎨 Generieren", variant="primary", size="lg")
                        g_stat = gr.Textbox(label="Status", interactive=False, visible=True)
                        
                    with gr.Column():
                        g_out = gr.Image(label="Ergebnis", type="filepath", show_download_button=False, height=400)
                        
                        with gr.Row():
                             g_download_file = gr.File(label="Download", scale=1)
                             g_save_btn = gr.Button("💾 In Storage Box speichern", visible=False, scale=1, variant="secondary")
                        
                        g_save_status = gr.Markdown("")

                # State for path
                g_img_path = gr.State(value=None)
                
                # UI Updates
                g_provider.change(
                    lambda p: update_g_ui(p, force_all=False), 
                    inputs=g_provider, 
                    outputs=[g_badge, g_model]
                )
                g_load_all.click(
                    lambda p: update_g_ui(p, force_all=True), 
                    inputs=g_provider, 
                    outputs=[g_badge, g_model]
                )
                
                # Execution Logic
                g_btn.click(
                    generate_and_handle_ui, 
                    inputs=[g_prompt, g_provider, g_model, g_w, g_h, g_steps, g_key, session_state],  # ← ADD session_state
                    outputs=[g_out, g_stat, g_img_path, g_download_file, g_save_btn, g_save_status]
                )

                # Save to Storage Box Logic
                g_save_btn.click(
                    process_gallery_save,
                    inputs=[g_img_path, g_provider, g_prompt, g_model, session_state], 
                    outputs=[g_save_status, g_save_btn]
                )

            # --- TAB 5: VERLAUF & VERWALTUNG ---
            with gr.TabItem("📚 Verlauf & Verwaltung", id="tab_management"):
                
                gr.Markdown("### ⚙️ Verwaltung")
                
                # Optional: Helper to make tables look full
                def pad_data(data, width, min_rows=6):
                    while len(data) < min_rows:
                        # Use empty strings instead of None to prevent JS freezes
                        row = [""] * width 
                        data.append(row)
                    return data

                with gr.Tabs() as history_tabs:

                    # =========================================================
                    # 1. TRANSCRIPTIONS
                    # =========================================================
                    with gr.TabItem("🎙️ Transkriptions-Verlauf") as trans_tab:
                        # State to store real data (for lookup)
                        trans_state = gr.State([])

                        # 1. TABLE (Initialized with empty skeleton rows)
                        with gr.Group():
                            gr.Markdown("#### 📋 Gespeicherte Transkripte")
                            trans_history = gr.Dataframe(
                                headers=["ID", "Datum", "Titel", "Provider", "Sprache"],
                                value=[[None, "", "", "", ""]] * 6, # <--- INITIALIZES EMPTY GRID
                                interactive=False,
                                wrap=True,
                                height=300,
                                datatype=["number", "str", "str", "str", "str"],
                                column_widths=["10%", "20%", "40%", "15%", "15%"]
                            )

                        # 2. CONTROLS
                        with gr.Row(variant="panel", equal_height=True):
                            with gr.Column(scale=1):
                                trans_id_input = gr.Number(label="Ausgewählte ID", precision=0, minimum=0)
                            with gr.Column(scale=0, min_width=50):
                                refresh_trans_btn = gr.Button("🔄", size="lg")
                            with gr.Column(scale=2):
                                with gr.Row():
                                    load_trans_btn = gr.Button("📄 Laden", variant="secondary", size="lg")
                                    trans_to_chat_btn = gr.Button("📨 An Chat senden", variant="primary", size="lg")
                            with gr.Column(scale=1):
                                delete_trans_btn = gr.Button("🗑️ Löschen", variant="stop", size="lg")

                        # 3. PREVIEW
                        loaded_trans_display = gr.Textbox(label="Inhalt", lines=8, max_lines=15, show_copy_button=True)
                        trans_action_status = gr.Markdown("")

                        # --- LOGIC ---
                        def load_trans_data(user_state=None):
                            if not user_state or not user_state.get("id"):
                                return pad_data([], 5), []
                            try:
                                t_list = get_user_transcriptions(user_state["id"])
                                clean_data = [[t.id, t.timestamp.strftime("%Y-%m-%d %H:%M"), t.title or "—", t.provider, t.language] for t in t_list]
                                return pad_data(list(clean_data), 5), clean_data
                            except Exception as e:
                                logger.exception(e)
                                return pad_data([], 5), []

                        def load_single_trans(tid, user_state):
                            if not tid or not user_state or not user_state.get("id"): 
                                return gr.update(), "❌"
                            db = SessionLocal()
                            t = db.query(Transcription).filter(Transcription.id == int(tid), Transcription.user_id == user_state["id"]).first()
                            db.close()
                            return (t.original_text, f"✅ Geladen: {t.title}") if t else ("", "❌ Nicht gefunden")
                        
                        def select_trans_row(evt: gr.SelectData, state_data, user_state):
                            """Smart Selection: Uses state to find ID from ANY column click"""
                            try:
                                if not user_state or not user_state.get("id"):
                                    return 0, "", "❌ Bitte anmelden"

                                row_idx = evt.index[0]
                                # Check if row exists in real data
                                if state_data and row_idx < len(state_data):
                                    real_row = state_data[row_idx]
                                    t_id = int(real_row[0]) # ID is col 0
                                    
                                    # Call loader with state
                                    content, status = load_single_trans(t_id, user_state)
                                    return t_id, content, status
                            except Exception as e: 
                                logger.error(f"Select Error: {e}")
                            
                            return 0, "", "" # Return safe defaults instead of gr.update()

                        def del_trans(tid, user_state):
                            if not user_state or not user_state.get("id"):
                                return "", "❌ Auth Fehler", [], []
                            
                            if delete_transcription(int(tid or 0), user_state["id"]):
                                d, s = load_trans_data(user_state) # Pass state recursively
                                return "", "✅ Gelöscht", d, s
                            
                            d, s = load_trans_data(user_state)
                            return "", "❌ Fehler", d, s

                        # Wiring
                        refresh_trans_btn.click(load_trans_data, inputs=[session_state], outputs=[trans_history, trans_state])

                        # --- AUTO-LOAD ON TAB SELECT ---
                        trans_tab.select(load_trans_data, inputs=[session_state], outputs=[trans_history, trans_state])

                        # Pass 'trans_state' to select so we know what was clicked
                        trans_history.select(
                            select_trans_row, 
                            inputs=[trans_state, session_state], 
                            outputs=[trans_id_input, loaded_trans_display, trans_action_status]
                        )
                        trans_id_input.change(load_single_trans, inputs=[trans_id_input, session_state], outputs=[loaded_trans_display, trans_action_status])
                        delete_trans_btn.click(del_trans, inputs=[trans_id_input, session_state], outputs=[loaded_trans_display, trans_action_status, trans_history, trans_state])
                        
                        # Chat Button
                        if 'msg_input' in locals():
                            trans_to_chat_btn.click(
                                lambda x: x, 
                                inputs=loaded_trans_display, 
                                outputs=c_msg 
                            )

                    # =========================================================
                    # 2. GENERATED IMAGES
                    # =========================================================
                    with gr.TabItem("🎨 Generierte Bilder") as images_tab:
                        img_state = gr.State([])

                        with gr.Group():
                            gr.Markdown("#### 🖼️ Bild-Historie")
                            images_history = gr.Dataframe(
                                headers=["ID", "Datum", "Prompt", "Modell"],
                                value=[[None, "", "", ""]] * 6, # Initial Skeleton
                                interactive=False,
                                wrap=True,
                                height=300,
                                datatype=["number", "str", "str", "str"],
                                column_widths=["10%", "20%", "50%", "20%"]
                            )

                        with gr.Row(variant="panel", equal_height=True):
                            with gr.Column(scale=1):
                                img_id_input = gr.Number(label="Ausgewählte ID", precision=0, minimum=0)
                            with gr.Column(scale=0, min_width=50):
                                refresh_images_btn = gr.Button("🔄", size="lg")
                            with gr.Column(scale=2):
                                with gr.Row():
                                    load_img_btn = gr.Button("🖼️ Laden", variant="secondary", size="lg")
                                    img_to_chat_btn = gr.Button("📨 Prompt an Chat", variant="primary", size="lg")
                            with gr.Column(scale=1):
                                delete_img_btn = gr.Button("🗑️ Löschen", variant="stop", size="lg")

                        with gr.Row():
                            loaded_img_display = gr.Image(label="Vorschau", height=300, type="filepath", interactive=False)
                            with gr.Column():
                                loaded_img_prompt = gr.Textbox(label="Prompt", lines=10, show_copy_button=True)
                                img_action_status = gr.Markdown("")

                        # --- LOGIC ---
                        def load_img_data(user_state=None):
                            if not user_state or not user_state.get("id"): 
                                return pad_data([], 4), []
                            try:
                                i_list = get_user_generated_images(user_state["id"])
                                clean = [[i.id, i.timestamp.strftime("%Y-%m-%d"), i.prompt, i.model] for i in i_list]
                                return pad_data(list(clean), 4), clean
                            except: return pad_data([], 4), []

                        def load_single_img(tid, user_state=None):
                            # Security check: technically images are static files, but we check ownership of metadata
                            if not tid or not user_state or not user_state.get("id"): 
                                return None, "", "❌"
                            
                            db = SessionLocal()
                            img = db.query(GeneratedImage).filter(GeneratedImage.id == int(tid), GeneratedImage.user_id == user_state["id"]).first()
                            db.close()
                            
                            if img and os.path.exists(img.image_path):
                                return img.image_path, img.prompt, f"✅ Geladen"
                            return None, "", "❌ Datei fehlt/Zugriff verweigert"

                        def select_img_row(evt: gr.SelectData, state_data, user_state):
                            try:
                                if not user_state or not user_state.get("id"):
                                    return 0, None, "", "❌ Bitte anmelden"

                                row_idx = evt.index[0]
                                if state_data and row_idx < len(state_data):
                                    real_row = state_data[row_idx]
                                    tid = int(real_row[0])
                                    
                                    # Call loader with state
                                    path, prmt, stat = load_single_img(tid, user_state)
                                    return tid, path, prmt, stat
                            except: pass
                            return 0, None, "", ""

                        def del_img(tid, user_state):
                            if not user_state or not user_state.get("id"):
                                return None, "", "❌ Auth", [], []

                            delete_generated_image(int(tid or 0), user_state["id"])
                            d, s = load_img_data(user_state)
                            return None, "", "✅ Gelöscht", d, s

                        refresh_images_btn.click(load_img_data, inputs=[session_state], outputs=[images_history, img_state])
                        
                        images_tab.select(load_img_data, inputs=[session_state], outputs=[images_history, img_state])

                        img_id_input.change(load_single_img, inputs=[img_id_input, session_state], outputs=[loaded_img_display, loaded_img_prompt, img_action_status])
                        
                        delete_img_btn.click(del_img, inputs=[img_id_input, session_state], outputs=[loaded_img_display, loaded_img_prompt, img_action_status, images_history, img_state])
                        
                        images_history.select(
                            select_img_row, 
                            inputs=[img_state, session_state], 
                            outputs=[img_id_input, loaded_img_display, loaded_img_prompt, img_action_status]
                        )

                        if 'msg_input' in locals():
                            img_to_chat_btn.click(
                                lambda x: x, 
                                inputs=loaded_img_prompt, 
                                outputs=c_msg
                            )

                    # =========================================================
                    # 3. CUSTOM PROMPTS
                    # =========================================================
                    with gr.TabItem("✏️ Eigene Prompt-Vorlagen") as prompts_tab:
                        prompt_state = gr.State([])
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("#### ✨ Neue Vorlage")
                                new_prompt_name = gr.Textbox(label="Name")
                                new_prompt_category = gr.Dropdown(["Transkription", "Chat", "Vision", "Allgemein"], value="Chat", label="Kategorie")
                                new_prompt_text = gr.Textbox(label="Text", lines=12)
                                save_prompt_btn = gr.Button("💾 Speichern", variant="primary")
                                save_prompt_status = gr.Markdown("")

                            with gr.Column(scale=1):
                                gr.Markdown("#### 📂 Gespeicherte Vorlagen")
                                saved_prompts = gr.Dataframe(
                                    headers=["ID", "Name", "Kategorie"],
                                    value=[[None, "", ""]] * 6, # Skeleton
                                    interactive=False,
                                    wrap=True,
                                    height=300,
                                    datatype=["number", "str", "str"],
                                    column_widths=["15%", "50%", "35%"]
                                )

                                with gr.Row(variant="panel", equal_height=True):
                                    with gr.Column(scale=1):
                                        prompt_id_load = gr.Number(label="ID", precision=0, minimum=0)
                                    with gr.Column(scale=0):
                                        refresh_prompts_btn = gr.Button("🔄")
                                    with gr.Column(scale=1):
                                        prompt_to_chat_btn = gr.Button("📨 An Chat", variant="secondary")
                                    with gr.Column(scale=0):
                                        delete_prompt_btn = gr.Button("🗑️", variant="stop")

                                loaded_prompt_display = gr.Textbox(label="Vorschau", lines=5)

                        # --- LOGIC ---
                        def load_prompts_data(user_state=None):
                            if not user_state or not user_state.get("id"): 
                                return pad_data([], 3), []
                            p_list = get_user_custom_prompts(user_state["id"])
                            clean = [[p.id, p.name, p.category] for p in p_list]
                            return pad_data(list(clean), 3), clean

                        def load_single_prompt(tid, user_state=None):
                            if not tid or not user_state or not user_state.get("id"): return ""
                            db = SessionLocal()
                            p = db.query(CustomPrompt).filter(CustomPrompt.id == int(tid), CustomPrompt.user_id == user_state["id"]).first()
                            db.close()
                            return p.prompt_text if p else ""

                        def select_prompt_row(evt: gr.SelectData, state_data, user_state):
                            try:
                                if not user_state or not user_state.get("id"):
                                    return 0, "❌"

                                if state_data and evt.index[0] < len(state_data):
                                    tid = int(state_data[evt.index[0]][0])
                                    return tid, load_single_prompt(tid, user_state)
                            except: pass
                            return 0, ""

                        def save_p(n, c, t, user_state=None):
                            if not user_state or not user_state.get("id"):
                                return "❌ Auth Fehler", [], []
                            
                            save_custom_prompt(user_state["id"], n, t, c.lower())
                            d, s = load_prompts_data(user_state)
                            return "✅ Gespeichert", d, s

                        def del_p(tid, user_state=None):
                            if not user_state or not user_state.get("id"):
                                return "❌ Auth Fehler", [], []

                            delete_custom_prompt(int(tid or 0), user_state["id"])
                            d, s = load_prompts_data(user_state)
                            return "", d, s

                        save_prompt_btn.click(save_p, inputs=[new_prompt_name, new_prompt_category, new_prompt_text, session_state], outputs=[save_prompt_status, saved_prompts, prompt_state])
                        refresh_prompts_btn.click(load_prompts_data, inputs=[session_state], outputs=[saved_prompts, prompt_state])
                        prompts_tab.select(load_prompts_data, inputs=[session_state], outputs=[saved_prompts, prompt_state])

                        prompt_id_load.change(load_single_prompt, inputs=[prompt_id_load, session_state], outputs=loaded_prompt_display)
                        delete_prompt_btn.click(del_p, inputs=[prompt_id_load, session_state], outputs=[loaded_prompt_display, saved_prompts, prompt_state])
                        saved_prompts.select(
                            select_prompt_row, 
                            inputs=[prompt_state, session_state],
                            outputs=[prompt_id_load, loaded_prompt_display]
                        )
                                                
                        if 'msg_input' in locals():
                            prompt_to_chat_btn.click(
                                lambda x: x, 
                                inputs=loaded_prompt_display, 
                                outputs=c_msg
                            )

                    
                    # =========================================================
                    # 4. RESUME FAILED JOBS
                    # =========================================================
                    with gr.TabItem("🔄 Abgebrochene Uploads") as jobs_tab:
                        gr.Markdown("### 🚧 Unvollständige Transkriptionen fortsetzen")
                        
                        failed_jobs_table = gr.Dataframe(
                            headers=["Job ID", "Datum", "Status", "Provider"],
                            value=[["", "", "", ""]],
                            interactive=False,
                            label="Offene Jobs"
                        )
                        
                        with gr.Row():
                            refresh_jobs_btn = gr.Button("🔄 Liste aktualisieren")
                            resume_job_id_input = gr.Number(label="Job ID zum Fortsetzen", precision=0)
                            resume_btn = gr.Button("▶️ Job Fortsetzen", variant="primary")
                        
                        resume_log = gr.Textbox(label="Resume Log", lines=5)
                        resume_result = gr.Textbox(label="Ergebnis", lines=10)

                        # Logic
                        def list_failed_jobs():
                            data = get_failed_jobs()
                            return data if data else [["Keine offenen Jobs", "", "", ""]]

                        def resume_job_process(jid):
                            """Resume logic"""
                            try:
                                jid = int(jid)
                                path = os.path.join(JOB_STATE_DIR, f"{jid}.json")
                                if not os.path.exists(path): return "❌ Job nicht gefunden", ""
                                
                                with open(path, "r") as f:
                                    job = json.load(f)
                                
                                client = get_client(job["provider"])
                                chunk_paths = [c["path"] for c in job["chunks"]]
                                
                                logs = f"🚀 Setze Job {jid} fort..."
                                full_text = ""
                                
                                transcriber = run_chunked_api_transcription(
                                    client, job["model"], chunk_paths, job["lang"], 
                                    job["prompt"], job["temp"], job_id=jid
                                )
                                
                                for update in transcriber:
                                    if len(update) < 300:
                                        logs += f"\n{update}"
                                        yield logs, full_text
                                    else:
                                        full_text = update
                                        
                                yield logs + "\n🎉 Job abgeschlossen!", full_text
                                
                            except Exception as e:
                                yield f"🔥 Fehler: {e}", ""

                        # Wiring
                        refresh_jobs_btn.click(list_failed_jobs, outputs=failed_jobs_table)
                        
                        # --- FIX: AUTO-LOAD ON TAB SELECT ---
                        jobs_tab.select(list_failed_jobs, outputs=failed_jobs_table)

                        resume_btn.click(resume_job_process, inputs=resume_job_id_input, outputs=[resume_log, resume_result])
                        
                        # Auto-fill ID on click
                        def select_job(evt: gr.SelectData, data):
                            try: return int(data.iloc[evt.index[0], 0]) 
                            except: return 0
                        
                        failed_jobs_table.select(select_job, failed_jobs_table, resume_job_id_input)

                    # --- TAB 7: MODEL PREFERENCES ---
                    with gr.TabItem("🎯 Modell-Einstellungen"):
                        gr.Markdown("## 🎯 Bevorzugte Modelle verwalten")
                        gr.Markdown("Wähle welche Modelle in den Dropdown-Menüs erscheinen sollen und lege die Reihenfolge fest.")
                        
                        with gr.Row():
                            pref_provider = gr.Dropdown(
                                choices=list(PROVIDERS.keys()),
                                value="Scaleway",
                                label="Provider auswählen"
                            )
                        
                        with gr.Row():
                            fetch_models_btn = gr.Button("🔄 Verfügbare Modelle laden", variant="secondary")
                            fetch_status = gr.Markdown("")
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("### 📋 Verfügbare Modelle")
                                available_models_list = gr.Dataframe(
                                    headers=["Modell-ID", "Anzeigename"],
                                    value=[["", ""]],
                                    interactive=False,
                                    wrap=True,
                                    height=400
                                )
                            
                            with gr.Column(scale=1):
                                gr.Markdown("### ✅ Deine Auswahl")
                                gr.Markdown("*Erste Modell = Standard. Drag & Drop zum Sortieren.*")
                                
                                selected_models_state = gr.State([])
                                
                                selected_models_display = gr.Dataframe(
                                    headers=["Reihenfolge", "Modell-ID", "Anzeigename", "Sichtbar"],
                                    value=[["", "", "", ""]],
                                    interactive=False,
                                    wrap=True,
                                    height=400,
                                    datatype=["number", "str", "str", "bool"]
                                )
                                
                                with gr.Row():
                                    save_prefs_btn = gr.Button("💾 Einstellungen speichern", variant="primary")
                                    reset_prefs_btn = gr.Button("🔄 Zurücksetzen", variant="secondary")
                                
                                save_prefs_status = gr.Markdown("")
                        
                        with gr.Accordion("⚙️ Modell-Verwaltung", open=True):
                            with gr.Row():
                                with gr.Column():
                                    model_to_add = gr.Textbox(
                                        label="Modell-ID hinzufügen",
                                        placeholder="z.B. llama-3.3-70b-instruct"
                                    )
                                    model_display_name = gr.Textbox(
                                        label="Anzeigename (optional)",
                                        placeholder="z.B. Llama 3.3 70B"
                                    )
                                    add_model_btn = gr.Button("➕ Hinzufügen", variant="secondary")
                                
                                with gr.Column():
                                    model_to_remove_idx = gr.Number(
                                        label="Position zum Entfernen (Reihenfolge-Nummer)",
                                        precision=0,
                                        value=1
                                    )
                                    remove_model_btn = gr.Button("➖ Entfernen", variant="secondary")
                                
                                with gr.Column():
                                    move_from_idx = gr.Number(
                                        label="Von Position",
                                        precision=0,
                                        value=1
                                    )
                                    move_to_idx = gr.Number(
                                        label="Zu Position",
                                        precision=0,
                                        value=2
                                    )
                                    move_model_btn = gr.Button("↕️ Verschieben", variant="secondary")
                            
                            model_mgmt_status = gr.Markdown("")
                        
                        gr.Markdown("""
                        ### 💡 Tipps
                        
                        - **Erstes Modell** = Standard-Modell für diesen Provider
                        - **Unsichtbare Modelle** werden nicht in Dropdown-Menüs angezeigt
                        - Klicke auf "🔄 Verfügbare Modelle laden" um die neuesten Modelle vom Provider zu laden
                        - Änderungen gelten sofort nach dem Speichern
                        """)
                        
                        # =========================================================
                        # EVENT HANDLERS FOR MODEL PREFERENCES
                        # =========================================================
                        
                        def fetch_models_for_provider(provider, user_state=None):
                            """Fetch and display available models"""
                            if not user_state or not user_state.get("id"):
                                return [["", ""]], "❌ Bitte anmelden", gr.update()
                            
                            provider_key = API_KEYS.get(provider.lower(), "")
                            models, error = fetch_available_models(provider, provider_key)
                            
                            if error: return [["", ""]], f"❌ {error}", gr.update()
                            if not models: return [["", ""]], "⚠️ Keine Modelle gefunden", gr.update()
                            
                            model_data = [[m["id"], m.get("name", m["id"])] for m in models]
                            return model_data, f"✅ {len(models)} Modelle geladen", gr.update()

                        def load_user_preferences(provider, user_state=None):
                            """Load user's saved preferences for provider"""
                            if not user_state or not user_state.get("id"):
                                return [["", "", "", ""]], []
                            
                            prefs = get_user_model_preferences(user_state["id"], provider)
                            
                            if not prefs:
                                # Defaults
                                default_models = PROVIDERS.get(provider, {}).get("chat_models", [])
                                display_data = []
                                state_data = []
                                for i, model_id in enumerate(default_models, 1):
                                    display_data.append([i, model_id, model_id, True])
                                    state_data.append({
                                        "model_id": model_id,
                                        "display_name": model_id,
                                        "is_visible": True,
                                        "display_order": i
                                    })
                                return display_data, state_data
                            
                            display_data = []
                            state_data = []
                            for i, pref in enumerate(prefs, 1):
                                display_data.append([i, pref.model_id, pref.display_name or pref.model_id, "✅" if pref.is_visible else "❌"])
                                state_data.append({
                                    "model_id": pref.model_id,
                                    "display_name": pref.display_name or pref.model_id,
                                    "is_visible": pref.is_visible,
                                    "display_order": i
                                })
                            return display_data, state_data
                        
                        def add_model_to_selection(provider, model_id, display_name, current_state):
                            """Add a model to user's selection"""
                            if not model_id:
                                return gr.update(), current_state, "❌ Modell-ID erforderlich"
                            
                            # Check if already exists
                            if any(m["model_id"] == model_id for m in current_state):
                                return gr.update(), current_state, "⚠️ Modell bereits in Auswahl"
                            
                            # Add to state
                            new_model = {
                                "model_id": model_id,
                                "display_name": display_name or model_id,
                                "is_visible": True,
                                "display_order": len(current_state) + 1
                            }
                            current_state.append(new_model)
                            
                            # Update display
                            display_data = []
                            for i, m in enumerate(current_state, 1):
                                display_data.append([
                                    i,
                                    m["model_id"],
                                    m["display_name"],
                                    "✅" if m["is_visible"] else "❌"
                                ])
                            
                            return display_data, current_state, f"✅ '{model_id}' hinzugefügt"
                        
                        def remove_model_from_selection(idx, current_state):
                            """Remove a model from selection"""
                            if not current_state or idx < 1 or idx > len(current_state):
                                return gr.update(), current_state, "❌ Ungültige Position"
                            
                            removed = current_state.pop(idx - 1)
                            
                            # Update display
                            display_data = []
                            for i, m in enumerate(current_state, 1):
                                display_data.append([
                                    i,
                                    m["model_id"],
                                    m["display_name"],
                                    "✅" if m["is_visible"] else "❌"
                                ])
                            
                            return display_data, current_state, f"✅ '{removed['model_id']}' entfernt"
                        
                        def move_model_in_selection(from_idx, to_idx, current_state):
                            """Move a model in the selection order"""
                            if not current_state:
                                return gr.update(), current_state, "❌ Keine Modelle vorhanden"
                            
                            if from_idx < 1 or from_idx > len(current_state):
                                return gr.update(), current_state, "❌ Ungültige Ausgangsposition"
                            
                            if to_idx < 1 or to_idx > len(current_state):
                                return gr.update(), current_state, "❌ Ungültige Zielposition"
                            
                            # Move item
                            item = current_state.pop(from_idx - 1)
                            current_state.insert(to_idx - 1, item)
                            
                            # Update display
                            display_data = []
                            for i, m in enumerate(current_state, 1):
                                display_data.append([
                                    i,
                                    m["model_id"],
                                    m["display_name"],
                                    "✅" if m["is_visible"] else "❌"
                                ])
                            
                            return display_data, current_state, f"✅ Modell von Position {from_idx} zu {to_idx} verschoben"
                        
                        def save_preferences(provider, current_state, user_state=None):
                            if not user_state or not user_state.get("id"):
                                return "❌ Bitte anmelden"
                            
                            if not current_state:
                                return "⚠️ Keine Modelle ausgewählt"
                            
                            # Update display_order
                            for i, model in enumerate(current_state):
                                model["display_order"] = i
                            
                            success, message = save_user_model_preferences(user_state["id"], provider, current_state)
                            return message
                        
                        def reset_to_defaults(provider):
                            """Reset to default provider models"""
                            default_models = PROVIDERS.get(provider, {}).get("chat_models", [])
                            
                            display_data = []
                            state_data = []
                            for i, model_id in enumerate(default_models, 1):
                                display_data.append([i, model_id, model_id, "✅"])
                                state_data.append({
                                    "model_id": model_id,
                                    "display_name": model_id,
                                    "is_visible": True,
                                    "display_order": i
                                })
                            
                            return display_data, state_data, "✅ Auf Standard zurückgesetzt"
                        
                        # Wire up event handlers
                        fetch_models_btn.click(
                            fetch_models_for_provider,
                            inputs=[pref_provider, session_state],
                            outputs=[available_models_list, fetch_status, selected_models_display]
                        )

                        pref_provider.change(
                            load_user_preferences,
                            inputs=[pref_provider, session_state],
                            outputs=[selected_models_display, selected_models_state]
                        )
                        
                        add_model_btn.click(
                            add_model_to_selection,
                            inputs=[pref_provider, model_to_add, model_display_name, selected_models_state],
                            outputs=[selected_models_display, selected_models_state, model_mgmt_status]
                        )
                        
                        remove_model_btn.click(
                            remove_model_from_selection,
                            inputs=[model_to_remove_idx, selected_models_state],
                            outputs=[selected_models_display, selected_models_state, model_mgmt_status]
                        )
                        
                        move_model_btn.click(
                            move_model_in_selection,
                            inputs=[move_from_idx, move_to_idx, selected_models_state],
                            outputs=[selected_models_display, selected_models_state, model_mgmt_status]
                        )
                        
                        save_prefs_btn.click(
                            save_preferences,
                            inputs=[pref_provider, selected_models_state, session_state],
                            outputs=[save_prefs_status]
                        )
                        
                        reset_prefs_btn.click(
                            reset_to_defaults,
                            inputs=[pref_provider],
                            outputs=[selected_models_display, selected_models_state, save_prefs_status]
                        )
                        
            # --- TAB 6: USER MANAGEMENT (ADMIN ONLY) ---
            with gr.TabItem("👥 Benutzerverwaltung", visible=False) as admin_tab:
                gr.Markdown("## 👥 Benutzerverwaltung")
                gr.Markdown("*Nur für Administratoren*")
                
                with gr.Tabs():
                    # =========================================================
                    # CREATE NEW USER
                    # =========================================================
                    with gr.TabItem("➕ Neuer Benutzer"):
                        gr.Markdown("### Neuen Benutzer erstellen")
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                new_user_username = gr.Textbox(
                                    label="Benutzername",
                                    placeholder="max.mustermann"
                                )
                                new_user_password = gr.Textbox(
                                    label="Passwort",
                                    placeholder="Mindestens 8 Zeichen",
                                    type="password"
                                )
                                new_user_password_confirm = gr.Textbox(
                                    label="Passwort bestätigen",
                                    placeholder="Passwort wiederholen",
                                    type="password"
                                )
                                new_user_email = gr.Textbox(
                                    label="E-Mail (Optional)",
                                    placeholder="max@example.com"
                                )
                                new_user_is_admin = gr.Checkbox(
                                    label="Als Administrator erstellen",
                                    value=False
                                )
                                
                                create_user_btn = gr.Button("➕ Benutzer erstellen", variant="primary", size="lg")
                                create_user_status = gr.Markdown("")
                            
                            with gr.Column(scale=1):
                                gr.Markdown("""
                                ### 📋 Hinweise
                                
                                **Benutzername:**
                                - Muss eindeutig sein
                                - Keine Leerzeichen
                                - Empfohlen: kleinbuchstaben
                                
                                **Passwort:**
                                - Mindestens 8 Zeichen empfohlen
                                - Wird sicher verschlüsselt (bcrypt)
                                
                                **Administrator:**
                                - Admins können:
                                - Alle Benutzer verwalten
                                - Andere Admins erstellen
                                - Alle Daten sehen

                                """)
                    
                    # =========================================================
                    # MANAGE EXISTING USERS
                    # =========================================================
                    with gr.TabItem("⚙️ Benutzer verwalten"):
                        gr.Markdown("### Bestehende Benutzer verwalten")
                        
                        with gr.Row():
                            refresh_users_btn = gr.Button("🔄 Liste aktualisieren", size="sm")
                        
                        users_table = gr.Dataframe(
                            headers=["ID", "Benutzername", "E-Mail", "Rolle", "Erstellt"],
                            value=[["", "", "", "", ""]],
                            interactive=False,
                            wrap=True,
                            height=400,
                            datatype=["number", "str", "str", "str", "str"],
                            column_widths=["10%", "25%", "25%", "20%", "20%"]
                        )
                        
                        with gr.Row():
                            selected_user_id = gr.Number(
                                label="Benutzer-ID",
                                precision=0,
                                value=0
                            )
                        
                        with gr.Tabs():
                            # RENAME USER
                            with gr.TabItem("✏️ Umbenennen"):
                                with gr.Row():
                                    rename_new_username = gr.Textbox(
                                        label="Neuer Benutzername",
                                        placeholder="neuer.name"
                                    )
                                    rename_user_btn = gr.Button("✏️ Umbenennen", variant="secondary")
                                rename_status = gr.Markdown("")
                            
                            # RESET PASSWORD
                            with gr.TabItem("🔑 Passwort zurücksetzen"):
                                with gr.Row():
                                    reset_new_password = gr.Textbox(
                                        label="Neues Passwort",
                                        placeholder="Neues Passwort",
                                        type="password"
                                    )
                                    reset_confirm_password = gr.Textbox(
                                        label="Passwort bestätigen",
                                        placeholder="Passwort wiederholen",
                                        type="password"
                                    )
                                reset_password_btn = gr.Button("🔑 Passwort zurücksetzen", variant="secondary")
                                reset_password_status = gr.Markdown("")
                            
                            # UPDATE EMAIL
                            with gr.TabItem("📧 E-Mail ändern"):
                                with gr.Row():
                                    update_email_input = gr.Textbox(
                                        label="Neue E-Mail",
                                        placeholder="neue@email.de"
                                    )
                                    update_email_btn = gr.Button("📧 E-Mail aktualisieren", variant="secondary")
                                update_email_status = gr.Markdown("")
                            
                            # TOGGLE ADMIN
                            with gr.TabItem("⬆️⬇️ Admin-Status"):
                                gr.Markdown("""
                                ### Admin-Status umschalten
                                
                                **Achtung:** 
                                - Du kannst deinen eigenen Status nicht ändern
                                - Admins haben volle Kontrolle über die App
                                """)
                                toggle_admin_btn = gr.Button("⬆️⬇️ Admin-Status umschalten", variant="secondary")
                                toggle_admin_status = gr.Markdown("")
                            
                            # DELETE USER
                            with gr.TabItem("🗑️ Benutzer löschen"):
                                gr.Markdown("""
                                ### ⚠️ WARNUNG: Benutzer löschen
                                
                                Diese Aktion:
                                - Löscht den Benutzer **permanent**
                                - Löscht alle zugehörigen Daten (Chats, Transkripte, etc.)
                                - **Kann nicht rückgängig gemacht werden**
                                - Du kannst dich nicht selbst löschen
                                """)
                                
                                with gr.Row():
                                    delete_confirm = gr.Textbox(
                                        label="Bestätigung",
                                        placeholder="Tippe 'LÖSCHEN' zur Bestätigung"
                                    )
                                    delete_user_btn = gr.Button("🗑️ BENUTZER LÖSCHEN", variant="stop")
                                delete_user_status = gr.Markdown("")
                        
                        # Select user from table
                        def select_user_from_table(evt: gr.SelectData):
                            """Select user by clicking on table row"""
                            try:
                                # Get the row data
                                row_idx = evt.index[0]
                                # Load fresh data
                                users_data = get_all_users()
                                if row_idx < len(users_data):
                                    user_id = users_data[row_idx][0]
                                    return int(user_id)
                            except Exception as e:
                                logger.exception(f"Error selecting user: {str(e)}")
                            return gr.update()
                        
                        users_table.select(select_user_from_table, outputs=selected_user_id)
                
                # =========================================================
                # EVENT HANDLERS
                # =========================================================
                
                # Create user
                def handle_create_user(username, password, password_confirm, email, is_admin):
                    if not username or not password:
                        return "❌ Benutzername und Passwort sind erforderlich"
                    
                    if password != password_confirm:
                        return "❌ Passwörter stimmen nicht überein"
                    
                    if len(password) < 8:
                        return "⚠️ Warnung: Passwort sollte mindestens 8 Zeichen haben"
                    
                    success, message = create_user(username, password, email, is_admin)
                    
                    if success:
                        # Clear form
                        return message
                    return message
                
                create_user_btn.click(
                    handle_create_user,
                    inputs=[
                        new_user_username,
                        new_user_password,
                        new_user_password_confirm,
                        new_user_email,
                        new_user_is_admin
                    ],
                    outputs=create_user_status
                )
                
                # Refresh users list
                def load_users_list(user_state=None):
                    if not user_state or not user_state.get("is_admin"):
                        return []
                    return get_all_users()
                        
                refresh_users_btn.click(
                    load_users_list, 
                    inputs=[session_state],
                    outputs=users_table
                )
                
                # Rename user
                def handle_rename_user(user_id, new_username):
                    if not new_username:
                        return "❌ Neuer Benutzername erforderlich", get_all_users()
                    
                    success, message = rename_user(int(user_id), new_username)
                    return message, get_all_users()
                
                rename_user_btn.click(
                    handle_rename_user,
                    inputs=[selected_user_id, rename_new_username],
                    outputs=[rename_status, users_table]
                )
                
                # Reset password
                def handle_reset_password(user_id, new_password, confirm_password):
                    if not new_password:
                        return "❌ Neues Passwort erforderlich"
                    
                    if new_password != confirm_password:
                        return "❌ Passwörter stimmen nicht überein"
                    
                    if len(new_password) < 8:
                        return "⚠️ Warnung: Passwort sollte mindestens 8 Zeichen haben"
                    
                    success, message = reset_user_password(int(user_id), new_password)
                    return message
                
                reset_password_btn.click(
                    handle_reset_password,
                    inputs=[selected_user_id, reset_new_password, reset_confirm_password],
                    outputs=reset_password_status
                )
                
                # Update email
                def handle_update_email(user_id, new_email):
                    success, message = update_user_email(int(user_id), new_email)
                    return message, get_all_users()
                
                update_email_btn.click(
                    handle_update_email,
                    inputs=[selected_user_id, update_email_input],
                    outputs=[update_email_status, users_table]
                )
                
                # Toggle admin status
                def handle_toggle_admin(user_id, user_state=None):
                    if not user_state or not user_state.get("id"):
                        return "❌ Nicht angemeldet", get_all_users()
                    
                    # Check if requesting user is actually admin
                    if not user_state.get("is_admin"):
                        return "⛔ Keine Berechtigung", get_all_users()

                    success, message = toggle_admin_status(int(user_id), user_state["id"])
                    return message, get_all_users()
                
                toggle_admin_btn.click(
                    handle_toggle_admin,
                    inputs=[selected_user_id, session_state],
                    outputs=[toggle_admin_status, users_table]
                )
                
                # Delete user
                def handle_delete_user(user_id, confirmation, user_state=None):
                    if confirmation != "LÖSCHEN":
                        return "❌ Bestätigung erforderlich: Tippe 'LÖSCHEN'", get_all_users()
                    
                    if not user_state or not user_state.get("id"):
                        return "❌ Nicht angemeldet", get_all_users()

                    if not user_state.get("is_admin"):
                        return "⛔ Keine Berechtigung", get_all_users()
                    
                    success, message = delete_user(int(user_id), user_state["id"])
                    return message, get_all_users()
                
                delete_user_btn.click(
                    handle_delete_user,
                    inputs=[selected_user_id, delete_confirm, session_state],
                    outputs=[delete_user_status, users_table]
                )

                # --- AUTO-LOAD ON TAB SWITCH ---
                trans_tab.select(fn=load_trans_data, outputs=[trans_history, trans_state])
                images_tab.select(fn=load_img_data, outputs=[images_history, img_state])
                prompts_tab.select(fn=load_prompts_data, outputs=[saved_prompts, prompt_state])
                
            

    def handle_login(username, password):
        # This logic creates the initial state
        success, message, show_app, show_login, state_data = login_user(username, password)
        
        status_text = f"👤 Angemeldet als: **{state_data['username']}**" if success else "👤 Nicht angemeldet"
        show_admin_tab = state_data.get("is_admin", False)
        
        if success:
            try:
                ensure_user_storage_dirs(username)
            except Exception as e:
                logger.error(f"Could not create storage dirs: {e}")
        
        return message, show_app, show_login, status_text, gr.update(visible=True), gr.update(visible=show_admin_tab), state_data
    
    def handle_logout():
        message, show_app, show_login, empty_state = logout_user()
        return message, show_app, show_login, "👤 Nicht angemeldet", gr.update(visible=False), gr.update(visible=False), empty_state
    
    login_btn.click(
        handle_login,
        inputs=[login_username, login_password],
        outputs=[login_message, main_app, login_screen, login_status, logout_btn, admin_tab, session_state]
    )

    logout_btn.click(
        handle_logout,
        outputs=[login_message, main_app, login_screen, login_status, logout_btn, admin_tab, session_state]
    )

# ==========================================
# 🚀 LAUNCH CONFIGURATION
# ==========================================
if __name__ == "__main__":
    from fastapi import FastAPI, Request, status
    from fastapi.responses import JSONResponse
    import uvicorn

    # 1. Configuration Constants
    APP_DIR = "/var/www/transkript_app"
    LOG_FILE = os.path.join(APP_DIR, "app.log")
    STATIC_DIR = os.path.join(APP_DIR, "static")
    IMAGES_DIR = os.path.join(APP_DIR, "generated_images")

    # 2. Ensure directories and permissions exist
    os.makedirs(IMAGES_DIR, exist_ok=True)
    
    if not os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, 'w') as f: f.write("")
            os.chmod(LOG_FILE, 0o666) 
        except Exception as e:
            print(f"⚠️ Could not create log file: {e}")

    # 3. Define FastAPI Wrapper for Security
    app = FastAPI()

    @app.middleware("http")
    async def block_api_endpoints(request: Request, call_next):
        # Allow internal UI paths (/run, /queue), block external API access
        if request.url.path.startswith("/api"):
             return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"detail": "External API access is disabled."}
            )
        response = await call_next(request)
        return response

    print(f"🚀 Starting Server on Port 7860 (Fast Shutdown Enabled)...")
    print(f"📂 Serving files from: {APP_DIR}")

    # 4. Mount Gradio
    app = gr.mount_gradio_app(
        app, 
        demo, 
        path="/", 
        allowed_paths=[APP_DIR, STATIC_DIR, IMAGES_DIR, "/tmp/gradio"],
    )

    # 5. Run with Timeout Configuration
    # timeout_graceful_shutdown=1 forces the server to kill connections 
    # and release DB locks instantly when you run `systemctl restart`
    config = uvicorn.Config(
        app, 
        host="0.0.0.0", 
        port=7860, 
        timeout_graceful_shutdown=1, # <--- THE FIX
        log_level="info"
    )
    server = uvicorn.Server(config)
    server.run()
