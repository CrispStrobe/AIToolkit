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
    """Delete a transcription"""
    db = get_db()
    trans = db.query(Transcription).filter(
        Transcription.id == trans_id,
        Transcription.user_id == user_id
    ).first()
    if trans:
        db.delete(trans)
        db.commit()
        db.close()
        return True
    db.close()
    return False

def delete_chat_history(chat_id: int, user_id: int):
    """Delete a chat history"""
    db = get_db()
    chat = db.query(ChatHistory).filter(
        ChatHistory.id == chat_id,
        ChatHistory.user_id == user_id
    ).first()
    if chat:
        db.delete(chat)
        db.commit()
        db.close()
        return True
    db.close()
    return False

def delete_vision_result(vision_id: int, user_id: int):
    """Delete a vision result"""
    db = get_db()
    vision = db.query(VisionResult).filter(
        VisionResult.id == vision_id,
        VisionResult.user_id == user_id
    ).first()
    if vision:
        db.delete(vision)
        db.commit()
        db.close()
        return True
    db.close()
    return False

def delete_generated_image(img_id: int, user_id: int):
    """Delete a generated image"""
    db = get_db()
    img = db.query(GeneratedImage).filter(
        GeneratedImage.id == img_id,
        GeneratedImage.user_id == user_id
    ).first()
    if img:
        # Also delete the file
        if img.image_path and os.path.exists(img.image_path):
            os.remove(img.image_path)
        db.delete(img)
        db.commit()
        db.close()
        return True
    db.close()
    return False

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
# AUDIO HELPERS
# ==========================================

import shutil

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
    Splits audio into segments using PyDub to bypass API file size limits.
    Returns: (List of file paths, temp_directory_path)
    """
    try:
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
            
            # Export (128k bitrate is good balance for API size vs quality)
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
        # Prevent self-deletion
        if user_id == current_user_id:
            return False, "❌ Du kannst dich nicht selbst löschen"
        
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return False, "❌ Benutzer nicht gefunden"
        
        username = user.username
        
        # Delete user (cascades will handle related data)
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

# Fügen Sie diese Funktion NACH den PROVIDERS Definitionen ein:

def get_compliance_html(provider):
    """Gibt HTML-Badge für DSGVO-Compliance zurück"""
    badges = {
        "Scaleway": "🇫🇷 <b>DSGVO-Konform</b> (Frankreich)",
        "Nebius": "🇪🇺 <b>DSGVO-Konform</b> (EU-Rechenzentren)",
        "Mistral": "🇫🇷 <b>DSGVO-Konform</b> (Frankreich)",
        "Gladia": "🇫🇷 <b>DSGVO-Konform</b> (Frankreich, Beste Audio-Qualität)",
        "OpenRouter": "🇺🇸 <b>US-Server</b> (Nicht DSGVO)",
        "Groq": "🇺🇸 <b>US-Server</b> (Schnell, Kostenlos)"
    }
    return f'<div style="background:#e3f2fd;padding:10px;border-radius:8px;margin:10px 0;">{badges.get(provider, "Unbekannt")}</div>'

# Gladia Spezial-Config
GLADIA_CONFIG = {
    "url": "https://api.gladia.io/v2",
    "vocab": [
        "Christian Ströbele", "Jesus Christus", "Amen", "Halleluja",
        "Evangelium", "Predigt", "Liturgie", "Gottesdienst", "Pfarrei",
        "Diözese", "Kirchenvorstand", "Fürbitten", "Akademie",
        "Tagungshaus", "Compliance", "Synode", "Ökumene"
    ]
}


# Global variable to track logged-in user
current_user = {"id": None, "username": None, "is_admin": False}

def login_user(username, password):
    """Login function"""
    user = authenticate_user(username, password)
    if user:
        current_user["id"] = user.id
        current_user["username"] = user.username
        current_user["is_admin"] = user.is_admin
        return True, f"✅ Willkommen, {user.username}!", gr.update(visible=True), gr.update(visible=False)
    return False, "❌ Ungültige Anmeldedaten", gr.update(visible=False), gr.update(visible=True)

def logout_user():
    """Logout function"""
    username = current_user["username"]
    current_user["id"] = None
    current_user["username"] = None
    current_user["is_admin"] = False
    return f"👋 Auf Wiedersehen, {username}!", gr.update(visible=False), gr.update(visible=True)

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

# ==========================================
# 1. CHAT LOGIK
# ==========================================

def run_chat(message, history, provider, model, temp, system_prompt, key):
    try:
        client = get_client(provider, key)
        
        # 1. Build Messages
        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": str(system_prompt)})
            
        for msg in history:
            # Ensure content is string and not None to prevent JSON errors
            content = str(msg["content"]) if msg.get("content") else ""
            messages.append({"role": msg["role"], "content": content})
            
        messages.append({"role": "user", "content": str(message)})
        
        # 2. Configure Stream
        stream_params = {
            "model": model,
            "messages": messages,
            "temperature": float(temp),
            "stream": True
            # REMOVED: "stream_options" (Causes 400 Error with Poe)
        }

        # 3. Execute
        stream = client.chat.completions.create(**stream_params)
        
        full_response = ""
        for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta.content:
                    full_response += delta.content
                    yield full_response
                
    except Exception as e:
        logger.exception(f"Chat error with {provider}: {str(e)}")
        # Show a friendly error in the chat window
        yield f"🔥 Fehler ({provider}): {str(e)}"
        
# ==========================================
# 2. VISION LOGIK
# ==========================================

def run_vision(image, prompt, provider, model, key):
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

def run_chunked_api_transcription(client, model, chunk_paths, lang, prompt, temp):
    """
    Iterates through chunks, calls API, and stitches results.
    Handles 'prompt' context injection to maintain consistency.
    """
    full_transcript = ""
    total = len(chunk_paths)
    
    for i, chunk_path in enumerate(chunk_paths):
        step = i + 1
        yield f"⏳ Verarbeite Teil {step}/{total}..."
        
        try:
            with open(chunk_path, "rb") as f:
                # Build API args
                args = {
                    "model": model,
                    "file": f,
                    "response_format": "json"
                }
                
                if lang and lang != "auto":
                    args["language"] = lang
                
                if temp is not None:
                    args["temperature"] = float(temp)
                    
                # Context Logic:
                # 1. First chunk gets the user's manual prompt (for vocab/style).
                # 2. Subsequent chunks get the last 200 chars of previous text 
                #    to help Whisper maintain context across cuts.
                if i == 0 and prompt:
                    args["prompt"] = prompt
                elif i > 0 and len(full_transcript) > 0:
                    # Provide previous context to link sentences across splits
                    prev_context = full_transcript[-220:]
                    args["prompt"] = prev_context

                # Call API
                logger.info(f"Transcribing chunk {step}/{total} ({os.path.getsize(chunk_path)/1024/1024:.2f}MB)")
                
                # Check provider type for correct endpoint call
                # (Mistral/Groq/Scaleway/Nebius all adhere to this standard)
                resp = client.audio.transcriptions.create(**args)
                
                # Extract Text
                text_part = resp.text if hasattr(resp, 'text') else str(resp)
                
                # Append with space
                full_transcript += text_part + " "
                
                # Yield progress + current result state
                yield f"✅ Teil {step}/{total} erledigt."
        
        except Exception as e:
            logger.error(f"Error in chunk {step}: {e}")
            yield f"⚠️ Fehler in Teil {step}: {str(e)}"
            # Recover: Add placeholder and continue to next chunk
            full_transcript += f" [Fehlerhafter Teil {step}] "

    yield full_transcript.strip()
        
def run_transcription(audio, provider, model, lang, whisper_temp, whisper_prompt, diar, trans, target, key):
    """
    Unified transcription router.
    - Gladia: Uses Native V2 API (good for long files, keeps speaker ID).
    - Others (Scaleway, Groq, Mistral, Nebius): Uses Local Chunking -> API (bypasses 25MB limits).
    """
    logger.info(f"TRANSCRIPTION START: {provider} | Model: {model} | File: {audio}")

    # Basic validation
    if not audio:
        logger.error("No audio file provided")
        yield "❌ Keine Datei.", "", ""
        return

    # Check file exists
    if not os.path.exists(audio):
        logger.error(f"Audio file does not exist: {audio}")
        yield f"❌ Datei nicht gefunden: {audio}", "", ""
        return

    # Check file size
    try:
        file_size = os.path.getsize(audio)
        logger.info(f"Audio file size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")

        if file_size == 0:
            logger.error("Audio file is empty (0 bytes)")
            yield "❌ Audiodatei ist leer (0 Bytes)", "", ""
            return

        # Warning for large files (though chunking handles them now)
        if file_size > 100 * 1024 * 1024:  # 100MB
            logger.warning(f"Large audio file: {file_size/1024/1024:.2f} MB")
    except Exception as size_error:
        logger.exception(f"Error checking file size: {str(size_error)}")

    # ==========================================
    # PATH A: GENERIC CHUNKING (Mistral, Scaleway, Groq, etc.)
    # ==========================================
    if provider != "Gladia":
        logger.info(f"Using Generic Chunking for provider: {provider}")
        
        # 1. Setup Client
        try:
            client = get_client(provider, key)
            logger.info(f"API client created successfully for {provider}")
        except Exception as e:
            logger.exception(f"Failed to create API client: {str(e)}")
            yield f"🔥 Konfigurations-Fehler: {str(e)}", "", ""
            return

        # 2. Defaults if model missing
        if not model:
            conf = PROVIDERS.get(provider, {})
            model = conf.get("audio_models", ["whisper-large-v3"])[0]
            logger.info(f"No model specified, using default: {model}")

        logs = f"🚀 Starte {provider} ({model})...\n✂️ Prüfe Audio-Länge..."
        yield logs, "", ""

        chunk_dir = None
        try:
            # 3. Split Audio (10 min chunks)
            # This handles the 3-4h requirement by splitting into manageable pieces
            chunks, chunk_dir = split_audio_into_chunks(audio, chunk_minutes=10)
            
            if not chunks:
                yield "❌ Fehler beim Aufteilen der Datei.", "", ""
                return

            num_chunks = len(chunks)
            logs += f"\n📂 Audio in {num_chunks} Teile zerlegt."
            yield logs, "", ""

            # 4. Run Sequential Processing
            full_text = ""
            
            # run_chunked_api_transcription is a generator that yields status updates
            transcriber = run_chunked_api_transcription(
                client, model, chunks, lang, whisper_prompt, whisper_temp
            )

            for update in transcriber:
                # Differentiate between status update (short) and final text (long)
                # The generator yields status strings starting with emoji, OR the final text at the very end
                if len(update) < 300 and (update.startswith("⏳") or update.startswith("✅") or update.startswith("⚠️")):
                    logs += f"\n{update}"
                    yield logs, full_text, ""
                else:
                    # This is the final accumulated text from the generator
                    full_text = update

            logger.info("Chunked transcription completed successfully")
            yield logs + "\n🎉 Transkription vollständig!", full_text, "(Keine Übersetzung verfügbar)"

        except Exception as e:
            logger.exception(f"Chunking Workflow Error: {e}")
            yield logs + f"\n🔥 Abbruch: {str(e)}\nDetails: {type(e).__name__}", "", ""
        
        finally:
            # 5. Cleanup
            cleanup_chunks(chunk_dir)
            
        return

    # ==========================================
    # PATH B: GLADIA V2 (Native Long-File Support)
    # ==========================================
    else:
        logger.info("Using Gladia provider")

        # Get API key
        api_key = key if key else API_KEYS.get("GLADIA", "")

        if not api_key:
            logger.error("No Gladia API key available")
            yield "❌ Kein Gladia Key.", "", ""
            return

        logger.info(f"Gladia API key available: {api_key[:10]}...")

        logs = "🚀 Start Gladia..."
        yield logs, "", ""

        try:
            headers = {"x-gladia-key": api_key, "accept": "application/json"}
            logger.info("Gladia headers prepared")

            # Upload
            logger.info("Starting Gladia file upload...")
            fname = os.path.basename(audio)
            logger.info(f"Uploading file: {fname}")

            try:
                with open(audio, 'rb') as f:
                    logger.info(f"Sending POST to {GLADIA_CONFIG['url']}/upload")
                    r = requests.post(
                        f"{GLADIA_CONFIG['url']}/upload",
                        headers=headers,
                        files={'audio': (fname, f, 'audio/wav')},
                        timeout=300  # 5 minute timeout
                    )
                    logger.info(f"Upload response status: {r.status_code}")
                    logger.debug(f"Upload response: {r.text[:500]}")

                if r.status_code != 200:
                    logger.error(f"Upload failed with status {r.status_code}: {r.text}")
                    raise Exception(f"Upload failed (Status {r.status_code}): {r.text[:200]}")

                upload_result = r.json()
                url = upload_result.get("audio_url")

                if not url:
                    logger.error(f"No audio_url in upload response: {upload_result}")
                    raise Exception("Keine audio_url in der Antwort")

                logger.info(f"File uploaded successfully, audio_url: {url[:50]}...")

            except requests.exceptions.Timeout:
                logger.error("Upload timed out after 5 minutes")
                yield "🔥 Upload-Timeout (5 Min)", "", ""
                return
            except requests.exceptions.RequestException as req_error:
                logger.exception(f"Upload request failed: {str(req_error)}")
                yield f"🔥 Upload-Fehler: {str(req_error)}", "", ""
                return

            # Payload Config
            logger.info("Building Gladia job configuration...")
            v_list = [{"value": w} for w in GLADIA_CONFIG['vocab']]
            logger.info(f"Custom vocabulary: {len(v_list)} terms")

            payload = {
                "audio_url": url,
                "language_config": {
                    "code_switching": (lang == "auto"),
                    "languages": [] if lang == "auto" else [lang]
                },
                "diarization": diar,
                "diarization_config": {"min_speakers": 1, "max_speakers": 10} if diar else None,
                "name_consistency": True,
                "punctuation_enhanced": True,
                "custom_vocabulary": True,
                "custom_vocabulary_config": {"vocabulary": v_list},
                "translation": trans,
                "translation_config": {
                    "target_languages": [target],
                    "model": "base",
                    "match_original_utterances": True
                } if trans else None
            }

            logger.debug(f"Gladia payload keys: {list(payload.keys())}")
            logger.info(f"Diarization: {diar}, Translation: {trans}")

            # Start transcription job
            logger.info("Starting Gladia transcription job...")
            try:
                r = requests.post(
                    f"{GLADIA_CONFIG['url']}/pre-recorded",
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                logger.info(f"Job creation response status: {r.status_code}")
                logger.debug(f"Job creation response: {r.text[:500]}")

                if r.status_code != 201:
                    logger.error(f"Job creation failed with status {r.status_code}: {r.text}")
                    raise Exception(f"Job creation failed (Status {r.status_code}): {r.text[:200]}")

                job_result = r.json()
                res_url = job_result.get("result_url")

                if not res_url:
                    logger.error(f"No result_url in job response: {job_result}")
                    raise Exception("Keine result_url in der Antwort")

                logger.info(f"Job created successfully, result_url: {res_url[:50]}...")

            except requests.exceptions.RequestException as req_error:
                logger.exception(f"Job creation request failed: {str(req_error)}")
                yield f"🔥 Job-Erstellung Fehler: {str(req_error)}", "", ""
                return

            # Poll for results
            logger.info("Starting result polling loop...")
            start_t = time.time()
            last_log = 0
            poll_count = 0

            while True:
                time.sleep(2)
                poll_count += 1
                elapsed = time.time() - start_t

                if elapsed - last_log > 5:
                    logs += f"\n⏱️ {format_duration(elapsed)}... (Poll #{poll_count})"
                    logger.info(f"Polling... elapsed: {elapsed:.1f}s, poll count: {poll_count}")
                    yield logs, "", ""
                    last_log = elapsed

                if elapsed > 3600:  # 1 Hour timeout hard limit for huge files
                    logger.error(f"Timeout after {elapsed:.1f} seconds")
                    raise Exception(f"Timeout: Transkription dauert zu lange ({format_duration(elapsed)})")

                try:
                    r = requests.get(res_url, headers=headers, timeout=10)

                    if r.status_code != 200:
                        logger.warning(f"Poll returned status {r.status_code}, retrying...")
                        continue

                    data = r.json()
                    status = data.get("status", "unknown")
                    logger.debug(f"Poll #{poll_count}: status = {status}")

                    if status == "done":
                        logger.info(f"Transcription completed after {elapsed:.1f}s and {poll_count} polls")
                        break

                    if status == "error":
                        error_detail = json.dumps(data, indent=2)
                        logger.error(f"Gladia job failed with error status: {error_detail}")
                        raise Exception(f"Gladia error: {error_detail[:500]}")

                    # Still processing
                    if status not in ["done", "error", "queued", "processing"]:
                        logger.warning(f"Unknown status: {status}")

                except requests.exceptions.Timeout:
                    logger.warning(f"Poll #{poll_count} timed out, retrying...")
                    continue
                except requests.exceptions.RequestException as req_error:
                    logger.warning(f"Poll #{poll_count} request failed: {str(req_error)}, retrying...")
                    continue

            # Process results
            logger.info("Processing Gladia results...")
            res = data.get("result", {})

            if not res:
                logger.error("No 'result' key in response data")
                yield "🔥 Keine Ergebnisse in Antwort", "", ""
                return

            # Extract transcription
            logger.info("Extracting transcription text...")
            transcription_data = res.get("transcription", {})
            utterances = transcription_data.get("utterances", [])

            logger.info(f"Found {len(utterances)} utterances")

            orig = smart_format(utterances, True, True, True, diar)

            if not orig:
                orig = transcription_data.get("full_transcript", "(Kein Text)")
                logger.warning("Using full_transcript as fallback")

            logger.info(f"Original transcript length: {len(orig)} characters")

            # Extract translation
            tr_txt = ""
            if trans:
                logger.info("Extracting translation...")
                translation_data = res.get("translation", {})
                tr_list = translation_data.get("results", [])

                logger.info(f"Found {len(tr_list)} translation results")

                if tr_list:
                    tr_utterances = tr_list[0].get("utterances", [])
                    logger.info(f"Found {len(tr_utterances)} translation utterances")
                    tr_txt = smart_format(tr_utterances, True, True, True, diar)
                    logger.info(f"Translation length: {len(tr_txt)} characters")
                else:
                    tr_txt = "(Keine Übersetzung verfügbar)"
                    logger.warning("No translation results found")

            logger.info("Gladia transcription completed successfully")
            logger.info(f"Original preview: {orig[:100]}...")
            if tr_txt:
                logger.info(f"Translation preview: {tr_txt[:100]}...")

            yield logs + "\n🎉 Fertig.", orig, tr_txt

        except Exception as e:
            logger.exception(f"Gladia transcription error: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error details: {str(e)}")

            import traceback
            tb_str = traceback.format_exc()
            logger.error(f"Full traceback:\n{tb_str}")

            yield logs + f"\n🔥 FEHLER: {str(e)}\n\nTyp: {type(e).__name__}\n\nDetails in Log-Datei", "", ""

def run_and_save_transcription(audio, provider, model, lang, w_temp, w_prompt, diar, trans, target, key):
    """Run transcription and save to database"""
    try:
        logger.info(f"Starting transcription: provider={provider}, model={model}, audio={audio}")

        if not audio:
            logger.error("No audio file provided")
            yield "❌ Keine Audiodatei hochgeladen.", "", ""
            return

        if not os.path.exists(audio):
            logger.error(f"Audio file does not exist: {audio}")
            yield f"❌ Datei nicht gefunden: {audio}", "", ""
            return

        file_size = os.path.getsize(audio)
        logger.info(f"Audio file size: {file_size} bytes")

        if file_size == 0:
            logger.error("Audio file is empty")
            yield "❌ Audiodatei ist leer.", "", ""
            return

        # Run transcription
        result = None
        for result in run_transcription(audio, provider, model, lang, w_temp, w_prompt, diar, trans, target, key):
            yield result

        # Save to database after completion
        if current_user["id"] and result and len(result) > 1 and result[1]:
            logger.info("Auto-saving transcription to database...")
            filename = os.path.basename(audio) if audio else None

            try:
                trans_id = save_transcription(
                    user_id=current_user["id"],
                    provider=provider,
                    model=model or "N/A",
                    original=result[1],
                    translated=result[2] if len(result) > 2 else None,
                    language=lang,
                    filename=filename
                )
                logger.info(f"Transcription auto-saved with ID: {trans_id}")

                # Update the log to show it was saved
                updated_log = result[0] + f"\n\n💾 Automatisch gespeichert (ID: {trans_id})"
                yield (updated_log, result[1], result[2] if len(result) > 2 else "")

            except Exception as save_error:
                logger.exception(f"Error auto-saving transcription: {str(save_error)}")
                # Don't fail the whole transcription
                updated_log = result[0] + f"\n\n⚠️ Speichern fehlgeschlagen: {str(save_error)}"
                yield (updated_log, result[1], result[2] if len(result) > 2 else "")
        else:
            logger.info("Not saving transcription (user not logged in or empty result)")

    except Exception as e:
        logger.exception(f"Critical error in run_and_save_transcription: {str(e)}")
        yield f"🔥 Kritischer Fehler: {str(e)}\n\nTyp: {type(e).__name__}", "", ""

# ==========================================
# 4. BILDGENERIERUNG 
# ==========================================

def run_image_gen(prompt, provider, model, width, height, steps, key):
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
# 📱 PWA CONFIGURATION
# ==========================================
PWA_HEAD = """
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<meta name="theme-color" content="#1976d2">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="Akademie KI">

<link rel="manifest" href="/manifest.json" crossorigin="use-credentials">

<link rel="icon" type="image/png" sizes="192x192" href="/static/icon-192.png">
<link rel="icon" type="image/png" sizes="512x512" href="/static/icon-512.png">
<link rel="apple-touch-icon" href="/static/icon-192.png">

<link rel="stylesheet" href="/static/custom.css">
<script src="/static/pwa.js" defer></script>
"""

# ==========================================
# UI HELPERS
# ==========================================

# --- CHAT UI UPDATE ---
def update_c_ui(prov, force_all=False):
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
    final_choices = get_filtered_model_choices(prov, choices, force_all)
    
    default_val = final_choices[0][1] if final_choices else None
    return gr.update(choices=final_choices, value=default_val), badge

# --- VISION UI UPDATE ---
def update_v_models(prov, force_all=False):
    p_data = PROVIDERS.get(prov, {})
    api_key = API_KEYS.get(p_data.get("key_name"))
    
    # 1. Fetch
    models, error = fetch_available_models(prov, api_key)
    choices = []
    
    if models and prov == "Poe":
        choices = [(m["name"], m["id"]) for m in models if "Vision" in m.get("type", "")]
    elif not models or prov != "Poe":
        choices = [(m, m) for m in p_data.get("vision_models", [])]

    # 2. Filter via Helper
    final_choices = get_filtered_model_choices(prov, choices, force_all)
    
    default_val = final_choices[0][1] if final_choices else None
    return gr.update(choices=final_choices, value=default_val)

# --- IMAGE UI UPDATE ---
def update_image_models(prov, force_all=False):
    p_data = PROVIDERS.get(prov, {})
    api_key = API_KEYS.get(p_data.get("key_name"))
    
    # 1. Fetch
    models, error = fetch_available_models(prov, api_key)
    choices = []
    
    if models and prov == "Poe":
        choices = [(f"{m['name']}", m['id']) for m in models if m.get('type') in ["Bild-Gen", "Video-Gen"]]
    
    if not choices:
        choices = [(m, m) for m in p_data.get("image_models", [])]
        
    # 2. Filter via Helper
    final_choices = get_filtered_model_choices(prov, choices, force_all)
    
    return gr.update(choices=final_choices, value=final_choices[0][1] if final_choices else None)

# --- TRANSCRIPTION UI UPDATE ---
def update_t_ui(prov, force_all=False):
    badge = get_compliance_html(prov)
    is_whisper = prov != "Gladia"
    
    # 1. Fetch (Only for Whisper providers)
    choices = []
    if is_whisper:
        p_data = PROVIDERS.get(prov, {})
        # Note: Usually we use static config for Whisper, but let's allow fetching if supported
        raw_list = p_data.get("audio_models", [])
        choices = [(m, m) for m in raw_list]
        
        # Filter via Helper
        choices = get_filtered_model_choices(prov, choices, force_all)

    default_val = choices[0][1] if choices else None

    return (
        badge,
        gr.update(visible=is_whisper, choices=choices, value=default_val),
        gr.update(visible=not is_whisper), # Gladia Opts
        gr.update(visible=is_whisper)      # Whisper Opts
    )
    
# --- Chat functions ---
def user_msg(msg, hist):
    return "", hist + [{"role": "user", "content": msg}]

def bot_msg(hist, prov, mod, temp, sys, key):
    """
    Execute chat with correct history slicing to prevent duplicate messages
    and ensure restored chats are respected as context.
    """
    if not hist: return hist

    # Ensure the last message is actually from the user before processing
    if hist[-1]["role"] != "user":
        return hist

    # 1. Get the latest user message
    last_user_msg = hist[-1]["content"]

    # 2. Append the empty assistant bubble (for streaming output)
    hist.append({"role": "assistant", "content": ""})

    # 3. Prepare Context for API
    # hist is now [..., PrevBot, CurrUser, CurrEmptyBot]
    # We want [..., PrevBot] as history, because run_chat adds CurrUser itself.
    # Therefore we slice up to -2.
    raw_context = hist[:-2]
    
    # 4. Normalize Context (Fixes "Poe" history being used in "Scaleway")
    clean_context = []
    for m in raw_context:
        role = m["role"]
        # Normalize 'bot' (Poe) to 'assistant' (OpenAI standard)
        if role in ["bot", "model"]: 
            role = "assistant"
        clean_context.append({"role": role, "content": m["content"]})

    # 5. Stream
    try:
        for chunk in run_chat(last_user_msg, clean_context, prov, mod, temp, sys, key):
            hist[-1]["content"] = chunk
            yield hist
    except Exception as e:
        hist[-1]["content"] = f"🔥 Fehler: {str(e)}"
        yield hist

def save_chat(hist, prov, mod):
    """Save current chat to database"""
    try:
        logger.info(f"Attempting to save chat for user {current_user.get('id')}")

        if not current_user["id"]:
            logger.warning("Save chat failed: User not logged in")
            return "❌ Bitte anmelden"

        if not hist or len(hist) == 0:
            logger.warning("Save chat failed: Empty chat history")
            return "❌ Kein Chat zum Speichern"

        logger.debug(f"Chat history length: {len(hist)}")

        # Generate title from first message
        first_content = hist[0].get("content", "") if isinstance(hist[0], dict) else str(hist[0])
        title = first_content[:50] + "..." if len(first_content) > 50 else first_content

        logger.info(f"Saving chat with title: {title}")
        chat_id = save_chat_history(current_user["id"], prov, mod, hist, title)
        logger.info(f"Chat saved successfully with ID: {chat_id}")

        return f"✅ Chat gespeichert (ID: {chat_id})"

    except Exception as e:
        logger.exception(f"Error saving chat: {str(e)}")
        return f"🔥 Fehler beim Speichern: {str(e)}\n\nDetails: {type(e).__name__}"

def load_chat_list():
    """Load user's chat history"""
    if not current_user["id"]:
        return [["Bitte anmelden", "", "", ""]]

    chats = get_user_chat_history(current_user["id"])
    data = []
    for chat in chats:
        data.append([
            chat.id,
            chat.timestamp.strftime("%Y-%m-%d %H:%M"),
            chat.title or "Ohne Titel",
            chat.model
        ])
    return data if data else [["Keine Chats vorhanden", "", "", ""]]

def load_chat_list_with_state():
    """Load user's chat history returning both Display Data and State Data"""
    if not current_user["id"]:
        return [["Bitte anmelden", "", "", ""]], []

    chats = get_user_chat_history(current_user["id"])
    
    # Clean data for State (ID, Date, Title, Model)
    state_data = [[c.id, c.timestamp.strftime("%Y-%m-%d %H:%M"), c.title or "Ohne Titel", c.model] for c in chats]
    
    if not state_data:
        return [["Keine Chats vorhanden", "", "", ""]], []
        
    return state_data, state_data

def select_chat_row(evt: gr.SelectData, state_data):
    """Smart Selection for Chat List"""
    try:
        row_idx = evt.index[0]
        if row_idx < len(state_data):
            real_row = state_data[row_idx]
            # ID is in column 0
            return int(real_row[0])
    except Exception as e:
        logger.error(f"Selection error: {e}")
    return gr.update()

def attach_content_to_chat(hist, attach_type, attach_id, custom_text, uploaded_file):
    """
    Attach content (Transcript, Vision, Custom Text, or File) to chat.
    """
    if not current_user["id"]:
        return hist, "❌ Bitte anmelden"

    content_to_add = ""

    # 1. Handle File Upload
    if attach_type == "Datei uploaden" and uploaded_file:
        try:
            filename = os.path.basename(uploaded_file.name)
            # Simple text extraction for code/text files
            if filename.lower().endswith(('.txt', '.md', '.csv', '.json', '.py', '.js', '.html', '.css', '.xml', '.yaml')):
                with open(uploaded_file.name, "r", encoding="utf-8") as f:
                    file_content = f.read()
                content_to_add = f"[Datei Inhalt: {filename}]\n\n{file_content}"
            else:
                # For binaries/PDFs/Images (placeholder for future OCR/Vision logic)
                content_to_add = f"[Datei hochgeladen: {filename}] (Inhalt konnte nicht als Text extrahiert werden, aber Datei liegt vor.)"
        except Exception as e:
            return hist, f"❌ Fehler beim Lesen der Datei: {str(e)}"

    # 2. Handle Transcript
    elif attach_type == "Transkript":
        if not attach_id: return hist, "❌ ID fehlt"
        db = get_db()
        trans = db.query(Transcription).filter(Transcription.id == int(attach_id), Transcription.user_id == current_user["id"]).first()
        db.close()
        if trans: content_to_add = f"[Transkript #{trans.id}]\n\n{trans.original_text}"
        else: return hist, "❌ Transkript nicht gefunden"

    # 3. Handle Vision
    elif attach_type == "Vision-Ergebnis":
        if not attach_id: return hist, "❌ ID fehlt"
        db = get_db()
        vision = db.query(VisionResult).filter(VisionResult.id == int(attach_id), VisionResult.user_id == current_user["id"]).first()
        db.close()
        if vision: content_to_add = f"[Vision #{vision.id}]\n\n{vision.result}"
        else: return hist, "❌ Vision nicht gefunden"

    # 4. Handle Custom Text
    elif attach_type == "Eigener Text":
        if not custom_text: return hist, "❌ Text fehlt"
        content_to_add = custom_text

    # Add to chat history
    if not hist: hist = []
    if content_to_add:
        hist.append({"role": "user", "content": content_to_add})
        return hist, f"✅ {attach_type} angehängt"
    
    return hist, "❌ Nichts zum Anhängen"

def get_user_prompt_choices():
    """Get list of user's custom prompt names for dropdown"""
    if not current_user["id"]: return []
    prompts = get_user_custom_prompts(current_user["id"])
    return [p.name for p in prompts]

def insert_custom_prompt(prompt_name, current_msg):
    """Insert selected custom prompt text into message box"""
    if not current_user["id"] or not prompt_name: return current_msg
    
    db = get_db()
    p = db.query(CustomPrompt).filter(CustomPrompt.user_id == current_user["id"], CustomPrompt.name == prompt_name).first()
    db.close()
    
    if p:
        # Append if text exists, otherwise replace
        if current_msg:
            return current_msg + "\n\n" + p.prompt_text
        return p.prompt_text
    return current_msg

def load_single_chat(chat_id):
    """Load a specific chat into the UI"""
    if not current_user["id"] or not chat_id:
        return None, "❌ Ungültige ID"

    chat = get_single_chat(int(chat_id), current_user["id"])
    if chat:
        try:
            messages = json.loads(chat.messages)
            # Basic validation to ensure it's a list of dicts
            if isinstance(messages, list) and len(messages) > 0 and isinstance(messages[0], dict):
                return messages, f"✅ Chat '{chat.title}' geladen"
            else:
                return None, "⚠️ Chat-Format veraltet oder ungültig"
        except Exception as e:
            return None, f"🔥 Ladefehler: {str(e)}"
            
    return None, "❌ Chat nicht gefunden"

def delete_chat(chat_id):
    """Delete a chat"""
    if not current_user["id"] or not chat_id:
        return "❌ Ungültige ID", load_chat_list()

    if delete_chat_history(int(chat_id), current_user["id"]):
        return "✅ Chat gelöscht", load_chat_list()
    return "❌ Fehler beim Löschen", load_chat_list()

def clear_chat():
    """Clear current chat"""
    return [], ""

def get_filtered_model_choices(provider, available_models_list, force_all):
    """
    Helper to filter models based on user preferences.
    available_models_list: list of (name, id) tuples
    """
    if not current_user["id"]:
        return available_models_list

    # 1. Get User Preferences (Ordered ID list)
    pref_ids = get_user_visible_models(current_user["id"], provider)
    
    # If no preferences set, return everything
    if not pref_ids:
        return available_models_list

    val_to_name = {v: k for k, v in available_models_list}
    
    # 2. Separate into Favorites and Others
    fav_choices = []
    seen_ids = set()
    
    # Add favorites in specific order
    for pid in pref_ids:
        if pid in val_to_name:
            fav_choices.append((val_to_name[pid], pid))
            seen_ids.add(pid)
            
    # Add the rest
    rest_choices = []
    for name, val in available_models_list:
        if val not in seen_ids:
            rest_choices.append((name, val))
            
    # 3. Decision Logic
    # If NOT force_all, show ONLY favorites (if any exist).
    # If force_all, show Favorites first, then the rest.
    if not force_all and fav_choices:
        return fav_choices
    
    return fav_choices + rest_choices

# --- Functions for Transkription ---

# Send to Chat functionality
def send_transcript_to_chat(transcript, template, notes, custom_prompt, provider, model, api_key):
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


def generate_and_handle_ui(prompt, provider, model, width, height, steps, key):
    """Generates image and updates ALL UI components"""
    img_path, status = run_image_gen(prompt, provider, model, width, height, steps, key)

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
def process_gallery_save(img_path, provider, prompt, model):
    """Explicit wrapper to handle DB saving safely"""
    try:
        if not current_user["id"]:
            return "❌ Bitte anmelden", gr.update(visible=True)
        if not img_path or not os.path.exists(img_path):
            return "❌ Datei nicht gefunden (Session abgelaufen?)", gr.update(visible=True)

        import shutil
        permanent_dir = "/var/www/transkript_app/generated_images"
        os.makedirs(permanent_dir, exist_ok=True)
        filename = f"img_{int(time.time())}_{os.path.basename(img_path)}"
        permanent_path = os.path.join(permanent_dir, filename)
        shutil.copy2(img_path, permanent_path)

        img_id = save_generated_image(
            user_id=int(current_user["id"]), 
            provider=str(provider), 
            model=str(model), 
            prompt=str(prompt), 
            image_path=str(permanent_path)
        )
        return f"✅ Gespeichert (ID: {img_id})", gr.update(visible=False)

    except Exception as e:
        logger.exception(f"Gallery Save Error: {e}")
        return f"🔥 Fehler: {str(e)}", gr.update(visible=True)

def manual_save_transcription(original, translated, provider, model, lang, filename="manual_save.mp3"):
    """Manually save transcription to database"""
    try:
        if not current_user["id"]:
            return "❌ Bitte anmelden"

        if not original or original.strip() == "":
            return "❌ Kein Transkript zum Speichern"

        trans_id = save_transcription(
            user_id=current_user["id"],
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
                        # Top Bar: Provider, Model, Load All
                        with gr.Row():
                            c_prov = gr.Dropdown(list(PROVIDERS.keys()), value="Scaleway", label="Anbieter", scale=1)
                            c_model = gr.Dropdown(PROVIDERS["Scaleway"]["chat_models"], value=PROVIDERS["Scaleway"]["chat_models"][0], label="Modell", scale=2)
                            c_load_all_btn = gr.Button("🌍 Alle laden", scale=0, size="sm", variant="secondary", min_width=80)

                        c_badge = gr.HTML(value=PROVIDERS["Scaleway"]["badge"])

                        # UI Update Logic
                        def trigger_update_c_ui(prov, load_all_click=False):
                            return update_c_ui(prov, force_all=load_all_click)

                        c_prov.change(trigger_update_c_ui, inputs=[c_prov], outputs=[c_model, c_badge])
                        c_load_all_btn.click(lambda p: update_c_ui(p, force_all=True), inputs=[c_prov], outputs=[c_model, c_badge])

                        # Chat Area
                        c_bot = gr.Chatbot(height=500, type="messages")
                        c_msg = gr.Textbox(placeholder="Nachricht eingeben...", show_label=False, lines=3)

                        with gr.Row():
                            c_btn = gr.Button("📤 Senden", variant="primary", scale=2)
                            c_stop_btn = gr.Button("🛑 Stop", variant="stop", scale=1)
                            c_save_btn = gr.Button("💾 Speichern", scale=1)
                            c_clear_btn = gr.Button("🗑️ Neu", scale=1)

                        c_save_status = gr.Markdown("")

                    # Right Sidebar (Settings & Tools)
                    with gr.Column(scale=1):
                        
                        # 1. SETTINGS (Default: Closed)
                        with gr.Accordion("⚙️ Einstellungen", open=False):
                            c_key = gr.Textbox(label="API Key (Optional)", type="password")
                            c_sys = gr.Textbox(label="System Rolle", value="Du bist ein hilfreicher Assistent.", lines=3)
                            c_temp = gr.Slider(0, 2, value=0.7, label="Kreativität")

                        # 2. LOAD CHATS (Default: Closed, Preloads on Tab Select)
                        with gr.Accordion("📚 Alte Chats laden", open=False):
                            c_history_state = gr.State([]) # Store real data
                            refresh_chats_btn = gr.Button("🔄 Liste aktualisieren", size="sm")
                            
                            old_chats = gr.Dataframe(
                                headers=["ID", "Datum", "Titel", "Modell"],
                                value=[[None, "", "", ""]],
                                label="Klicken zum Laden",
                                interactive=False,
                                height=200,
                                wrap=True
                            )
                            
                            with gr.Row():
                                load_chat_id = gr.Number(label="Chat-ID", precision=0)
                                delete_chat_btn = gr.Button("🗑️", scale=0)
                            
                            load_chat_btn = gr.Button("📥 Chat wiederherstellen", variant="primary")
                            chat_load_status = gr.Markdown("")

                        # 3. ATTACHMENTS (Default: Closed)
                        with gr.Accordion("📎 Inhalt & Prompts", open=False):
                            
                            # A. Custom Prompts
                            gr.Markdown("**📝 Vorlagen**")
                            with gr.Row():
                                c_prompt_select = gr.Dropdown(choices=[], label="Vorlage wählen", scale=2)
                                c_prompt_refresh = gr.Button("🔄", scale=0, size="sm")
                            c_insert_prompt_btn = gr.Button("⬇️ In Textfeld einfügen", size="sm")
                            
                            gr.Markdown("---")
                            
                            # B. Content Attachments
                            gr.Markdown("**📎 Anhang**")
                            attach_type = gr.Radio(
                                ["Transkript", "Vision-Ergebnis", "Eigener Text", "Datei uploaden"],
                                value="Transkript",
                                label="Typ"
                            )
                            
                            # Dynamic inputs
                            attach_id = gr.Number(label="ID (Transkript/Vision)", precision=0, visible=True)
                            attach_custom = gr.Textbox(label="Text einfügen", lines=3, visible=False)
                            attach_file = gr.File(label="Datei wählen", visible=False)
                            
                            attach_btn = gr.Button("➕ An Chat anhängen", variant="secondary")
                            attach_status = gr.Markdown("")

                            # Input visibility logic
                            def toggle_attach_inputs(atype):
                                return (
                                    gr.update(visible=atype in ["Transkript", "Vision-Ergebnis"]), # ID
                                    gr.update(visible=atype == "Eigener Text"),                   # Text
                                    gr.update(visible=atype == "Datei uploaden")                  # File
                                )
                            attach_type.change(toggle_attach_inputs, attach_type, [attach_id, attach_custom, attach_file])

                # --- EVENT WIRING ---

                # Chat Execution with STOP function
                user_event = c_msg.submit(user_msg, [c_msg, c_bot], [c_msg, c_bot], queue=False)
                bot_event = user_event.then(bot_msg, [c_bot, c_prov, c_model, c_temp, c_sys, c_key], c_bot)
                
                send_click = c_btn.click(user_msg, [c_msg, c_bot], [c_msg, c_bot], queue=False)
                bot_click_event = send_click.then(bot_msg, [c_bot, c_prov, c_model, c_temp, c_sys, c_key], c_bot)

                # Stop Button
                c_stop_btn.click(fn=None, cancels=[bot_event, bot_click_event])

                # Save & Clear
                c_save_btn.click(save_chat, [c_bot, c_prov, c_model], c_save_status)
                c_clear_btn.click(lambda: ([], ""), outputs=[c_bot, c_save_status])

                # --- History Logic ---
                # Preload on Tab Select
                chat_tab.select(load_chat_list_with_state, outputs=[old_chats, c_history_state])
                
                # Refresh Button
                refresh_chats_btn.click(load_chat_list_with_state, outputs=[old_chats, c_history_state])
                
                # Smart Selection (Click row -> Set ID)
                old_chats.select(select_chat_row, inputs=[c_history_state], outputs=[load_chat_id])
                
                # Load / Delete Actions
                load_chat_btn.click(load_single_chat, load_chat_id, [c_bot, chat_load_status])
                delete_chat_btn.click(delete_chat, load_chat_id, [chat_load_status, old_chats]) # Note: Ideally chain reload here

                # --- Attachment Logic ---
                attach_btn.click(
                    attach_content_to_chat,
                    [c_bot, attach_type, attach_id, attach_custom, attach_file],
                    [c_bot, attach_status]
                )

                # --- Prompt Template Logic ---
                # Refresh Dropdown
                c_prompt_refresh.click(
                    get_user_prompt_choices, 
                    outputs=c_prompt_select
                )
                # Auto-load prompts when opening accordion (optional, using mouseover/click for now or chaining)
                chat_tab.select(get_user_prompt_choices, outputs=c_prompt_select)

                # Insert Text
                c_insert_prompt_btn.click(
                    insert_custom_prompt,
                    inputs=[c_prompt_select, c_msg],
                    outputs=[c_msg]
                )

            # --- TAB 2: TRANSKRIPTION - WITH WHISPER OPTIONS ---
            with gr.TabItem("🎙️ Transkription"):
                with gr.Row():
                    with gr.Column():
                        t_audio = gr.Audio(type="filepath", label="Datei")
                        
                        with gr.Row():
                            t_prov = gr.Radio(["Gladia", "Mistral", "Scaleway", "Groq"], value="Gladia", label="Engine", scale=2)
                            # LOAD ALL BUTTON (Only visible for Whisper providers ideally, but we keep it simple)
                            t_load_all = gr.Button("🌍 Alle", scale=0, size="sm")
                            
                        t_model = gr.Dropdown(choices=[], value=None, visible=False, label="Modell")
                        t_badge = gr.HTML(value=get_compliance_html("Gladia"))

                        # GLADIA OPTIONS
                        gladia_opts = gr.Accordion("⚙️ Gladia Optionen", open=True, visible=True)
                        with gladia_opts:
                            t_lang = gr.Dropdown([("Auto-Erkennung", "auto"), ("Deutsch", "de"), ("Englisch", "en")], value="de", label="Sprache")
                            t_diar = gr.Checkbox(True, label="🎭 Sprecher erkennen")
                            t_trans = gr.Checkbox(False, label="🌍 Übersetzen")
                            t_target = gr.Dropdown([("Deutsch", "de"), ("Englisch", "en")], value="en", label="Zielsprache")

                        # WHISPER OPTIONS
                        whisper_opts = gr.Accordion("⚙️ Whisper Optionen", open=True, visible=False)
                        with whisper_opts:
                            w_lang = gr.Dropdown([("Auto-Erkennung", "auto"), ("Deutsch", "de"), ("Englisch", "en")], value="de", label="Sprache")
                            w_temp = gr.Slider(0, 1, value=0, step=0.1, label="Temperatur")
                            w_prompt = gr.Textbox(label="Kontext-Prompt")

                        t_key = gr.Textbox(label="🔑 API Key", type="password")
                        t_btn = gr.Button("▶️ Starten", variant="primary", size="lg")
                        t_log = gr.Textbox(label="Log", lines=5)

                        # UI Updates
                        # Normal update (uses prefs)
                        t_prov.change(lambda p: update_t_ui(p, force_all=False), t_prov, [t_badge, t_model, gladia_opts, whisper_opts])
                        # Load All update
                        t_load_all.click(lambda p: update_t_ui(p, force_all=True), t_prov, [t_badge, t_model, gladia_opts, whisper_opts])

                    with gr.Column():
                        t_orig = gr.Textbox(label="📄 Original Transkript", lines=15, show_copy_button=True)
                        t_trsl = gr.Textbox(label="🌍 Übersetzung", lines=15, show_copy_button=True)

                        with gr.Row():
                            t_save_btn = gr.Button("💾 Transkript speichern", variant="secondary")
                            t_save_status = gr.Markdown("")

                # Send to Chat Section
                with gr.Accordion("💬 An Chat senden", open=False) as send_to_chat_section:
                    gr.Markdown("### Transkript automatisch weiterverarbeiten")

                    with gr.Row():
                        prompt_template = gr.Dropdown(
                            choices=list(TRANSCRIPT_PROMPTS.keys()),
                            value="Veranstaltungsrückblick",
                            label="📋 Prompt-Vorlage",
                            info="Wähle eine vordefinierte Aufgabe"
                        )
                        chat_provider = gr.Dropdown(
                            list(PROVIDERS.keys()),
                            value="Scaleway",
                            label="🤖 Chat Provider"
                        )

                    chat_model_for_transcript = gr.Dropdown(
                        PROVIDERS["Scaleway"]["chat_models"],
                        value=PROVIDERS["Scaleway"]["chat_models"][0],
                        label="Modell"
                    )

                    additional_notes = gr.Textbox(
                        label="📝 Zusätzliche Hinweise (Optional)",
                        placeholder="z.B.: Erwähne unseren Kooperationspartner, betone den innovativen Ansatz...",
                        lines=3
                    )

                    custom_prompt_input = gr.Textbox(
                        label="✏️ Eigener Prompt (nur bei 'Eigener Prompt')",
                        placeholder="Dein individueller Prompt...",
                        lines=3,
                        visible=False
                    )

                    send_to_chat_btn = gr.Button("💬 An Chat senden und verarbeiten", variant="primary", size="lg")
                    send_status = gr.Markdown("")

                # Update chat model dropdown when provider changes
                def update_chat_model_dropdown(prov):
                    ms = PROVIDERS.get(prov, {}).get("chat_models", [])
                    return gr.update(choices=ms, value=ms[0] if ms else "")

                chat_provider.change(update_chat_model_dropdown, chat_provider, chat_model_for_transcript)

                # Show/hide custom prompt field
                def toggle_custom_prompt(template):
                    return gr.update(visible=(template == "Eigener Prompt"))

                prompt_template.change(toggle_custom_prompt, prompt_template, custom_prompt_input)

                t_prov.change(update_t_ui, t_prov, [t_badge, t_model, gladia_opts, whisper_opts])

                # Sync language between Gladia and Whisper
                def sync_languages(lang_value):
                    return lang_value

                t_lang.change(sync_languages, t_lang, w_lang)
                w_lang.change(sync_languages, w_lang, t_lang)

                # Connect transcription button
                t_btn.click(
                    run_and_save_transcription,
                    inputs=[
                        t_audio, t_prov, t_model,
                        t_lang,
                        w_temp,
                        w_prompt,
                        t_diar,
                        t_trans,
                        t_target,
                        t_key
                    ],
                    outputs=[t_log, t_orig, t_trsl]
                )

                # Connect manual save button
                t_save_btn.click(
                    manual_save_transcription,
                    inputs=[t_orig, t_trsl, t_prov, t_model, t_lang],
                    outputs=t_save_status
                )

                send_to_chat_btn.click(
                    send_transcript_to_chat,
                    inputs=[
                        t_orig,  # Use original transcript
                        prompt_template,
                        additional_notes,
                        custom_prompt_input,
                        chat_provider,
                        chat_model_for_transcript,
                        t_key
                    ],
                    outputs=send_status
                )

            # --- TAB 3: VISION ---
            with gr.TabItem("👁️ Vision"):
                with gr.Row():
                    with gr.Column():
                        v_img = gr.Image(type="filepath", label="Bild")
                        with gr.Row():
                            v_prov = gr.Dropdown(["Scaleway", "Mistral", "Nebius", "OpenRouter", "Poe"], value="Scaleway", label="Provider", scale=2)
                            # LOAD ALL BUTTON
                            v_load_all = gr.Button("🌍 Alle", scale=0, size="sm")
                            
                        v_model = gr.Dropdown(PROVIDERS["Scaleway"]["vision_models"], value="pixtral-12b-2409", label="Modell", allow_custom_value=True)
                        v_key = gr.Textbox(label="Key (Optional)", type="password")
                        v_prompt = gr.Textbox(label="Frage", value="Beschreibe dieses Bild detailliert.")
                        v_btn = gr.Button("Analysieren", variant="primary")
                    
                    with gr.Column():
                        v_out = gr.Markdown(label="Ergebnis")
                        
                # UI Updates
                v_prov.change(lambda p: update_v_models(p, force_all=False), v_prov, v_model)
                v_load_all.click(lambda p: update_v_models(p, force_all=True), v_prov, v_model)
                
                v_btn.click(run_vision, [v_img, v_prompt, v_prov, v_model, v_key], v_out)
            
            # --- TAB 4: BILDERZEUGUNG ---
            with gr.TabItem("🎨 Bilderzeugung"):
                with gr.Row():
                    with gr.Column():
                        g_prompt = gr.Textbox(label="Prompt", placeholder="Eine futuristische Kirche...", lines=3)
                        
                        with gr.Row():
                            g_provider = gr.Dropdown(["Nebius", "Scaleway", "OpenRouter", "Poe"], value="Nebius", label="Provider", scale=2)
                            # LOAD ALL BUTTON
                            g_load_all = gr.Button("🌍 Alle", scale=0, size="sm")
                            
                        g_model = gr.Dropdown(PROVIDERS["Nebius"]["image_models"], value="black-forest-labs/flux-schnell", label="Modell")
                        
                        # UI Updates
                        g_provider.change(lambda p: update_image_models(p, force_all=False), inputs=[g_provider], outputs=[g_model])
                        g_load_all.click(lambda p: update_image_models(p, force_all=True), inputs=[g_provider], outputs=[g_model])
                        
                        # (Rest of Image Tab components: sliders, buttons, outputs - same as before)
                        with gr.Row():
                            g_w = gr.Slider(256, 1024, value=1024, step=64, label="Breite")
                            g_h = gr.Slider(256, 1024, value=768, step=64, label="Höhe")
                        g_steps = gr.Slider(4, 16, value=10, label="Schritte")
                        g_key = gr.Textbox(label="Key", type="password")
                        g_btn = gr.Button("🎨 Generieren", variant="primary")
                        g_stat = gr.Textbox(label="Status", interactive=False)
                        
                        with gr.Row():
                            g_save_btn = gr.Button("💾 Speichern", visible=False)
                        g_save_status = gr.Markdown("")

                    with gr.Column():
                        g_out = gr.Image(label="Ergebnis", type="filepath", show_download_button=False)
                        g_download_file = gr.File(label="Download")

                g_img_path = gr.State(value=None)
                
                # Logic
                g_btn.click(generate_and_handle_ui, [g_prompt, g_provider, g_model, g_w, g_h, g_steps, g_key], [g_out, g_stat, g_img_path, g_download_file, g_save_btn, g_save_status])
                g_save_btn.click(process_gallery_save, [g_img_path, g_provider, g_prompt, g_model], [g_save_status, g_save_btn])

            # --- TAB 5: VERLAUF & VERWALTUNG ---
            with gr.TabItem("📚 Verlauf & Verwaltung", id="tab_management"):
                # Display Current User
                display_user = current_user.get('username') if current_user.get('username') else "Gast"
                gr.Markdown(f"### 👤 Angemeldet als: **{display_user}**")

                # Helper to make tables look full (adds empty rows)
                def pad_data(data, width, min_rows=6):
                    while len(data) < min_rows:
                        row = [None] * width
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
                        def load_trans_data():
                            if not current_user["id"]:
                                return pad_data([], 5), []
                            try:
                                t_list = get_user_transcriptions(current_user["id"])
                                # Prepare data list
                                clean_data = [[t.id, t.timestamp.strftime("%Y-%m-%d %H:%M"), t.title or "—", t.provider, t.language] for t in t_list]
                                # Create padded version for display
                                padded_display = pad_data(list(clean_data), 5)
                                # Return padded for display, clean for state logic
                                return padded_display, clean_data
                            except Exception as e:
                                logger.exception(e)
                                return pad_data([], 5), []

                        def load_single_trans(tid):
                            if not tid or not current_user["id"]: return gr.update(), "❌"
                            db = SessionLocal()
                            t = db.query(Transcription).filter(Transcription.id == int(tid), Transcription.user_id == current_user["id"]).first()
                            db.close()
                            return (t.original_text, f"✅ Geladen: {t.title}") if t else ("", "❌ Nicht gefunden")

                        def select_trans_row(evt: gr.SelectData, state_data):
                            """Smart Selection: Uses state to find ID from ANY column click"""
                            try:
                                row_idx = evt.index[0]
                                # Check if row exists in real data (ignore padding clicks)
                                if row_idx < len(state_data):
                                    real_row = state_data[row_idx]
                                    t_id = int(real_row[0]) # ID is col 0
                                    content, status = load_single_trans(t_id)
                                    return t_id, content, status
                            except: pass
                            return gr.update(), gr.update(), ""

                        def del_trans(tid):
                            if delete_transcription(int(tid or 0), current_user["id"]):
                                d, s = load_trans_data() # Reload table
                                return "", "✅ Gelöscht", d, s
                            d, s = load_trans_data()
                            return "", "❌ Fehler", d, s

                        # Wiring
                        refresh_trans_btn.click(load_trans_data, outputs=[trans_history, trans_state])
                        # Pass 'trans_state' to select so we know what was clicked
                        trans_history.select(select_trans_row, inputs=[trans_state], outputs=[trans_id_input, loaded_trans_display, trans_action_status])
                        trans_id_input.change(load_single_trans, trans_id_input, [loaded_trans_display, trans_action_status])
                        delete_trans_btn.click(del_trans, trans_id_input, [loaded_trans_display, trans_action_status, trans_history, trans_state])

                        # Chat Button (Checks if msg_input exists in global scope)
                        if 'msg_input' in locals():
                            trans_to_chat_btn.click(lambda x: x, inputs=loaded_trans_display, outputs=msg_input)


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
                        def load_img_data():
                            if not current_user["id"]: return pad_data([], 4), []
                            try:
                                i_list = get_user_generated_images(current_user["id"])
                                clean = [[i.id, i.timestamp.strftime("%Y-%m-%d"), i.prompt, i.model] for i in i_list]
                                return pad_data(list(clean), 4), clean
                            except: return pad_data([], 4), []

                        def load_single_img(tid):
                            if not tid or not current_user["id"]: return None, "", "❌"
                            db = SessionLocal()
                            img = db.query(GeneratedImage).filter(GeneratedImage.id == int(tid)).first()
                            db.close()
                            if img and os.path.exists(img.image_path):
                                return img.image_path, img.prompt, f"✅ Geladen"
                            return None, "", "❌ Datei fehlt"

                        def select_img_row(evt: gr.SelectData, state_data):
                            try:
                                row_idx = evt.index[0]
                                if row_idx < len(state_data):
                                    tid = int(state_data[row_idx][0])
                                    path, prmt, stat = load_single_img(tid)
                                    return tid, path, prmt, stat
                            except: pass
                            return gr.update(), None, "", ""

                        def del_img(tid):
                            delete_generated_image(int(tid or 0), current_user["id"])
                            d, s = load_img_data()
                            return None, "", "✅ Gelöscht", d, s

                        refresh_images_btn.click(load_img_data, outputs=[images_history, img_state])
                        images_history.select(select_img_row, inputs=[img_state], outputs=[img_id_input, loaded_img_display, loaded_img_prompt, img_action_status])
                        img_id_input.change(load_single_img, img_id_input, [loaded_img_display, loaded_img_prompt, img_action_status])
                        delete_img_btn.click(del_img, img_id_input, [loaded_img_display, loaded_img_prompt, img_action_status, images_history, img_state])

                        if 'msg_input' in locals():
                            img_to_chat_btn.click(lambda x: x, inputs=loaded_img_prompt, outputs=msg_input)


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
                        def load_prompts_data():
                            if not current_user["id"]: return pad_data([], 3), []
                            p_list = get_user_custom_prompts(current_user["id"])
                            clean = [[p.id, p.name, p.category] for p in p_list]
                            return pad_data(list(clean), 3), clean

                        def load_single_prompt(tid):
                            if not tid: return ""
                            db = SessionLocal()
                            p = db.query(CustomPrompt).filter(CustomPrompt.id == int(tid)).first()
                            db.close()
                            return p.prompt_text if p else ""

                        def select_prompt_row(evt: gr.SelectData, state_data):
                            try:
                                if evt.index[0] < len(state_data):
                                    tid = int(state_data[evt.index[0]][0])
                                    return tid, load_single_prompt(tid)
                            except: pass
                            return gr.update(), ""

                        def save_p(n, c, t):
                            save_custom_prompt(current_user["id"], n, t, c.lower())
                            d, s = load_prompts_data()
                            return "✅ Gespeichert", d, s

                        def del_p(tid):
                            delete_custom_prompt(int(tid or 0), current_user["id"])
                            d, s = load_prompts_data()
                            return "", d, s

                        save_prompt_btn.click(save_p, [new_prompt_name, new_prompt_category, new_prompt_text], [save_prompt_status, saved_prompts, prompt_state])
                        refresh_prompts_btn.click(load_prompts_data, outputs=[saved_prompts, prompt_state])
                        saved_prompts.select(select_prompt_row, inputs=[prompt_state], outputs=[prompt_id_load, loaded_prompt_display])
                        prompt_id_load.change(load_single_prompt, prompt_id_load, loaded_prompt_display)
                        delete_prompt_btn.click(del_p, prompt_id_load, [loaded_prompt_display, saved_prompts, prompt_state])

                        if 'msg_input' in locals():
                            prompt_to_chat_btn.click(lambda x: x, inputs=loaded_prompt_display, outputs=msg_input)


                    # =========================================================
                    # 4. USER ADMIN
                    # =========================================================
                    with gr.TabItem("👥 Benutzerverwaltung"):
                        users_list = gr.Dataframe(headers=["ID", "User", "Admin", "Erstellt"], height=400)
                        refresh_users_btn = gr.Button("🔄 Liste aktualisieren")

                        def load_users():
                            if not current_user.get("is_admin"): return []
                            db = SessionLocal()
                            usrs = db.query(User).all()
                            db.close()
                            return [[u.id, u.username, u.is_admin, u.created_at.strftime("%Y-%m-%d")] for u in usrs]

                        refresh_users_btn.click(load_users, outputs=users_list)
                        
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
                    def load_users_list():
                        return get_all_users()
                    
                    refresh_users_btn.click(load_users_list, outputs=users_table)
                    
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
                    def handle_toggle_admin(user_id):
                        if not current_user["id"]:
                            return "❌ Nicht angemeldet", get_all_users()
                        
                        success, message = toggle_admin_status(int(user_id), current_user["id"])
                        return message, get_all_users()
                    
                    toggle_admin_btn.click(
                        handle_toggle_admin,
                        inputs=selected_user_id,
                        outputs=[toggle_admin_status, users_table]
                    )
                    
                    # Delete user
                    def handle_delete_user(user_id, confirmation):
                        if confirmation != "LÖSCHEN":
                            return "❌ Bestätigung erforderlich: Tippe 'LÖSCHEN'", get_all_users()
                        
                        if not current_user["id"]:
                            return "❌ Nicht angemeldet", get_all_users()
                        
                        success, message = delete_user(int(user_id), current_user["id"])
                        return message, get_all_users()
                    
                    delete_user_btn.click(
                        handle_delete_user,
                        inputs=[selected_user_id, delete_confirm],
                        outputs=[delete_user_status, users_table]
                    )

                    # --- AUTO-LOAD ON TAB SWITCH ---
                    trans_tab.select(fn=load_trans_data, outputs=[trans_history, trans_state])
                    images_tab.select(fn=load_img_data, outputs=[images_history, img_state])
                    prompts_tab.select(fn=load_prompts_data, outputs=[saved_prompts, prompt_state])
                    
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
                    
                    def fetch_models_for_provider(provider, api_key=None):
                        """Fetch and display available models"""
                        if not current_user["id"]:
                            return [["", ""]], "❌ Bitte anmelden", gr.update()
                        
                        # Get API key for provider if available
                        provider_key = API_KEYS.get(provider.lower(), "")
                        
                        models, error = fetch_available_models(provider, provider_key)
                        
                        if error:
                            return [["", ""]], f"❌ {error}", gr.update()
                        
                        if not models:
                            return [["", ""]], "⚠️ Keine Modelle gefunden", gr.update()
                        
                        # Format for display
                        model_data = [[m["id"], m.get("name", m["id"])] for m in models]
                        
                        return model_data, f"✅ {len(models)} Modelle geladen", gr.update()
                    
                    def load_user_preferences(provider):
                        """Load user's saved preferences for provider"""
                        if not current_user["id"]:
                            return [["", "", "", ""]], []
                        
                        prefs = get_user_model_preferences(current_user["id"], provider)
                        
                        if not prefs:
                            # No preferences, show default models
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
                        
                        # Load saved preferences
                        display_data = []
                        state_data = []
                        for i, pref in enumerate(prefs, 1):
                            display_data.append([
                                i,
                                pref.model_id,
                                pref.display_name or pref.model_id,
                                "✅" if pref.is_visible else "❌"
                            ])
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
                    
                    def save_preferences(provider, current_state):
                        """Save preferences to database"""
                        if not current_user["id"]:
                            return "❌ Bitte anmelden"
                        
                        if not current_state:
                            return "⚠️ Keine Modelle ausgewählt"
                        
                        # Update display_order
                        for i, model in enumerate(current_state):
                            model["display_order"] = i
                        
                        success, message = save_user_model_preferences(
                            current_user["id"],
                            provider,
                            current_state
                        )
                        
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
                        inputs=[pref_provider],
                        outputs=[available_models_list, fetch_status, selected_models_display]
                    )
                    
                    pref_provider.change(
                        load_user_preferences,
                        inputs=[pref_provider],
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
                        inputs=[pref_provider, selected_models_state],
                        outputs=[save_prefs_status]
                    )
                    
                    reset_prefs_btn.click(
                        reset_to_defaults,
                        inputs=[pref_provider],
                        outputs=[selected_models_display, selected_models_state, save_prefs_status]
                    )

    # Login/Logout handlers
    def handle_login(username, password):
        success, message, show_app, show_login = login_user(username, password)
        status = f"👤 Angemeldet als: **{current_user['username']}**"
        
        # Show admin tab only for admins
        show_admin_tab = current_user.get("is_admin", False)
        
        return message, show_app, show_login, status, gr.update(visible=True), gr.update(visible=show_admin_tab)

    def handle_logout():
        message, show_app, show_login = logout_user()
        return message, show_app, show_login, "👤 Nicht angemeldet", gr.update(visible=False), gr.update(visible=False)

    login_btn.click(
        handle_login,
        [login_username, login_password],
        [login_message, main_app, login_screen, login_status, logout_btn, admin_tab]  # Add admin_tab
    )

    logout_btn.click(
        handle_logout,
        outputs=[login_message, main_app, login_screen, login_status, logout_btn, admin_tab]  # Add admin_tab
    )

# ==========================================
# 🚀 LAUNCH CONFIGURATION
# ==========================================
if __name__ == "__main__":

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
            os.chmod(LOG_FILE, 0o666) # Writable for everyone
        except Exception as e:
            print(f"⚠️ Could not create log file: {e}")

    print(f"🚀 Starting Server on Port 7860...")
    print(f"📂 Serving files from: {APP_DIR}")

    # 3. Launch with consolidated allowed_paths
    demo.queue(
        default_concurrency_limit=10,
        max_size=50,
    ).launch(
        server_name="0.0.0.0",
        server_port=7860,
        auth=None,
        share=False,
        debug=True,
        allowed_paths=[
            APP_DIR,      # Allows manifest.json, pwa.js, custom.css
            STATIC_DIR,   # Allows /static/icon-192.png etc.
            IMAGES_DIR,   # Allows viewing generated images
            "/tmp/gradio" # Allows internal Gradio processing
        ],
        max_file_size="1000mb",
        show_error=True
    )
