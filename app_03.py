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

# Set Gradio file upload limits
os.environ['GRADIO_TEMP_DIR'] = '/tmp/gradio'
os.makedirs('/tmp/gradio', exist_ok=True)

# Increase max file size
gr.set_static_paths(paths=["/var/www/transkript_app/static"])

# --- IMPORTS & SETUP ---
try:
    from PIL import Image
    import openai
except ImportError:
    print("⚠️ WARNUNG: Bitte 'pip install openai pillow' ausführen.")

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

# ==========================================
# 🗄️ DATABASE SETUP
# ==========================================

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, Boolean
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
                username="admin",
                password_hash=hash_password("akademie2025"),
                email="stroebele@akademie-rs.de",
                is_admin=True
            )
            db.add(admin)

            # Create regular user
            user = User(
                username="user",
                password_hash=hash_password("dialog2025"),
                email="dialog@akademie-rs.de",
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
# ⚙️ ZENTRALE KONFIGURATION
# ==========================================

# API Keys (aus Environment oder direkt hier eintragen)
API_KEYS = {
    "SCALEWAY": os.environ.get("SCALEWAY_API_KEY", ""),
    "NEBIUS": os.environ.get("NEBIUS_API_KEY", ""),
    "MISTRAL": os.environ.get("MISTRAL_API_KEY", ""),
    "GLADIA": os.environ.get("GLADIA_API_KEY", ""),
    "OPENROUTER": os.environ.get("OPENROUTER_API_KEY", ""),
    "GROQ": os.environ.get("GROQ_API_KEY", ""),
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
        "vision_models": ["pixtral-large-2411", "pixtral-12b-2409", "mistral-ocr-latest"]
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

        messages = [{"role": "system", "content": system_prompt}]

        for msg in history:
            if isinstance(msg, dict):
                messages.append({"role": msg["role"], "content": str(msg["content"])})
            else:
                # Fallback for old tuple format (u, a)
                messages.append({"role": "user", "content": str(msg[0])})
                messages.append({"role": "assistant", "content": str(msg[1])})

        messages.append({"role": "user", "content": message})

        stream = client.chat.completions.create(
            model=model, messages=messages, temperature=temp, stream=True, max_tokens=2048
        )

        partial = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                partial += chunk.choices[0].delta.content
                yield partial

    except Exception as e:
        yield f"🔥 Fehler: {str(e)}"

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

        response = client.chat.completions.create(
            model=model, messages=messages, max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"🔥 Vision Fehler: {str(e)}"

# ==========================================
# 3. TRANSKRIPTION (HYBRID: GLADIA V6 + WHISPER) - COMPLETE FIX
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

def run_transcription(audio, provider, model, lang, whisper_temp, whisper_prompt, diar, trans, target, key):
    """
    Unified transcription function with comprehensive logging
    - lang: used by BOTH Gladia and Whisper
    - whisper_temp, whisper_prompt: only used by Whisper
    - diar, trans, target: only used by Gladia
    """
    logger.info("=" * 80)
    logger.info("TRANSCRIPTION START")
    logger.info(f"Provider: {provider}")
    logger.info(f"Model: {model}")
    logger.info(f"Language: {lang}")
    logger.info(f"Audio file: {audio}")
    logger.info(f"Whisper temp: {whisper_temp}")
    logger.info(f"Whisper prompt: {whisper_prompt}")
    logger.info(f"Diarization: {diar}")
    logger.info(f"Translation: {trans}")
    logger.info(f"Target lang: {target}")
    logger.info(f"API key provided: {bool(key)}")
    logger.info("=" * 80)

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

        if file_size > 100 * 1024 * 1024:  # 100MB
            logger.warning(f"Large audio file: {file_size/1024/1024:.2f} MB")
    except Exception as size_error:
        logger.exception(f"Error checking file size: {str(size_error)}")

    # PFAD A: Standard Whisper (Scaleway/Groq)
    if provider != "Gladia":
        logger.info(f"Using Whisper provider: {provider}")

        try:
            logs = f"🚀 Start {provider} Whisper..."
            logger.info(f"Starting Whisper transcription with {provider}")
            yield logs, "", ""

            # Get API client
            logger.info("Creating API client...")
            try:
                client = get_client(provider, key)
                logger.info(f"API client created successfully for {provider}")
            except Exception as client_error:
                logger.exception(f"Failed to create API client: {str(client_error)}")
                yield f"🔥 API-Client Fehler: {str(client_error)}\n\nTyp: {type(client_error).__name__}", "", ""
                return

            # Model selection
            if not model:
                available_models = PROVIDERS.get(provider, {}).get("audio_models", [])
                model = available_models[0] if available_models else "whisper-large-v3"
                logger.info(f"No model specified, using default: {model}")
            else:
                logger.info(f"Using specified model: {model}")

            logs += f"\n🎯 Modell: {model}"
            yield logs, "", ""

            # Build Whisper parameters
            logger.info("Building Whisper API parameters...")

            try:
                file_handle = open(audio, "rb")
                logger.info(f"Opened audio file for reading: {audio}")
            except Exception as file_error:
                logger.exception(f"Failed to open audio file: {str(file_error)}")
                yield f"🔥 Datei-Fehler: {str(file_error)}", "", ""
                return

            whisper_params = {
                "model": model,
                "file": file_handle,
            }

            # Response format
            if provider == "Groq":
                whisper_params["response_format"] = "verbose_json"
                logger.info("Using response format: verbose_json (Groq)")
            else:
                whisper_params["response_format"] = "json"
                logger.info("Using response format: json (Scaleway)")

            # Language parameter
            if lang and lang != "auto":
                whisper_params["language"] = lang
                logs += f"\n🌐 Sprache: {lang.upper()}"
                logger.info(f"Language set to: {lang}")
            else:
                logs += f"\n🌐 Sprache: Auto-Erkennung"
                logger.info("Language: auto-detection")
            yield logs, "", ""

            # Temperature
            if whisper_temp and whisper_temp > 0:
                whisper_params["temperature"] = whisper_temp
                logs += f"\n🌡️ Temperatur: {whisper_temp}"
                logger.info(f"Temperature set to: {whisper_temp}")
                yield logs, "", ""

            # Prompt for context
            if whisper_prompt and whisper_prompt.strip():
                whisper_params["prompt"] = whisper_prompt.strip()
                logs += f"\n📝 Kontext-Prompt aktiv"
                logger.info(f"Context prompt provided: {whisper_prompt[:50]}...")
                yield logs, "", ""

            logs += f"\n⏳ Transkribiere..."
            logger.info("Starting Whisper API call...")
            logger.debug(f"API parameters: {list(whisper_params.keys())}")
            yield logs, "", ""

            # Call Whisper API
            try:
                logger.info("Calling client.audio.transcriptions.create()...")
                start_time = time.time()

                res = client.audio.transcriptions.create(**whisper_params)

                elapsed = time.time() - start_time
                logger.info(f"Whisper API call completed in {elapsed:.2f} seconds")
                logger.debug(f"Response type: {type(res)}")
                logger.debug(f"Response attributes: {dir(res)}")

            except Exception as api_error:
                file_handle.close()
                logger.exception(f"Whisper API call failed: {str(api_error)}")
                logger.error(f"Error type: {type(api_error).__name__}")
                logger.error(f"Error details: {str(api_error)}")
                yield f"🔥 API-Fehler ({provider}/{model}): {str(api_error)}\n\nTyp: {type(api_error).__name__}\n\nBitte Log-Datei prüfen", "", ""
                return
            finally:
                # Always close file handle
                try:
                    file_handle.close()
                    logger.info("Audio file handle closed")
                except:
                    pass

            # Process response
            logger.info("Processing Whisper response...")
            result_text = ""
            detected_lang = ""

            # Check for segments (verbose_json format - Groq)
            if hasattr(res, 'segments') and res.segments:
                logger.info(f"Response contains {len(res.segments)} segments")
                try:
                    segment_count = 0
                    for seg in res.segments:
                        start = seg.start if hasattr(seg, 'start') else 0
                        text = seg.text if hasattr(seg, 'text') else ''
                        text = text.strip()

                        if text:
                            result_text += f"[{format_duration(start)}] {text}\n"
                            segment_count += 1

                    logs += f"\n📊 {segment_count} Segmente verarbeitet"
                    logger.info(f"Successfully processed {segment_count} segments")

                except Exception as seg_error:
                    logger.exception(f"Error formatting segments: {str(seg_error)}")
                    # Fallback to plain text
                    result_text = res.text if hasattr(res, 'text') else str(res)
                    logger.info("Falling back to plain text format")
            else:
                # Simple text format (Scaleway, or fallback)
                logger.info("Response in simple text format (no segments)")
                result_text = res.text if hasattr(res, 'text') else str(res)

            logger.info(f"Final transcript length: {len(result_text)} characters")

            if not result_text or result_text.strip() == "":
                logger.warning("Transcription result is empty!")
                yield "⚠️ Warnung: Transkription ist leer", "", ""
                return

            # Get detected language
            if hasattr(res, 'language'):
                detected_lang = f"\n🔍 Erkannt: {res.language.upper()}"
                logger.info(f"Detected language: {res.language}")

            logger.info("Whisper transcription completed successfully")
            logger.info(f"Result preview: {result_text[:100]}...")

            yield f"{logs}{detected_lang}\n✅ Fertig!", result_text, "(Whisper: Keine Übersetzung verfügbar)"

        except Exception as e:
            logger.exception(f"Unexpected error in Whisper transcription: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error details: {str(e)}")

            import traceback
            tb_str = traceback.format_exc()
            logger.error(f"Full traceback:\n{tb_str}")

            yield f"🔥 Fehler ({provider}/{model}): {str(e)}\n\nTyp: {type(e).__name__}\n\nDetails in Log-Datei", "", ""

        return

    # PFAD B: Gladia V2 (Advanced)
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

                if elapsed > 600:  # 10 minute timeout
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
# 4. BILDGENERIERUNG (NEBIUS FLUX)
# ==========================================

def run_image_gen(prompt, model, width, height, steps, key):
    try:
        client = get_client("Nebius", key)

        response = client.images.generate(
            model=model, prompt=prompt, response_format="b64_json",
            extra_body={"response_extension": "jpg", "width": width, "height": height, "num_inference_steps": steps}
        )

        img_data = base64.b64decode(response.data[0].b64_json)
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        tfile.write(img_data)
        tfile.close()
        return tfile.name, "✅ Erfolg"
    except Exception as e:
        return None, f"🔥 Fehler: {str(e)}"

# Add this BEFORE the GUI builder section (after the image generation function)

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
            with gr.TabItem("💬 Chat"):
                with gr.Row():
                    with gr.Column(scale=3):
                        with gr.Row():
                            c_prov = gr.Dropdown(list(PROVIDERS.keys()), value="Scaleway", label="Anbieter")
                            c_model = gr.Dropdown(PROVIDERS["Scaleway"]["chat_models"], value=PROVIDERS["Scaleway"]["chat_models"][0], label="Modell")

                        c_badge = gr.HTML(value=PROVIDERS["Scaleway"]["badge"])

                        def update_c_ui(prov):
                            p_data = PROVIDERS.get(prov, {})
                            ms = p_data.get("chat_models", [])
                            return gr.update(choices=ms, value=ms[0] if ms else ""), p_data.get("badge", "")

                        c_prov.change(update_c_ui, c_prov, [c_model, c_badge])

                        c_bot = gr.Chatbot(height=500, type="messages")
                        c_msg = gr.Textbox(placeholder="Nachricht...", show_label=False)

                        with gr.Row():
                            c_btn = gr.Button("📤 Senden", variant="primary", scale=3)
                            c_save_btn = gr.Button("💾 Chat speichern", scale=1)
                            c_clear_btn = gr.Button("🗑️ Löschen", scale=1)

                        c_save_status = gr.Markdown("")

                    with gr.Column(scale=1):
                        with gr.Accordion("⚙️ Einstellungen", open=True):
                            c_key = gr.Textbox(label="API Key (Optional)", type="password")
                            c_sys = gr.Textbox(label="System Rolle", value="Du bist ein hilfreicher Assistent.", lines=4)
                            c_temp = gr.Slider(0, 2, value=0.7, label="Kreativität")

                        with gr.Accordion("📚 Alte Chats laden", open=False):
                            refresh_chats_btn = gr.Button("🔄 Aktualisieren")
                            old_chats = gr.Dataframe(
                                headers=["ID", "Datum", "Titel", "Modell"],
                                label="Gespeicherte Chats",
                                interactive=False
                            )
                            with gr.Row():
                                load_chat_id = gr.Number(label="Chat-ID", precision=0)
                                load_chat_btn = gr.Button("📥 Laden")
                                delete_chat_btn = gr.Button("🗑️", scale=0)
                            chat_load_status = gr.Markdown("")

                        with gr.Accordion("📎 Inhalt anhängen", open=False):
                            attach_type = gr.Radio(
                                ["Transkript", "Vision-Ergebnis", "Eigener Text"],
                                value="Transkript",
                                label="Typ"
                            )
                            attach_id = gr.Number(label="ID (für Transkript/Vision)", precision=0, visible=True)
                            attach_custom = gr.Textbox(label="Eigener Text", lines=5, visible=False)
                            attach_btn = gr.Button("📎 Anhängen", variant="secondary")
                            attach_status = gr.Markdown("")

                            def toggle_attach_inputs(attach_type):
                                if attach_type == "Eigener Text":
                                    return gr.update(visible=False), gr.update(visible=True)
                                return gr.update(visible=True), gr.update(visible=False)

                            attach_type.change(toggle_attach_inputs, attach_type, [attach_id, attach_custom])

                # Chat functions
                def user_msg(msg, hist):
                    return "", hist + [{"role": "user", "content": msg}]

                def bot_msg(hist, prov, mod, temp, sys, key):
                    if not hist or len(hist) == 0:
                        return hist

                    user_messages = [m for m in hist if m["role"] == "user"]
                    if not user_messages:
                        return hist

                    last_user_msg = user_messages[-1]["content"]
                    hist.append({"role": "assistant", "content": ""})

                    for chunk in run_chat(last_user_msg, hist[:-1], prov, mod, temp, sys, key):
                        hist[-1]["content"] = chunk
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

                def load_single_chat(chat_id):
                    """Load a specific chat"""
                    if not current_user["id"] or not chat_id:
                        return None, "❌ Ungültige ID"

                    chat = get_single_chat(int(chat_id), current_user["id"])
                    if chat:
                        messages = json.loads(chat.messages)
                        return messages, f"✅ Chat '{chat.title}' geladen"
                    return None, "❌ Chat nicht gefunden"

                def delete_chat(chat_id):
                    """Delete a chat"""
                    if not current_user["id"] or not chat_id:
                        return "❌ Ungültige ID", load_chat_list()

                    if delete_chat_history(int(chat_id), current_user["id"]):
                        return "✅ Chat gelöscht", load_chat_list()
                    return "❌ Fehler beim Löschen", load_chat_list()

                def attach_content_to_chat(hist, attach_type, attach_id, custom_text):
                    """Attach content to current chat"""
                    if not current_user["id"]:
                        return hist, "❌ Bitte anmelden"

                    content_to_add = ""

                    if attach_type == "Transkript":
                        if not attach_id:
                            return hist, "❌ Transkript-ID erforderlich"

                        db = get_db()
                        trans = db.query(Transcription).filter(
                            Transcription.id == int(attach_id),
                            Transcription.user_id == current_user["id"]
                        ).first()
                        db.close()

                        if trans:
                            content_to_add = f"[Transkript #{trans.id}]\n\n{trans.original_text}"
                        else:
                            return hist, "❌ Transkript nicht gefunden"

                    elif attach_type == "Vision-Ergebnis":
                        if not attach_id:
                            return hist, "❌ Vision-ID erforderlich"

                        db = get_db()
                        vision = db.query(VisionResult).filter(
                            VisionResult.id == int(attach_id),
                            VisionResult.user_id == current_user["id"]
                        ).first()
                        db.close()

                        if vision:
                            content_to_add = f"[Vision-Analyse #{vision.id}]\nPrompt: {vision.prompt}\n\nErgebnis:\n{vision.result}"
                        else:
                            return hist, "❌ Vision-Ergebnis nicht gefunden"

                    elif attach_type == "Eigener Text":
                        if not custom_text:
                            return hist, "❌ Text erforderlich"
                        content_to_add = custom_text

                    # Add to chat
                    if not hist:
                        hist = []
                    hist.append({"role": "user", "content": content_to_add})

                    return hist, f"✅ {attach_type} angehängt"

                def clear_chat():
                    """Clear current chat"""
                    return [], ""

                # Connect handlers
                c_msg.submit(user_msg, [c_msg, c_bot], [c_msg, c_bot], queue=False).then(
                    bot_msg, [c_bot, c_prov, c_model, c_temp, c_sys, c_key], c_bot
                )
                c_btn.click(user_msg, [c_msg, c_bot], [c_msg, c_bot], queue=False).then(
                    bot_msg, [c_bot, c_prov, c_model, c_temp, c_sys, c_key], c_bot
                )

                c_save_btn.click(save_chat, [c_bot, c_prov, c_model], c_save_status)
                c_clear_btn.click(clear_chat, outputs=[c_bot, c_save_status])

                refresh_chats_btn.click(load_chat_list, outputs=old_chats)
                load_chat_btn.click(load_single_chat, load_chat_id, [c_bot, chat_load_status])
                delete_chat_btn.click(delete_chat, load_chat_id, [chat_load_status, old_chats])

                attach_btn.click(
                    attach_content_to_chat,
                    [c_bot, attach_type, attach_id, attach_custom],
                    [c_bot, attach_status]
                )

            # --- TAB 2: TRANSKRIPTION - WITH WHISPER OPTIONS ---
            with gr.TabItem("🎙️ Transkription"):
                with gr.Row():
                    with gr.Column():
                        t_audio = gr.Audio(type="filepath", label="Datei")
                        t_prov = gr.Radio(["Gladia", "Scaleway", "Groq"], value="Gladia", label="Engine")
                        t_model = gr.Dropdown(choices=[], value=None, visible=False, label="Modell")
                        t_badge = gr.HTML(value=get_compliance_html("Gladia"))

                        # GLADIA OPTIONS (visible by default)
                        gladia_opts = gr.Accordion("⚙️ Gladia Optionen", open=True, visible=True)
                        with gladia_opts:
                            t_lang = gr.Dropdown(
                                [("Auto-Erkennung", "auto"), ("Deutsch", "de"), ("Englisch", "en"), ("Französisch", "fr"), ("Spanisch", "es"), ("Italienisch", "it")],
                                value="de",
                                label="Sprache"
                            )
                            t_diar = gr.Checkbox(True, label="🎭 Sprecher erkennen (Diarization)")
                            t_trans = gr.Checkbox(False, label="🌍 Übersetzen")
                            t_target = gr.Dropdown(
                                [("Deutsch", "de"), ("Englisch", "en"), ("Französisch", "fr"), ("Spanisch", "es")],
                                value="en",
                                label="Zielsprache"
                            )

                        # WHISPER OPTIONS (hidden by default)
                        whisper_opts = gr.Accordion("⚙️ Whisper Optionen", open=True, visible=False)
                        with whisper_opts:
                            w_lang = gr.Dropdown(
                                [("Auto-Erkennung", "auto"), ("Deutsch", "de"), ("Englisch", "en"), ("Französisch", "fr"), ("Spanisch", "es"), ("Italienisch", "it")],
                                value="de",
                                label="🌐 Sprache (WICHTIG: 'de' für deutsche Audios!)"
                            )
                            w_temp = gr.Slider(
                                0, 1,
                                value=0,
                                step=0.1,
                                label="🌡️ Temperatur (0 = präzise, 1 = kreativ)",
                                info="Höhere Werte können bei undeutlicher Audio helfen"
                            )
                            w_prompt = gr.Textbox(
                                label="📝 Kontext-Prompt (Optional)",
                                placeholder="z.B.: 'Vortrag über Theologie, Akademie, Diözese, Liturgie'",
                                lines=2,
                                info="Hilft bei Fachbegriffen und Namen"
                            )
                            gr.Markdown("""
                            💡 **Tipps für bessere Ergebnisse:**
                            - Wähle die **korrekte Sprache** (nicht 'auto' für Deutsch!)
                            - Bei Namen/Fachbegriffen: Prompt nutzen
                            - `whisper-large-v3-turbo` ist schneller als `whisper-large-v3`
                            """)

                        t_key = gr.Textbox(label="🔑 API Key (Optional)", type="password")
                        t_btn = gr.Button("▶️ Transkription starten", variant="primary", size="lg")
                        t_log = gr.Textbox(label="📋 Log", lines=5)

                    with gr.Column():
                        t_orig = gr.Textbox(label="📄 Original Transkript", lines=15, show_copy_button=True)
                        t_trsl = gr.Textbox(label="🌍 Übersetzung", lines=15, show_copy_button=True)

                        with gr.Row():
                            t_save_btn = gr.Button("💾 Transkript speichern", variant="secondary")
                            t_save_status = gr.Markdown("")

                # NEW: Send to Chat Section
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

                # Update UI based on provider selection
                def update_t_ui(prov):
                    badge = get_compliance_html(prov)
                    is_whisper = prov != "Gladia"
                    ms = PROVIDERS.get(prov, {}).get("audio_models", [])

                    return (
                        badge,
                        gr.update(visible=is_whisper, choices=ms, value=ms[0] if ms else None),
                        gr.update(visible=not is_whisper),
                        gr.update(visible=is_whisper)
                    )

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

                # Connect manual save button
                t_save_btn.click(
                    manual_save_transcription,
                    inputs=[t_orig, t_trsl, t_prov, t_model, t_lang],
                    outputs=t_save_status
                )


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
                        v_prov = gr.Dropdown(["Scaleway", "Mistral", "Nebius", "OpenRouter"], value="Scaleway", label="Provider")
                        v_model = gr.Dropdown(PROVIDERS["Scaleway"]["vision_models"], value="pixtral-12b-2409", label="Modell", allow_custom_value=True)
                        v_key = gr.Textbox(label="Key (Optional)", type="password")
                        v_prompt = gr.Textbox(label="Frage", value="Beschreibe dieses Bild detailliert.")
                        v_btn = gr.Button("Analysieren", variant="primary")
                    with gr.Column():
                        v_out = gr.Markdown(label="Ergebnis")

                def update_v_models(prov):
                    ms = PROVIDERS.get(prov, {}).get("vision_models", [])
                    return gr.update(choices=ms, value=ms[0] if ms else "")

                v_prov.change(update_v_models, v_prov, v_model)
                v_btn.click(run_vision, [v_img, v_prompt, v_prov, v_model, v_key], v_out)

            # --- TAB 4: BILDERZEUGUNG ---
            with gr.TabItem("🎨 Bilderzeugung"):
                with gr.Row():
                    with gr.Column():
                        g_prompt = gr.Textbox(label="Prompt", placeholder="Eine futuristische Kirche...", lines=3)
                        g_model = gr.Dropdown(
                            PROVIDERS["Nebius"]["image_models"],
                            value="black-forest-labs/flux-schnell",
                            label="Modell (Nebius)"
                        )
                        with gr.Row():
                            g_w = gr.Slider(256, 1024, value=1024, step=64, label="Breite")
                            g_h = gr.Slider(256, 1024, value=768, step=64, label="Höhe")
                        g_steps = gr.Slider(4, 16, value=10, label="Schritte (max 16)")
                        g_key = gr.Textbox(label="Nebius Key (Optional)", type="password")
                        g_btn = gr.Button("Generieren", variant="primary")
                        g_stat = gr.Textbox(label="Status", interactive=False)

                        # ADD SAVE BUTTON
                        with gr.Row():
                            g_save_btn = gr.Button("💾 Bild speichern", variant="secondary", visible=False)
                            g_save_status = gr.Markdown("")

                    with gr.Column():
                        g_out = gr.Image(label="Ergebnis")

                # Store the generated image path in a state variable
                g_img_path = gr.State(value=None)

                # Update image generation to show save button and store path
                def generate_and_show_save(prompt, model, width, height, steps, key):
                    img_path, status = run_image_gen(prompt, model, width, height, steps, key)

                    if img_path:
                        # Show save button
                        return img_path, status, img_path, gr.update(visible=True), ""
                    else:
                        return None, status, None, gr.update(visible=False), ""

                g_btn.click(
                    generate_and_show_save,
                    [g_prompt, g_model, g_w, g_h, g_steps, g_key],
                    [g_out, g_stat, g_img_path, g_save_btn, g_save_status]
                )

                # save function
                def save_generated_image_to_db(img_path, prompt, model):
                    """Save generated image to database"""
                    try:
                        if not current_user["id"]:
                            return "❌ Bitte anmelden", gr.update(visible=True)

                        if not img_path:
                            return "❌ Kein Bild zum Speichern", gr.update(visible=True)

                        # Copy to permanent location
                        import shutil
                        permanent_dir = "/var/www/transkript_app/generated_images"
                        os.makedirs(permanent_dir, exist_ok=True)

                        filename = f"img_{int(time.time())}_{os.path.basename(img_path)}"
                        permanent_path = os.path.join(permanent_dir, filename)
                        shutil.copy2(img_path, permanent_path)

                        img_id = save_generated_image(
                            user_id=current_user["id"],
                            provider="Nebius",
                            model=model,
                            prompt=prompt,
                            image_path=permanent_path
                        )

                        return f"✅ Bild gespeichert (ID: {img_id})", gr.update(visible=False)

                    except Exception as e:
                        logger.exception(f"Error saving generated image: {str(e)}")
                        return f"🔥 Fehler: {str(e)}", gr.update(visible=True)

                # Connect save button
                g_save_btn.click(
                    save_generated_image_to_db,
                    inputs=[g_img_path, g_prompt, g_model],
                    outputs=[g_save_status, g_save_btn]
                )

            # --- TAB 5: VERLAUF & VERWALTUNG ---
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

                # --- AUTO-LOAD ON TAB SWITCH ---
                trans_tab.select(fn=load_trans_data, outputs=[trans_history, trans_state])
                images_tab.select(fn=load_img_data, outputs=[images_history, img_state])
                prompts_tab.select(fn=load_prompts_data, outputs=[saved_prompts, prompt_state])

    # Login/Logout handlers
    def handle_login(username, password):
        success, message, show_app, show_login = login_user(username, password)
        status = f"👤 Angemeldet als: **{current_user['username']}**"
        return message, show_app, show_login, status, gr.update(visible=True)

    def handle_logout():
        message, show_app, show_login = logout_user()
        return message, show_app, show_login, "👤 Nicht angemeldet", gr.update(visible=False)

    login_btn.click(
        handle_login,
        [login_username, login_password],
        [login_message, main_app, login_screen, login_status, logout_btn]
    )

    logout_btn.click(
        handle_logout,
        outputs=[login_message, main_app, login_screen, login_status, logout_btn]
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
