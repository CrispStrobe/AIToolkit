#!/usr/bin/env python3
import os
import sys
import json
import pickle
import argparse
import base64
import getpass
import logging
import io
from datetime import datetime

# Database Imports
from sqlalchemy import Column, Integer, String, Text, ForeignKey, Table, create_engine, Boolean
from sqlalchemy.orm import relationship, sessionmaker

# YouTube API Imports
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Internal App Imports
try:
    from app import Base, engine, SessionLocal, User, authenticate_user
    from crypto_utils import crypto, key_wrapper
except ImportError as e:
    print(f"❌ CRITICAL ERROR: Missing dependency or module: {e}")
    sys.exit(1)

# ==========================================
# 🗄️ DATABASE MIGRATION & MODELS
# ==========================================

channel_access = Table(
    'channel_access', Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id'), primary_key=True),
    Column('credential_id', Integer, ForeignKey('youtube_credentials.id'), primary_key=True),
    extend_existing=True
)

class YouTubeCredential(Base):
    __tablename__ = 'youtube_credentials'
    id = Column(Integer, primary_key=True)
    channel_alias = Column(String(100), unique=True, nullable=False)
    encrypted_secrets = Column(Text, nullable=False)
    encrypted_token = Column(Text, nullable=True)
    owner_id = Column(Integer, ForeignKey('users.id'))
    is_active = Column(Boolean, default=True)

    owner = relationship("User", backref="owned_channels")
    authorized_users = relationship("User", secondary=channel_access, backref="shared_channels")
    extend_existing = True

def run_migrations():
    """Ensures the YouTube tables exist in the database."""
    Base.metadata.create_all(bind=engine)

# ==========================================
# 🔐 SECURE TOOLKIT LOGIC
# ==========================================

class SecureYouTubeToolkit:
    def __init__(self, safe_user, umk, alias):
        self.user = safe_user
        self.umk = umk
        self.alias = alias
        self.youtube = self._load_session()

    def _load_session(self):
        db = SessionLocal()
        try:
            cred = db.query(YouTubeCredential).filter(YouTubeCredential.channel_alias == self.alias).first()
            if not cred:
                raise ValueError(f"No credential found for alias '{self.alias}'")
            
            # Access Control
            is_authorized = (cred.owner_id == self.user.id or any(u.id == self.user.id for u in cred.authorized_users))
            if not is_authorized:
                raise PermissionError(f"Access denied for channel '{self.alias}'")

            secrets_raw = crypto.decrypt_text(cred.encrypted_secrets, key=self.umk)
            client_config = json.loads(secrets_raw)

            creds = None
            if cred.encrypted_token:
                token_b64 = crypto.decrypt_text(cred.encrypted_token, key=self.umk)
                if token_b64 != "[Decryption Failed]":
                    creds = pickle.loads(base64.b64decode(token_b64))

            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    logging.info("🔄 Refreshing token...")
                    creds.refresh(Request())
                else:
                    logging.info("🔑 New authentication required...")
                    flow = InstalledAppFlow.from_client_config(client_config, ['https://www.googleapis.com/auth/youtube.force-ssl', 'https://www.googleapis.com/auth/youtube.upload'])
                    creds = flow.run_local_server(port=0, prompt='consent', access_type='offline')
                
                self._vault_token(db, cred, creds)

            return build('youtube', 'v3', credentials=creds)
        finally:
            db.close()

    def _vault_token(self, db, cred_entry, creds):
        pickle_data = base64.b64encode(pickle.dumps(creds)).decode('utf-8')
        cred_entry.encrypted_token = crypto.encrypt_text(pickle_data, key=self.umk)
        db.commit()

    def upload_video(self, file_path, title, description):
        """Robust upload using 1MB chunks to prevent 99% hangs."""
        body = {'snippet': {'title': title, 'description': description, 'categoryId': '22'}, 'status': {'privacyStatus': 'private', 'selfDeclaredMadeForKids': False}}
        media = MediaFileUpload(file_path, mimetype='video/mp4', chunksize=1024*1024, resumable=True)
        request = self.youtube.videos().insert(part="snippet,status", body=body, media_body=media)
        
        logging.info(f"🚀 Uploading {file_path}...")
        response = None
        while response is None:
            status, response = request.next_chunk()
            if status: print(f"    Progress: {int(status.progress() * 100)}%", end="\r")
        print(f"\n✅ Success! Video ID: {response['id']}")

# ==========================================
# 🛠️ VAULT ADMINISTRATION
# ==========================================

def vault_secrets_and_pickle(safe_user, umk, alias, secrets_path, pickle_path=None):
    """Encrypts and stores both JSON secrets and (optionally) an existing pickle."""
    if not os.path.exists(secrets_path):
        print(f"❌ Secrets file not found: {secrets_path}")
        return

    with open(secrets_path, 'r') as f:
        secrets_data = f.read()
    enc_secrets = crypto.encrypt_text(secrets_data, key=umk)

    enc_token = None
    if pickle_path and os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            pickle_raw = f.read()
        token_b64 = base64.b64encode(pickle_raw).decode('utf-8')
        enc_token = crypto.encrypt_text(token_b64, key=umk)
        print(f"📦 Found existing pickle: {pickle_path}")

    db = SessionLocal()
    try:
        # Update if exists, else create
        entry = db.query(YouTubeCredential).filter(YouTubeCredential.channel_alias == alias).first()
        if entry:
            entry.encrypted_secrets = enc_secrets
            if enc_token: entry.encrypted_token = enc_token
            print(f"✅ Updated existing vault entry: {alias}")
        else:
            entry = YouTubeCredential(channel_alias=alias, encrypted_secrets=enc_secrets, encrypted_token=enc_token, owner_id=safe_user.id)
            db.add(entry)
            print(f"✅ Created new vault entry: {alias}")
        db.commit()
    finally:
        db.close()

def share_access(safe_user, alias, target_username):
    db = SessionLocal()
    try:
        cred = db.query(YouTubeCredential).filter(YouTubeCredential.channel_alias == alias, YouTubeCredential.owner_id == safe_user.id).first()
        target = db.query(User).filter(User.username == target_username).first()
        if target and cred and target not in cred.authorized_users:
            cred.authorized_users.append(target)
            db.commit()
            print(f"✅ Shared {alias} with {target_username}")
    finally:
        db.close()

# ==========================================
# 💻 CLI ENTRY
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Secure YouTube Vault Toolkit")
    parser.add_argument("--user", required=True)
    parser.add_argument("--alias", required=True)
    parser.add_argument("-v", "--verbose", action="store_true")
    
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("whoami")
    
    up = subparsers.add_parser("upload")
    up.add_argument("--file", required=True)

    # NEW: The command you requested for migration
    vlt = subparsers.add_parser("vault-pickle")
    vlt.add_argument("--secrets", required=True)
    vlt.add_argument("--pickle", help="Path to local .pickle file")

    sh = subparsers.add_parser("share")
    sh.add_argument("--target", required=True)

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format='[%(levelname)s] %(message)s')
    run_migrations()

    safe_user, umk = authenticate_user(args.user, getpass.getpass(f"Password for {args.user}: "))
    if not safe_user: sys.exit(1)

    if args.command == "vault-pickle":
        vault_secrets_and_pickle(safe_user, umk, args.alias, args.secrets, args.pickle)
    elif args.command == "share":
        share_access(safe_user, args.alias, args.target)
    elif args.command == "upload":
        tk = SecureYouTubeToolkit(safe_user, umk, args.alias)
        tk.upload_video(args.file, os.path.basename(args.file), "Uploaded via Vault Toolkit")
    elif args.command == "whoami":
        tk = SecureYouTubeToolkit(safe_user, umk, args.alias)
        res = tk.youtube.channels().list(part="snippet", mine=True).execute()
        print(f"🔓 Logged into: {res['items'][0]['snippet']['title']}")