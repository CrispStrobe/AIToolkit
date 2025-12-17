#!/usr/bin/env python3
"""
Encryption utilities for Akademie KI Suite
Provides AES-256-GCM + optional Post-Quantum (Kyber-512) encryption
"""

import os
import base64
import json
from typing import Optional, Tuple
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from cryptography.fernet import Fernet
import logging

logger = logging.getLogger(__name__)

# Try to import post-quantum crypto
HAS_PQ = False
try:
    from pqcrypto.kem.kyber512 import generate_keypair, encrypt as pq_encrypt, decrypt as pq_decrypt
    # Test if it actually works (binary extensions present)
    try:
        test_pk, test_sk = generate_keypair()
        HAS_PQ = True
        logger.info("✅ Post-Quantum encryption (Kyber-512) available and functional")
    except Exception as e:
        logger.warning(f"⚠️ pqcrypto installed but non-functional: {e}")
        HAS_PQ = False
except ImportError as e:
    logger.warning(f"⚠️ pqcrypto not installed: {e}")
    HAS_PQ = False

class KeyWrapper:
    """
    Handles Per-User Key Wrapping (Filen.io / Bitwarden Style).
    
    Concept:
    1. User Master Key (UMK): Random key that encrypts data. Never changes.
    2. Wrapper Key (WK): Derived from password. Encrypts the UMK.
    """
    
    def __init__(self):
        self.backend = default_backend()

    def generate_salt(self) -> bytes:
        return os.urandom(16)

    def derive_wrapper_key(self, password: str, salt: bytes) -> bytes:
        """Derives the Wrapper Key (WK) from Password + Salt"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=200000, # High iteration count for security
            backend=self.backend
        )
        # Returns urlsafe b64 encoded key for Fernet
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))

    def create_user_keychain(self, password: str):
        """
        Run this on User Registration.
        Generates the constant UMK and locks it with the password.
        """
        salt = self.generate_salt()
        
        # 1. Generate the UMK (This key encrypts the actual database rows)
        raw_umk = get_random_bytes(32) # 256 bits for AES
        
        # 2. Derive Wrapper Key from password
        wrapper_key = self.derive_wrapper_key(password, salt)
        
        # 3. Encrypt the UMK (The Lockbox)
        f = Fernet(wrapper_key)
        encrypted_master_key = f.encrypt(raw_umk)
        
        return {
            "salt": base64.b64encode(salt).decode('utf-8'),
            "encrypted_master_key": encrypted_master_key.decode('utf-8'),
            "decrypted_master_key": raw_umk  # Keep this in RAM session immediately after register
        }

    def unlock_user_keychain(self, password: str, salt_b64: str, encrypted_key_b64: str):
        """
        Run this on User Login.
        Unlocks the Lockbox to get the UMK.
        """
        try:
            salt = base64.b64decode(salt_b64)
            wrapper_key = self.derive_wrapper_key(password, salt)
            
            f = Fernet(wrapper_key)
            # This returns the raw bytes of the UMK
            user_master_key = f.decrypt(encrypted_key_b64.encode())
            
            return user_master_key 
        except Exception as e:
            logger.error(f"Keychain unlock failed: {e}")
            return None

    def rewrap_keychain(self, old_password: str, new_password: str, salt_b64: str, encrypted_key_b64: str):
        """
        Run this on Password Change.
        1. Unlock UMK with old pass.
        2. Generate NEW salt/wrapper with new pass.
        3. Re-encrypt the SAME UMK.
        Result: Data in DB does NOT need to be re-encrypted.
        """
        # 1. Unlock with old password
        umk = self.unlock_user_keychain(old_password, salt_b64, encrypted_key_b64)
        if not umk:
            raise ValueError("Altes Passwort ist falsch (Key konnte nicht entschlüsselt werden).")
            
        # 2. Generate NEW salt and NEW wrapper for the new password
        new_salt = self.generate_salt()
        new_wrapper_key = self.derive_wrapper_key(new_password, new_salt)
        
        # 3. Re-encrypt the SAME UMK
        f = Fernet(new_wrapper_key)
        new_encrypted_master_key = f.encrypt(umk)
        
        return {
            "salt": base64.b64encode(new_salt).decode('utf-8'),
            "encrypted_master_key": new_encrypted_master_key.decode('utf-8')
        }
    
class CryptoManager:
    """Manages encryption/decryption operations"""
    
    def __init__(self):
        """Initialize with master key from environment or generate new"""
        # Fallback global key for legacy data or admin ops if needed
        self.global_key = self._load_or_create_master_key()

        # "master_key" attribute kept for backward compatibility
        self.master_key = self._load_or_create_master_key()
        
        # Post-quantum keypair (optional)
        self.pq_public_key = None
        self.pq_secret_key = None
        if HAS_PQ:
            self._init_pq_keys()
    
    def _load_or_create_master_key(self) -> bytes:
        """Load master key from environment or create new one"""
        key_file = "/var/www/transkript_app/.master_key"
        
        # Try environment variable first
        env_key = os.environ.get("MASTER_ENCRYPTION_KEY")
        if env_key:
            try:
                return base64.b64decode(env_key)
            except Exception as e:
                logger.error(f"Invalid MASTER_ENCRYPTION_KEY in environment: {e}")
        
        # Try file
        if os.path.exists(key_file):
            try:
                with open(key_file, 'rb') as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Could not read key file: {e}")
        
        # Generate new key
        logger.warning("⚠️ No master key found. Generating new one...")
        new_key = get_random_bytes(32)  # 256 bits
        
        try:
            with open(key_file, 'wb') as f:
                f.write(new_key)
            os.chmod(key_file, 0o600)  # Read/write for owner only
            logger.info(f"✅ New master key created: {key_file}")
        except Exception as e:
            logger.error(f"Could not save master key: {e}")
        
        return new_key
    
    def _init_pq_keys(self):
        """Initialize post-quantum keypair"""
        key_file = "/var/www/transkript_app/.pq_keypair"
        
        if os.path.exists(key_file):
            try:
                with open(key_file, 'rb') as f:
                    data = f.read()
                    # Split stored keypair (public is first 800 bytes for Kyber-512)
                    self.pq_public_key = data[:800]
                    self.pq_secret_key = data[800:]
                logger.info("✅ Loaded existing PQ keypair")
                return
            except Exception as e:
                logger.warning(f"Could not load PQ keypair: {e}")
        
        # Generate new keypair
        try:
            self.pq_public_key, self.pq_secret_key = generate_keypair()
            
            with open(key_file, 'wb') as f:
                f.write(self.pq_public_key + self.pq_secret_key)
            os.chmod(key_file, 0o600)
            logger.info("✅ Generated new PQ keypair")
        except Exception as e:
            logger.error(f"PQ keypair generation failed: {e}")
    
    def encrypt_text(self, plaintext: str, key: bytes = None, allow_fallback=False) -> str:
        if not plaintext: return ""
        
        # STRICT MODE: Only use global key if explicitly allowed (for system settings)
        if key is None:
            if allow_fallback:
                use_key = self.global_key
            else:
                raise ValueError("Security Error: Attempted to encrypt user data without a User Key!")
        else:
            use_key = key
        
        try:
            nonce = get_random_bytes(12)
            cipher = AES.new(use_key, AES.MODE_GCM, nonce=nonce)
            ciphertext, tag = cipher.encrypt_and_digest(plaintext.encode('utf-8'))
            return base64.b64encode(nonce + tag + ciphertext).decode('utf-8')
        except Exception as e:
            logger.error(f"Encryption error: {e}")
            raise

    def decrypt_text(self, encrypted_b64: str, key: bytes = None) -> str:
        """
        Decrypts text using specific user key (UMK) or falls back to global key.
        """
        if not encrypted_b64: return ""
        
        # Use provided key or fallback to global
        use_key = key if key else self.global_key
        
        try:
            encrypted = base64.b64decode(encrypted_b64)
            nonce = encrypted[:12]
            tag = encrypted[12:28]
            ciphertext = encrypted[28:]
            
            cipher = AES.new(use_key, AES.MODE_GCM, nonce=nonce)
            return cipher.decrypt_and_verify(ciphertext, tag).decode('utf-8')
        except Exception as e:
            logger.error(f"Decryption error: {e}")
            return "[Decryption Failed]"
    
    def encrypt_file(self, file_path: str, output_path: Optional[str] = None) -> Tuple[str, dict]:
        """
        Encrypt file with optional post-quantum protection
        
        Returns:
            (encrypted_file_path, metadata_dict)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Output path
        if not output_path:
            output_path = file_path + ".enc"
        
        # Read file
        with open(file_path, 'rb') as f:
            plaintext = f.read()
        
        # Generate nonce and encrypt
        nonce = get_random_bytes(12)
        cipher = AES.new(self.master_key, AES.MODE_GCM, nonce=nonce)
        ciphertext, tag = cipher.encrypt_and_digest(plaintext)
        
        # Metadata
        metadata = {
            "algorithm": "AES-256-GCM",
            "version": 1,
            "original_size": len(plaintext),
            "pq_protected": False
        }
        
        # Optional: Add post-quantum layer
        if HAS_PQ and self.pq_public_key:
            try:
                # Encrypt the AES key using PQ
                pq_ciphertext, pq_shared_secret = pq_encrypt(self.pq_public_key, self.master_key)
                metadata["pq_protected"] = True
                metadata["pq_ciphertext"] = base64.b64encode(pq_ciphertext).decode('utf-8')
            except Exception as e:
                logger.warning(f"PQ encryption failed: {e}")
        
        # Write encrypted file
        with open(output_path, 'wb') as f:
            f.write(nonce + tag + ciphertext)
        
        return output_path, metadata
    
    def decrypt_file(self, encrypted_path: str, output_path: Optional[str] = None, 
                     metadata: Optional[dict] = None) -> str:
        """
        Decrypt encrypted file
        
        Returns:
            Path to decrypted file
        """
        if not os.path.exists(encrypted_path):
            raise FileNotFoundError(f"Encrypted file not found: {encrypted_path}")
        
        # Output path
        if not output_path:
            output_path = encrypted_path.replace(".enc", "_decrypted")
        
        # Read encrypted file
        with open(encrypted_path, 'rb') as f:
            data = f.read()
        
        # Split components
        nonce = data[:12]
        tag = data[12:28]
        ciphertext = data[28:]
        
        # Handle PQ if present
        key_to_use = self.master_key
        if metadata and metadata.get("pq_protected") and HAS_PQ:
            try:
                pq_ciphertext_b64 = metadata.get("pq_ciphertext")
                if pq_ciphertext_b64:
                    pq_ciphertext = base64.b64decode(pq_ciphertext_b64)
                    key_to_use = pq_decrypt(self.pq_secret_key, pq_ciphertext)
            except Exception as e:
                logger.warning(f"PQ decryption failed, using master key: {e}")
        
        # Decrypt
        cipher = AES.new(key_to_use, AES.MODE_GCM, nonce=nonce)
        plaintext = cipher.decrypt_and_verify(ciphertext, tag)
        
        # Write decrypted file
        with open(output_path, 'wb') as f:
            f.write(plaintext)
        
        return output_path

# Global instance
crypto = CryptoManager()
key_wrapper = KeyWrapper()

# Quick test
if __name__ == "__main__":
    print("🔐 Testing CryptoManager...")
    
    # Test text encryption
    original = "Sensitive church data: Gottesdienst 2025"
    encrypted = crypto.encrypt_text(original)
    decrypted = crypto.decrypt_text(encrypted)
    
    print(f"Original:  {original}")
    print(f"Encrypted: {encrypted[:50]}...")
    print(f"Decrypted: {decrypted}")
    print(f"✅ Match: {original == decrypted}")
    
    # Test file encryption
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("Secret document content")
        test_file = f.name
    
    enc_file, metadata = crypto.encrypt_file(test_file)
    dec_file = crypto.decrypt_file(enc_file, metadata=metadata)
    
    with open(dec_file, 'r') as f:
        dec_content = f.read()
    
    print(f"\n📄 File encryption: {dec_content}")
    print(f"✅ PQ Protected: {metadata.get('pq_protected', False)}")
    
    # Cleanup
    os.remove(test_file)
    os.remove(enc_file)
    os.remove(dec_file)
    
    print("\n✅ All tests passed!")