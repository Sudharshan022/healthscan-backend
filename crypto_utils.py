"""
crypto_utils.py – AES-256-GCM Image Encryption
================================================
Provides end-to-end encryption for uploaded image bytes.

Key design decisions:
  • AES-256-GCM: authenticated encryption (confidentiality + integrity).
  • A fresh 256-bit key and 96-bit nonce are generated per image.
  • The nonce + auth-tag are prepended to the ciphertext so the file
    is self-contained (no separate metadata file needed).
  • The plaintext key is NEVER written to disk. In production, replace
    _derive_key() with a call to a KMS (AWS KMS, GCP Cloud KMS, etc.)
    and store only the `key_id` in the database.

Wire format (bytes):
  [12B nonce][16B auth-tag][N bytes ciphertext]
"""

import os
import uuid
import base64
import logging
from typing import Tuple

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

log = logging.getLogger(__name__)

# ── In-memory key store (DEV ONLY) ───────────────────────────────────────────
# In production: integrate with AWS KMS / HashiCorp Vault.
# Map of key_id → raw 32-byte key.
_KEY_STORE: dict[str, bytes] = {}

NONCE_SIZE = 12   # 96 bits – GCM recommended
TAG_SIZE   = 16   # 128-bit authentication tag


def _generate_key() -> Tuple[str, bytes]:
    """Generate a fresh AES-256 key, register it, return (key_id, raw_key)."""
    key_id  = str(uuid.uuid4())
    raw_key = os.urandom(32)          # 256 bits
    _KEY_STORE[key_id] = raw_key
    return key_id, raw_key


def _load_key(key_id: str) -> bytes:
    """Retrieve a key by ID. Raises if not found (key rotation / expiry)."""
    key = _KEY_STORE.get(key_id)
    if key is None:
        raise KeyError(f"Encryption key '{key_id}' not found. "
                       "Was the key store restarted?")
    return key


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def encrypt_image_bytes(raw_bytes: bytes) -> Tuple[bytes, dict]:
    """
    Encrypt raw image bytes with AES-256-GCM.

    Args:
        raw_bytes: Plaintext image data.

    Returns:
        (encrypted_payload, encryption_meta)
        • encrypted_payload : bytes  – [nonce | ciphertext+tag]
        • encryption_meta   : dict   – {key_id, algorithm, nonce_b64}
    """
    key_id, raw_key = _generate_key()
    nonce           = os.urandom(NONCE_SIZE)

    aesgcm    = AESGCM(raw_key)
    # AESGCM.encrypt appends the 16-byte tag to the ciphertext automatically.
    ciphertext_with_tag = aesgcm.encrypt(nonce, raw_bytes, associated_data=None)

    # Wire format: nonce ‖ ciphertext+tag
    encrypted_payload = nonce + ciphertext_with_tag

    meta = {
        "key_id":    key_id,
        "algorithm": "AES-256-GCM",
        "nonce_b64": base64.b64encode(nonce).decode(),
    }

    log.debug(f"Encrypted {len(raw_bytes)} bytes → {len(encrypted_payload)} bytes | key={key_id}")
    return encrypted_payload, meta


def decrypt_image_bytes(encrypted_payload: bytes, encryption_meta: dict) -> bytes:
    """
    Decrypt an encrypted image payload in-memory only.

    This function is intentionally kept separate from any disk I/O.
    The decrypted bytes MUST NOT be written to disk anywhere in the
    calling code.

    Args:
        encrypted_payload : bytes – [nonce | ciphertext+tag]
        encryption_meta   : dict  – must contain 'key_id'

    Returns:
        plaintext image bytes
    """
    key_id  = encryption_meta["key_id"]
    raw_key = _load_key(key_id)

    nonce               = encrypted_payload[:NONCE_SIZE]
    ciphertext_with_tag = encrypted_payload[NONCE_SIZE:]

    aesgcm = AESGCM(raw_key)
    try:
        plaintext = aesgcm.decrypt(nonce, ciphertext_with_tag, associated_data=None)
    except Exception as exc:
        # Includes InvalidTag – tampered or corrupted data
        log.error(f"Decryption failed for key={key_id}: {exc}")
        raise ValueError("Image decryption failed: data may be corrupted or tampered.") from exc

    log.debug(f"Decrypted {len(encrypted_payload)} bytes → {len(plaintext)} bytes | key={key_id}")
    return plaintext


def rotate_key(old_key_id: str, encrypted_payload: bytes) -> Tuple[bytes, dict]:
    """
    Re-encrypt an existing payload under a new key (key rotation).
    Useful for periodic security compliance.

    Returns the newly encrypted payload and new meta.
    """
    old_meta   = {"key_id": old_key_id}
    plaintext  = decrypt_image_bytes(encrypted_payload, old_meta)
    new_payload, new_meta = encrypt_image_bytes(plaintext)

    # Zero out plaintext buffer immediately (best-effort in CPython)
    plaintext = b"\x00" * len(plaintext)  # noqa

    # Optionally remove old key from store
    _KEY_STORE.pop(old_key_id, None)
    log.info(f"Key rotation complete: {old_key_id} → {new_meta['key_id']}")
    return new_payload, new_meta
