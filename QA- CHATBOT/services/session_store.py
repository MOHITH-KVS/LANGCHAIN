# =============================================================
# services/session_store.py  —  NEW FILE (Multi-tenant V2)
# =============================================================
#
# PURPOSE:
#   This file is the "bank locker" described in the explanation.
#   Instead of saving FAISS indexes and chunks to Render's disk
#   (which gets wiped on every redeploy), we store everything in
#   Upstash Redis — which is permanent and never gets wiped.
#
#   Each user session gets its own private "locker" in Redis,
#   keyed by their session_id.
#
# WHAT GETS STORED PER SESSION:
#   1. FAISS index (serialized as bytes → base64 string)
#   2. All chunks (pickled → base64 string)
#   3. Document registry (pickled → base64 string)
#
# TTL: 2 hours — sessions auto-expire after 2 hours of inactivity.
#   This prevents Redis from filling up over time.
#
# REDIS KEY STRUCTURE:
#   session:{session_id}:faiss      ← the FAISS index
#   session:{session_id}:chunks     ← all text chunks
#   session:{session_id}:registry   ← document profiles
#
# =============================================================

import os
import io
import base64
import pickle
import tempfile

from upstash_redis import Redis
from dotenv import load_dotenv

load_dotenv()

# =============================================================
# CONNECT TO REDIS
# =============================================================

try:
    redis_client = Redis(
        url=os.getenv("UPSTASH_REDIS_REST_URL"),
        token=os.getenv("UPSTASH_REDIS_REST_TOKEN")
    )
    print("✅ Session store Redis connected")
except Exception as e:
    redis_client = None
    print(f"⚠️  Session store Redis failed: {e}")


# =============================================================
# SETTINGS
# =============================================================

SESSION_TTL = 7200      # 2 hours in seconds
KEY_PREFIX = "session"


# =============================================================
# HELPERS
# =============================================================

def _make_key(session_id: str, data_type: str) -> str:
    """
    Build a Redis key like: session:abc123:faiss
    """
    return f"{KEY_PREFIX}:{session_id}:{data_type}"


def _to_base64(data: bytes) -> str:
    """Convert bytes to base64 string for Redis storage."""
    return base64.b64encode(data).decode("utf-8")


def _from_base64(data: str) -> bytes:
    """Convert base64 string back to bytes."""
    return base64.b64decode(data.encode("utf-8"))


# =============================================================
# SAVE SESSION DATA
# =============================================================

def save_session_data(session_id: str, vector_store, all_chunks: list, document_registry: dict):
    """
    Save a user's complete session to Redis.

    Called after every /upload so the session data is always
    up to date in Redis.

    Parameters:
        session_id    : unique ID per browser tab (from frontend)
        vector_store  : FAISS vector store object
        all_chunks    : list of all text chunk dicts
        document_registry : dict of document profiles/summaries
    """
    if redis_client is None:
        print("⚠️  Redis not available — cannot save session data")
        return False

    try:
        # ── 1. Serialize FAISS index ──────────────────────────
        # FAISS can only save to disk, not directly to bytes.
        # So we create a temporary folder, save to it, read the
        # files as bytes, then delete the temp folder.
        with tempfile.TemporaryDirectory() as tmp_dir:
            vector_store.save_local(tmp_dir)

            # FAISS creates two files: index.faiss and index.pkl
            faiss_data = {}
            for filename in os.listdir(tmp_dir):
                filepath = os.path.join(tmp_dir, filename)
                with open(filepath, "rb") as f:
                    faiss_data[filename] = _to_base64(f.read())

        # Store the FAISS file dict as pickled base64
        faiss_serialized = _to_base64(pickle.dumps(faiss_data))

        # ── 2. Serialize chunks ────────────────────────────────
        chunks_serialized = _to_base64(pickle.dumps(all_chunks))

        # ── 3. Serialize document registry ───────────────────
        registry_serialized = _to_base64(pickle.dumps(document_registry))

        # ── 4. Store all three in Redis with TTL ──────────────
        redis_client.setex(
            _make_key(session_id, "faiss"),
            SESSION_TTL,
            faiss_serialized
        )
        redis_client.setex(
            _make_key(session_id, "chunks"),
            SESSION_TTL,
            chunks_serialized
        )
        redis_client.setex(
            _make_key(session_id, "registry"),
            SESSION_TTL,
            registry_serialized
        )

        print(f"✅ Session saved to Redis: {session_id[:12]}...")
        return True

    except Exception as e:
        print(f"⚠️  Failed to save session to Redis: {e}")
        return False


# =============================================================
# LOAD SESSION DATA
# =============================================================

def load_session_data(session_id: str, embedding_model):
    """
    Load a user's FAISS index, chunks, and registry from Redis.

    Returns:
        (vector_store, all_chunks, document_registry)
        or (None, None, None) if session not found or expired.
    """
    if redis_client is None:
        return None, None, None

    try:
        # ── 1. Load FAISS index ───────────────────────────────
        faiss_raw = redis_client.get(_make_key(session_id, "faiss"))
        if not faiss_raw:
            print(f"⚠️  No session found in Redis for: {session_id[:12]}...")
            return None, None, None

        faiss_data = pickle.loads(_from_base64(faiss_raw))

        # Write FAISS files to a temp directory and load from there
        with tempfile.TemporaryDirectory() as tmp_dir:
            for filename, file_b64 in faiss_data.items():
                filepath = os.path.join(tmp_dir, filename)
                with open(filepath, "wb") as f:
                    f.write(_from_base64(file_b64))

            from langchain_community.vectorstores import FAISS
            vector_store = FAISS.load_local(
                tmp_dir,
                embedding_model,
                allow_dangerous_deserialization=True
            )

        # ── 2. Load chunks ─────────────────────────────────────
        chunks_raw = redis_client.get(_make_key(session_id, "chunks"))
        all_chunks = pickle.loads(_from_base64(chunks_raw)) if chunks_raw else []

        # ── 3. Load document registry ─────────────────────────
        registry_raw = redis_client.get(_make_key(session_id, "registry"))
        document_registry = pickle.loads(_from_base64(registry_raw)) if registry_raw else {}

        # ── 4. Refresh TTL (user is still active) ─────────────
        redis_client.expire(_make_key(session_id, "faiss"), SESSION_TTL)
        redis_client.expire(_make_key(session_id, "chunks"), SESSION_TTL)
        redis_client.expire(_make_key(session_id, "registry"), SESSION_TTL)

        print(f"✅ Session loaded from Redis: {session_id[:12]}...")
        return vector_store, all_chunks, document_registry

    except Exception as e:
        print(f"⚠️  Failed to load session from Redis: {e}")
        return None, None, None


# =============================================================
# CHECK IF SESSION EXISTS
# =============================================================

def session_exists(session_id: str) -> bool:
    """Check if a session exists in Redis without loading it."""
    if redis_client is None:
        return False
    try:
        return redis_client.exists(_make_key(session_id, "chunks")) > 0
    except Exception:
        return False


# =============================================================
# CLEAR SESSION
# =============================================================

def clear_session(session_id: str):
    """Delete all Redis keys for this session."""
    if redis_client is None:
        return
    try:
        redis_client.delete(
            _make_key(session_id, "faiss"),
            _make_key(session_id, "chunks"),
            _make_key(session_id, "registry")
        )
        print(f"✅ Session cleared: {session_id[:12]}...")
    except Exception as e:
        print(f"⚠️  Failed to clear session: {e}")