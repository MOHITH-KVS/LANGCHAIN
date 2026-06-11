# =============================================================
# services/cache_manager.py  —  PHASE 2 REDIS VERSION
# =============================================================
#
# WHAT CHANGED FROM YOUR ORIGINAL:
#
#   BEFORE: response_cache = {}
#           A plain Python dictionary.
#           Every time you restart uvicorn → cache is WIPED.
#           Cache only lives in RAM of one process.
#
#   AFTER:  Cache stored in Upstash Redis (cloud).
#           Survives restarts. Survives deployments.
#           TTL = 1 hour (answers expire automatically).
#           If Redis is unavailable, falls back gracefully (no crash).
#
# HOW IT WORKS:
#   - Same two functions: get_cached_response() and save_to_cache()
#   - Your chatbot_engine.py calls them exactly the same way as before
#   - Zero changes needed in chatbot_engine.py
#
# =============================================================

import os
import json
import hashlib

from upstash_redis import Redis
from dotenv import load_dotenv

load_dotenv()


# =============================================================
# CONNECT TO UPSTASH REDIS
# =============================================================
# Redis() automatically reads UPSTASH_REDIS_REST_URL and
# UPSTASH_REDIS_REST_TOKEN from your .env file.

try:
    redis_client = Redis(
        url=os.getenv("UPSTASH_REDIS_REST_URL"),
        token=os.getenv("UPSTASH_REDIS_REST_TOKEN")
    )
    print("✅ Redis connected successfully")
except Exception as e:
    redis_client = None
    print(f"⚠️  Redis connection failed: {e}. Running without cache.")


# =============================================================
# CACHE SETTINGS
# =============================================================

CACHE_TTL_SECONDS = 3600   # cache expires after 1 hour
CACHE_PREFIX = "rag:cache:"  # all our keys start with this


# =============================================================
# HELPER: MAKE A CACHE KEY
# =============================================================
# We hash the question so special characters don't cause issues.
# "What is Mohith's CGPA?" → "rag:cache:a3f9b2c1..."

def make_cache_key(question: str) -> str:
    cleaned = question.lower().strip()
    hashed = hashlib.md5(cleaned.encode()).hexdigest()
    return f"{CACHE_PREFIX}{hashed}"


# =============================================================
# GET CACHED RESPONSE
# =============================================================
# Called by chatbot_engine.py before running the full RAG pipeline.
# If we find a cached answer → return it immediately (no LLM call).
# If not found or Redis is down → return None (run full pipeline).

def get_cached_response(question: str):
    if redis_client is None:
        return None   # Redis not available, skip cache

    try:
        key = make_cache_key(question)
        cached = redis_client.get(key)

        if cached:
            print("\n✅ CACHE HIT — returning cached response")
            # Redis returns a string, we stored JSON, so parse it back
            return json.loads(cached)

        print("\n❌ CACHE MISS — running full RAG pipeline")
        return None

    except Exception as e:
        print(f"⚠️  Cache read error: {e}")
        return None   # if cache fails, just run the pipeline normally


# =============================================================
# SAVE TO CACHE
# =============================================================
# Called by chatbot_engine.py after generating a fresh answer.
# Saves the full response to Redis with a 1-hour expiry.

def save_to_cache(question: str, response: dict):
    if redis_client is None:
        return   # Redis not available, skip silently

    try:
        key = make_cache_key(question)
        # Convert response dict to JSON string for storage
        redis_client.setex(
            key,
            CACHE_TTL_SECONDS,
            json.dumps(response)
        )
        print(f"✅ Response cached for 1 hour")

    except Exception as e:
        print(f"⚠️  Cache write error: {e}")
        # Don't crash if cache write fails — just continue