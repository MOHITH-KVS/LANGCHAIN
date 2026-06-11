# =============================================================
# services/conversation_memory.py  —  NEW FILE (PHASE 2)
# =============================================================
#
# WHY THIS FILE EXISTS:
#
#   Your chatbot_engine.py has this:
#
#       chat_history = []    ← plain Python list
#
#   Problem: This list lives in RAM.
#   Every time uvicorn restarts → chat_history = [] again.
#   Also, if two users talk at the same time → they SHARE the same list!
#   User A's history leaks into User B's conversation. Big problem.
#
#   This file fixes both problems using Redis:
#   - Each user/session gets their OWN history stored by session_id
#   - History survives restarts (stored in Redis cloud)
#   - Auto-expires after 30 minutes of inactivity
#
# HOW SESSION IDs WORK:
#   A session_id is just a unique string per browser tab/user.
#   Your frontend will generate one (we'll add that in Phase 3).
#   For now, we use "default" as a fallback session.
#
# =============================================================

import os
import json

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
except Exception as e:
    redis_client = None
    print(f"⚠️  Redis not available for conversation memory: {e}")


# =============================================================
# SETTINGS
# =============================================================

MEMORY_TTL_SECONDS = 1800   # conversation expires after 30 min idle
MAX_HISTORY_TURNS = 5       # keep last 5 Q&A pairs per session
MEMORY_PREFIX = "rag:memory:"


# =============================================================
# HELPER: MAKE A MEMORY KEY
# =============================================================

def make_memory_key(session_id: str) -> str:
    return f"{MEMORY_PREFIX}{session_id}"


# =============================================================
# GET CONVERSATION HISTORY
# =============================================================
# Returns the last N turns for this session as a formatted string.
# This string is passed to the LLM as conversation context.
#
# Example output:
#   "User: What is Mohith's CGPA?
#    Assistant: Mohith's CGPA is 8.84.
#    User: Which college does he attend?
#    Assistant: Gayatri Vidya Parishad College..."

def get_conversation_history(session_id: str = "default") -> str:
    if redis_client is None:
        return ""   # no Redis, return empty context

    try:
        key = make_memory_key(session_id)
        data = redis_client.get(key)

        if not data:
            return ""

        history = json.loads(data)

        # Format last N turns as conversation context for the LLM
        context = ""
        for turn in history[-MAX_HISTORY_TURNS:]:
            context += f"\nUser: {turn['question']}\n"
            context += f"Assistant: {turn['answer']}\n"

        return context

    except Exception as e:
        print(f"⚠️  Memory read error: {e}")
        return ""


# =============================================================
# SAVE TURN TO HISTORY
# =============================================================
# Called after every successful answer.
# Appends this Q&A turn to the session's history in Redis.

def save_turn_to_history(
    session_id: str,
    question: str,
    answer: str
):
    if redis_client is None:
        return

    try:
        key = make_memory_key(session_id)

        # Load existing history
        data = redis_client.get(key)
        history = json.loads(data) if data else []

        # Add new turn
        history.append({
            "question": question,
            "answer": answer
        })

        # Keep only last MAX_HISTORY_TURNS turns
        if len(history) > MAX_HISTORY_TURNS:
            history = history[-MAX_HISTORY_TURNS:]

        # Save back with refreshed TTL
        redis_client.setex(
            key,
            MEMORY_TTL_SECONDS,
            json.dumps(history)
        )

    except Exception as e:
        print(f"⚠️  Memory write error: {e}")


# =============================================================
# CLEAR SESSION HISTORY
# =============================================================
# Optional: call this to reset a conversation (like a "New Chat" button).

def clear_conversation_history(session_id: str = "default"):
    if redis_client is None:
        return

    try:
        key = make_memory_key(session_id)
        redis_client.delete(key)
        print(f"✅ Conversation history cleared for session: {session_id}")
    except Exception as e:
        print(f"⚠️  Memory clear error: {e}")