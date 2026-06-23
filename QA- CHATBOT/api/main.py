# =============================================================
# api/main.py  —  V2 MULTI-TENANT VERSION
# =============================================================
#
# WHAT CHANGED FROM V1:
#
#   1. /upload now accepts session_id in the form data
#      → passes it to index_single_document()
#      → each user's PDF goes into their own private Redis index
#
#   2. /chat already had session_id — no change needed there
#      (chatbot_engine.py handles loading the right session's index)
#
#   3. /clear-history now also clears the session's document index
#      → when user starts a new chat, their old documents are cleared too
#
# EVERYTHING ELSE from Phase 1 and Phase 2 is UNCHANGED:
#   - CORS, rate limiting, file validation, feedback — all same
#
# =============================================================

from fastapi import FastAPI, UploadFile, File, Request, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from engine.chatbot_engine import ask_question
from services.indexing_service import index_single_document
from services.feedback_manager import save_feedback
from services.conversation_memory import clear_conversation_history
from services.session_store import clear_session      # ✅ NEW


# =============================================================
# RATE LIMITER
# =============================================================

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="DocSense AI Backend", version="2.0.0")

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# =============================================================
# CORS
# =============================================================

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "https://5star-insight.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# =============================================================
# CONSTANTS
# =============================================================

ALLOWED_FILE_TYPES = {"application/pdf"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
MAX_QUESTION_LENGTH = 500


# =============================================================
# REQUEST MODELS
# =============================================================

class ChatRequest(BaseModel):
    question: str
    document_name: str | None = None
    session_id: str = "default"

    @field_validator("question")
    @classmethod
    def question_must_not_be_too_long(cls, v):
        v = v.strip()
        if len(v) == 0:
            raise ValueError("Question cannot be empty.")
        if len(v) > MAX_QUESTION_LENGTH:
            raise ValueError(f"Question too long. Maximum {MAX_QUESTION_LENGTH} characters allowed.")
        return v


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


class FeedbackRequest(BaseModel):
    question: str
    feedback: str


# =============================================================
# ROOT
# =============================================================

@app.get("/")
def root():
    return {"message": "DocSense AI Backend v2.0 — Multi-tenant RAG"}


# =============================================================
# CHAT ENDPOINT  (unchanged from Phase 2)
# =============================================================

@app.post("/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
def chat(chat_request: ChatRequest, request: Request):
    try:
        response = ask_question(
            question=chat_request.question,
            document_name=chat_request.document_name,
            session_id=chat_request.session_id
        )
        answer = response.get("answer", "")
        while answer.endswith(","):
            answer = answer[:-1].strip()
        response["answer"] = answer
        return response
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing your question: {str(e)}"
        )


# =============================================================
# UPLOAD ENDPOINT  (✅ CHANGED — now accepts session_id)
# =============================================================

@app.post("/upload")
@limiter.limit("5/minute")
async def upload_pdf(
    request: Request,
    file: UploadFile = File(...),
    session_id: str = Form(default="default")    # ✅ NEW: which user's session
):
    """
    Upload and index a PDF for a specific user session.

    session_id comes from the frontend (generated per browser tab).
    Each session gets its own private index in Redis.
    Documents from different sessions NEVER mix.
    """

    # ── File type validation ──────────────────────────────────
    if file.content_type not in ALLOWED_FILE_TYPES:
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must have a .pdf extension.")

    # ── Read content ──────────────────────────────────────────
    content = await file.read()

    # ── File size validation ──────────────────────────────────
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 10 MB.")

    try:
        # ── Save temporarily to disk for processing ───────────
        # We still need to write to disk temporarily for PyMuPDF
        # to read the PDF content. We delete it after indexing.
        import os
        os.makedirs("documents", exist_ok=True)
        file_path = f"documents/{session_id[:8]}_{file.filename}"

        with open(file_path, "wb") as f:
            f.write(content)

        # ── Index into THIS user's private session ────────────
        # The session_id tells indexing_service which Redis key to use
        index_single_document(
            pdf_path=file_path,
            session_id=session_id    # ✅ KEY CHANGE — per-session indexing
        )

        # ── Delete the temp file after indexing ───────────────
        # We don't need it on disk anymore — it's in Redis
        try:
            os.remove(file_path)
        except Exception:
            pass

        return {
            "message": "PDF uploaded and indexed successfully",
            "filename": file.filename,
            "session_id": session_id[:12] + "..."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


# =============================================================
# FEEDBACK ENDPOINT  (unchanged)
# =============================================================

@app.post("/feedback")
@limiter.limit("20/minute")
def submit_feedback(request: Request, feedback_request: FeedbackRequest):
    try:
        save_feedback(feedback_request.question, feedback_request.feedback)
        return {"status": "success", "message": "Feedback saved successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================
# CLEAR HISTORY ENDPOINT  (✅ UPDATED — also clears document index)
# =============================================================

@app.post("/clear-history")
@limiter.limit("10/minute")
def clear_history(request: Request, session_id: str = "default"):
    """
    Clear both conversation history AND document index for this session.
    Called when user clicks "New Chat" — starts completely fresh.
    """
    try:
        # Clear Redis conversation memory
        clear_conversation_history(session_id)

        # ✅ NEW: Also clear the session's document index
        # This means a new chat = fresh document upload required
        # Which is the correct behavior — no stale old documents
        clear_session(session_id)

        return {"status": "success", "message": "Chat history and documents cleared."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))