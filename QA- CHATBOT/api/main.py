# =============================================================
# api/main.py  —  PHASE 2 UPDATED VERSION
# =============================================================
#
# WHAT CHANGED FROM PHASE 1:
#
#   1. ChatRequest now accepts optional session_id field
#      Frontend will send a unique session_id per browser tab.
#      If not sent, defaults to "default" (backward compatible).
#
#   2. /chat endpoint passes session_id to ask_question()
#      so Redis memory is stored per-user, not shared globally.
#
#   3. New /clear-history endpoint so user can start a fresh chat.
#
#   Everything else from Phase 1 is UNCHANGED.
#
# =============================================================

from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from engine.chatbot_engine import ask_question
from services.indexing_service import index_single_document
from services.feedback_manager import save_feedback
from services.conversation_memory import clear_conversation_history   # ✅ NEW


# =============================================================
# RATE LIMITER
# =============================================================

limiter = Limiter(key_func=get_remote_address)


# =============================================================
# FASTAPI APP
# =============================================================

app = FastAPI(
    title="Industrial RAG Chatbot API",
    version="2.0.0"
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# =============================================================
# CORS
# =============================================================

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "https://docsense-ai-alpha.vercel.app", 
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
# REQUEST / RESPONSE MODELS
# =============================================================

class ChatRequest(BaseModel):
    question: str
    document_name: str | None = None
    session_id: str = "default"    # ✅ NEW — unique ID per user/tab

    @field_validator("question")
    @classmethod
    def question_must_not_be_too_long(cls, v):
        v = v.strip()
        if len(v) == 0:
            raise ValueError("Question cannot be empty.")
        if len(v) > MAX_QUESTION_LENGTH:
            raise ValueError(
                f"Question too long. Maximum {MAX_QUESTION_LENGTH} characters allowed."
            )
        return v


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


class UploadResponse(BaseModel):
    message: str
    filename: str


class FeedbackRequest(BaseModel):
    question: str
    feedback: str


# =============================================================
# ROOT
# =============================================================

@app.get("/")
def root():
    return {"message": "Industrial RAG Chatbot API v2.0 Running"}


# =============================================================
# CHAT ENDPOINT
# =============================================================

@app.post("/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
def chat(chat_request: ChatRequest, request: Request):
    try:
        response = ask_question(
            question=chat_request.question,
            document_name=chat_request.document_name,
            session_id=chat_request.session_id    # ✅ NEW — pass session_id
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
# UPLOAD ENDPOINT
# =============================================================

@app.post("/upload", response_model=UploadResponse)
@limiter.limit("5/minute")
async def upload_pdf(request: Request, file: UploadFile = File(...)):

    if file.content_type not in ALLOWED_FILE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Only PDF files are allowed."
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="File must have a .pdf extension."
        )

    content = await file.read()

    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail="File too large. Maximum size is 10 MB."
        )

    try:
        file_path = f"documents/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(content)

        index_single_document(file_path)

        return {
            "message": "PDF uploaded and indexed successfully",
            "filename": file.filename
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


# =============================================================
# FEEDBACK ENDPOINT
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
# CLEAR HISTORY ENDPOINT  (✅ NEW)
# =============================================================
# Frontend can call this when user clicks "New Chat" button.
# Clears the Redis conversation history for that session.

@app.post("/clear-history")
@limiter.limit("10/minute")
def clear_history(request: Request, session_id: str = "default"):
    try:
        clear_conversation_history(session_id)
        return {"status": "success", "message": "Conversation history cleared."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
