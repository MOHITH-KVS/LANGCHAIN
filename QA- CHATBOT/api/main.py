# =============================================================
# api/main.py  —  PHASE 1 SECURED VERSION
# =============================================================
#
# WHAT CHANGED FROM YOUR ORIGINAL:
#
#  1. CORS locked to specific origins (was wide open)
#  2. Rate limiting on /chat  (10 requests/minute per IP)
#  3. /upload now validates file type (PDF only) + size (10 MB max)
#  4. /chat now returns HTTP 500 on error (was returning 200 with error text)
#  5. Input length validation on question field (max 500 chars)
#  6. Proper HTTP exceptions used throughout
#
# NEW PACKAGES NEEDED:
#   pip install slowapi
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


# =============================================================
# RATE LIMITER SETUP
# =============================================================
# slowapi reads the client's IP address and counts their requests.
# If they exceed the limit, it automatically returns HTTP 429.

limiter = Limiter(key_func=get_remote_address)


# =============================================================
# FASTAPI APP
# =============================================================

app = FastAPI(
    title="Industrial RAG Chatbot API",
    version="1.0.0"
)

# Tell FastAPI to use our rate limiter and its error handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# =============================================================
# CORS MIDDLEWARE
# =============================================================
# CORS = "who is allowed to call this API from a browser"
#
# BEFORE (your original): No CORS config = any website could call your API
# AFTER: Only the origins you list below are allowed
#
# HOW TO UPDATE:
#   - While testing locally:  keep "http://localhost:3000" and "http://127.0.0.1:5500"
#   - After deploying frontend to Vercel: add your Vercel URL, e.g. "https://my-chatbot.vercel.app"
#   - Remove localhost entries in production

ALLOWED_ORIGINS = [
    "http://localhost:3000",       # local React dev server (if you ever use it)
    "http://127.0.0.1:5500",       # VS Code Live Server (for testing your HTML frontend)
    "http://localhost:5500",
    # "https://my-chatbot.vercel.app",   # <-- uncomment and fill this in after deployment
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],   # only what you actually use
    allow_headers=["Content-Type"],
)


# =============================================================
# CONSTANTS FOR UPLOAD VALIDATION
# =============================================================

ALLOWED_FILE_TYPES = {"application/pdf"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024   # 10 MB
MAX_QUESTION_LENGTH = 500                 # characters


# =============================================================
# REQUEST / RESPONSE MODELS
# =============================================================

class ChatRequest(BaseModel):
    question: str
    document_name: str | None = None

    # This validator runs automatically when a request comes in.
    # If the question is too long, FastAPI returns HTTP 422 automatically.
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
# ROOT ENDPOINT
# =============================================================

@app.get("/")
def root():
    return {"message": "Industrial RAG Chatbot API Running"}


# =============================================================
# CHAT ENDPOINT
# =============================================================
# @limiter.limit("10/minute") means: max 10 calls per minute per IP address.
# If exceeded, slowapi automatically returns HTTP 429 Too Many Requests.

@app.post("/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
def chat(chat_request: ChatRequest, request: Request):
    # NOTE: FastAPI requires the Request object when using slowapi.
    # We pass 'http_request: Request' even though we don't use it directly.
    # slowapi reads it behind the scenes to get the IP address.

    try:
        response = ask_question(
            question=chat_request.question,
            document_name=chat_request.document_name
        )

        answer = response.get("answer", "")

        # Clean trailing commas (your existing logic)
        while answer.endswith(","):
            answer = answer[:-1].strip()

        response["answer"] = answer
        return response

    except Exception as e:
        # BEFORE: returned 200 OK with "Error: ..." in the answer field
        # AFTER:  returns proper HTTP 500 so the frontend knows something went wrong
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing your question: {str(e)}"
        )


# =============================================================
# UPLOAD ENDPOINT
# =============================================================

@app.post("/upload", response_model=UploadResponse)
@limiter.limit("5/minute")   # uploads are heavier, lower limit
async def upload_pdf(request: Request, file: UploadFile = File(...)):

    # --- VALIDATION 1: File type ---
    # file.content_type is set by the browser when it sends the file.
    # We only allow PDF.
    if file.content_type not in ALLOWED_FILE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Only PDF files are allowed."
        )

    # --- VALIDATION 2: File name must end in .pdf ---
    # Extra safety: even if content_type is spoofed, check the extension too.
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="File must have a .pdf extension."
        )

    # --- Read file content ---
    content = await file.read()

    # --- VALIDATION 3: File size ---
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is 10 MB."
        )

    # --- Save and index ---
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
        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {str(e)}"
        )


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