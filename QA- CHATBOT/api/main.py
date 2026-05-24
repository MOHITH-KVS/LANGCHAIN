# GOAL OF THIS FILE

# This file should ONLY:

#| Responsibility              | Why                              |
#| --------------------------- | -------------------------------- |
#| Create API endpoints        | frontend/backend communication   |
#| Validate incoming requests  | production safety                |
#| Connect chatbot engine      | scalable architecture            |
#| Return structured JSON      | frontend integration             |


from fastapi import (

    FastAPI,

    UploadFile,

    File
)

from pydantic import BaseModel

from engine.chatbot_engine import ask_question

from services.indexing_service import (

    index_single_document
)


# =========================
# FASTAPI APP
# =========================

app = FastAPI(

    title="Industrial RAG Chatbot API",

    version="1.0.0"
)


# =========================
# REQUEST MODEL
# =========================

class ChatRequest(BaseModel):

    question: str

    document_name: str | None = None


# =========================
# RESPONSE MODEL
# =========================

class ChatResponse(BaseModel):

    answer: str

    sources: list[str]


# =========================
# UPLOAD RESPONSE MODEL
# =========================

class UploadResponse(BaseModel):

    message: str

    filename: str

# =========================
# ROOT ENDPOINT
# =========================

@app.get("/")

def root():

    return {

        "message": "Industrial RAG Chatbot API Running"
    }


# =========================
# CHAT ENDPOINT
# =========================

@app.post(

    "/chat",

    response_model=ChatResponse
)

def chat(request: ChatRequest):

    try:

        response = ask_question(

            question=request.question,

            document_name=request.document_name
        )

        return response

    except Exception as e:

        return {

            "answer": f"Error: {str(e)}",

            "sources": []
        }
    
# =========================
# UPLOAD ENDPOINT
# =========================

@app.post(

    "/upload",

    response_model=UploadResponse
)

async def upload_pdf(

    file: UploadFile = File(...)
):

    try:


        # =========================
        # SAVE FILE
        # =========================

        file_path = f"documents/{file.filename}"


        with open(

            file_path,

            "wb"
        ) as f:

            content = await file.read()

            f.write(content)


        # =========================
        # REINDEX DOCUMENTS
        # =========================

        index_single_document(

            file_path
        )


        return {

            "message": "PDF uploaded and indexed successfully",

            "filename": file.filename
        }


    except Exception as e:

        return {

            "message": f"Upload failed: {str(e)}",

            "filename": file.filename
        }