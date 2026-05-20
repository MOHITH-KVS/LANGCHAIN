#GOAL OF THIS FILE

#This file should ONLY:

#| Responsibility              | Why                              |
#| --------------------------- | -------------------------------- |
#| Connect all modules         | end-to-end RAG pipeline          |
#| Handle user queries         | chatbot interaction              |
#| Execute retrieval flow      | document question answering      |
#| Generate grounded answers   | accurate PDF responses           |
#| Create conversational loop  | interactive chatbot              |

from document_processor import (
    load_pdf,
    clean_text,
    detect_section,
    create_metadata,
    is_useful_page
)

from chunking import create_chunks

from embeddings import embedding_model

from vector_store import (
    create_vector_store,
    retrieve_chunks
)

from reranker import rerank_chunks

from generation import generate_answer

docs = load_pdf(
    "GVP-MAAA DOCUMENTATION (1).pdf"
)

all_chunks = []

for page_num, doc in enumerate(docs):

    raw_text = doc.page_content

    if not is_useful_page(raw_text):
        continue

    cleaned_text = clean_text(raw_text)

    section = detect_section(cleaned_text)

    metadata = create_metadata(
        "GVP-MAAA DOCUMENTATION (1).pdf",
        page_num + 1,
        section
    )

    chunks = create_chunks(
        cleaned_text,
        metadata
    )

    all_chunks.extend(chunks)

vector_store = create_vector_store(
    all_chunks,
    embedding_model
)
while True:

    question = input("\nAsk Question: ")

    if question.lower() == "exit":
        break

    retrieved_chunks = retrieve_chunks(
        vector_store,
        question
    )
    reranked_chunks = rerank_chunks(
        question,
        retrieved_chunks
    )
    answer = generate_answer(
        question,
        reranked_chunks
    )

    print("\nANSWER:\n")

    print(answer)