# GOAL OF THIS FILE

# This file should ONLY:

#| Responsibility            | Why                           |
#| ------------------------- | ----------------------------- |
#| Process PDFs              | indexing pipeline             |
#| Create embeddings         | semantic vector generation    |
#| Build FAISS index         | retrieval preparation         |
#| Save vector database      | persistent storage            |
#| Prepare chatbot knowledge | scalable RAG architecture     |

#GOAL OF THIS FILE

#This file should ONLY:

#| Responsibility            | Why                           |
#| ------------------------- | ----------------------------- |
#| Process PDFs              | indexing pipeline             |
#| Create embeddings         | semantic vector generation    |
#| Build FAISS index         | retrieval preparation         |
#| Save vector database      | persistent storage            |
#| Prepare chatbot knowledge | scalable RAG architecture     |


from document_processor import (
    load_pdf,
    clean_text,
    create_metadata,
    split_into_sections
)

from chunking import create_chunks

from embeddings import embedding_model

from vector_store import create_vector_store

import pickle


# =========================
# LOAD PDF
# =========================

docs = load_pdf(
    "GVP-MAAA DOCUMENTATION (1).pdf"
)


# =========================
# CREATE CHUNKS
# =========================

all_chunks = []

for page_num, doc in enumerate(docs):

    raw_text = doc.page_content

    cleaned_text = clean_text(raw_text)

    sections = split_into_sections(cleaned_text)

    for section_data in sections:

        section_name = section_data["section"]

        section_content = section_data["content"]

        metadata = create_metadata(

            "GVP-MAAA DOCUMENTATION (1).pdf",

            page_num + 1,

            section_name
        )

        chunks = create_chunks(

            section_content,

            metadata
        )

        all_chunks.extend(chunks)


print("\nTOTAL CHUNKS:")
print(len(all_chunks))


# =========================
# CREATE VECTOR STORE
# =========================

vector_store = create_vector_store(
    all_chunks,
    embedding_model
)


# =========================
# SAVE FAISS DATABASE
# =========================

vector_store.save_local("faiss_index")


# =========================
# SAVE CHUNKS
# =========================

with open("chunks.pkl", "wb") as f:

    pickle.dump(all_chunks, f)


print("\nVECTOR DATABASE SAVED SUCCESSFULLY")