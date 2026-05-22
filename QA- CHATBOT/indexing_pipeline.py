# GOAL OF THIS FILE

# This file should ONLY:

#| Responsibility              | Why                              |
#| --------------------------- | -------------------------------- |
#| Load multiple PDFs          | scalable ingestion pipeline      |
#| Clean document text         | improve chunk quality            |
#| Generate semantic chunks    | stronger retrieval               |
#| Create embeddings           | vector retrieval                 |
#| Build FAISS database        | scalable semantic search         |
#| Save vector database        | persistent RAG system            |
#| Create document registry    | hierarchical retrieval           |


import os

import pickle

from document_processor import (

    load_pdf,

    clean_text,

    create_metadata,

    split_into_sections
)

from chunking import create_chunks

from embeddings import embedding_model

from vector_store import create_vector_store


# =========================
# DOCUMENTS FOLDER
# =========================

DOCUMENTS_FOLDER = "documents"


# =========================
# ALL CHUNKS
# =========================

all_chunks = []

global_chunk_id = 0


# =========================
# DOCUMENT REGISTRY
# =========================

document_registry = {}


# =========================
# LOAD ALL PDF FILES
# =========================

pdf_files = [

    file

    for file in os.listdir(DOCUMENTS_FOLDER)

    if file.endswith(".pdf")
]


print("\nTOTAL PDF FILES:")

print(len(pdf_files))


# =========================
# PROCESS EACH PDF
# =========================

for pdf_file in pdf_files:

    print(f"\nPROCESSING: {pdf_file}")


    pdf_path = os.path.join(

        DOCUMENTS_FOLDER,

        pdf_file
    )


    docs = load_pdf(pdf_path)


    # =========================
    # DOCUMENT SUMMARY
    # =========================

    document_summary = ""


    # =========================
    # PROCESS PAGES
    # =========================

    for page_num, doc in enumerate(docs):

        raw_text = doc.page_content

        cleaned_text = clean_text(raw_text)


        sections = split_into_sections(cleaned_text)


        # =========================
        # PROCESS SECTIONS
        # =========================

        for section_data in sections:

            section_name = section_data["section"]

            section_content = section_data["content"]


            metadata = create_metadata(

                pdf_file,

                page_num + 1,

                section_name
            )


            chunks = create_chunks(

                section_content,

                metadata
            )


            # =========================
            # GLOBAL CHUNK IDS
            # =========================

            for chunk in chunks:

                chunk["chunk_id"] = global_chunk_id

                global_chunk_id += 1


            all_chunks.extend(chunks)


            # =========================
            # BUILD DOCUMENT SUMMARY
            # =========================

            if len(document_summary) < 3000:

                document_summary += " " + section_content


    # =========================
    # SAVE DOCUMENT SUMMARY
    # =========================

    document_registry[pdf_file] = document_summary[:3000]


# =========================
# TOTAL CHUNKS
# =========================

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


# =========================
# SAVE DOCUMENT REGISTRY
# =========================

with open("document_registry.pkl", "wb") as f:

    pickle.dump(document_registry, f)


print("\nDOCUMENT REGISTRY CREATED")


print("\nMULTI-DOCUMENT INDEXING COMPLETED")