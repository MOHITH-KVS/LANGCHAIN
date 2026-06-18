# GOAL OF THIS FILE

# This file should ONLY:

#| Responsibility              | Why                              |
#| --------------------------- | -------------------------------- |
#| Process uploaded PDFs       | reusable ingestion service       |
#| Generate semantic chunks    | scalable retrieval               |
#| Create vector embeddings    | semantic search                  |
#| Build/update FAISS index    | industrial RAG architecture      |
#| Save vector database        | persistent retrieval             |
#| Maintain document registry  | hierarchical retrieval           |


import os

import re

import pickle

from document_processor import (

    load_pdf,

    clean_text,

    create_metadata,

    split_into_sections
)

from chunking import create_chunks

from embeddings import embedding_model

from vector_store import (

    create_vector_store,

    load_vector_store,

    add_documents_to_vector_store
)


import numpy as np


# =========================
# PATHS
# =========================

DOCUMENTS_FOLDER = "documents"

FAISS_INDEX_PATH = "faiss_index"

CHUNKS_PATH = "chunks.pkl"

DOCUMENT_REGISTRY_PATH = "document_registry.pkl"



def cosine_similarity_manual(a, b):
        a = np.array(a)
        b = np.array(b)
 
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
 
        return np.dot(a_norm, b_norm.T)


def build_document_summary(document_chunks):

    # =========================
    # EMPTY SAFETY
    # =========================

    if not document_chunks:

        return ""


    # =========================
    # GET CHUNK CONTENTS
    # =========================

    chunk_texts = [

        chunk["content"]

        for chunk in document_chunks
    ]


    # =========================
    # CREATE EMBEDDINGS
    # =========================

    chunk_embeddings = embedding_model.embed_documents(
        chunk_texts
    )


    # =========================
    # DOCUMENT CENTROID
    # =========================

    document_centroid = np.mean(

        chunk_embeddings,

        axis=0
    ).reshape(1, -1)


    # =========================
    # CHUNK SIMILARITY TO DOCUMENT
    # =========================

    similarities = cosine_similarity_manual(
 
        chunk_embeddings,
 
        document_centroid
    ).flatten()


    # =========================
    # RANK CHUNKS
    # =========================

    ranked_indices = np.argsort(

        similarities

    )[::-1]


    # =========================
    # SELECT REPRESENTATIVE CHUNKS
    # =========================

    selected_chunks = []


    for idx in ranked_indices:

        content = chunk_texts[idx]

        # Skip extremely tiny chunks

        if len(content.strip()) < 80:

            continue

        # Skip noisy chunks

        bad_patterns = [

            "page",
            "figure",
            "table",
            "references",
            "copyright",
            "journal",
            "www",
            "http"
        ]

        if any(pattern in content.lower() for pattern in bad_patterns):

            continue


        selected_chunks.append(content)


        # Industrial summary limit

        if len(selected_chunks) >= 6:

            break


    # =========================
    # FINAL DOCUMENT SUMMARY
    # =========================

    summary = "\n".join(selected_chunks)


    return summary[:3000]


# =========================
# EXTRACT DOCUMENT KEYWORDS
# =========================

def extract_document_keywords(document_chunks):

    # =========================
    # COLLECT TEXT
    # =========================

    combined_text = " ".join(

        chunk["content"]

        for chunk in document_chunks
    )


    combined_text = combined_text.lower()


    # =========================
    # TOKENIZATION
    # =========================

    words = combined_text.split()


    # =========================
    # CLEAN TOKENS
    # =========================

    cleaned_words = []


    for word in words:

        word = word.strip(

            ".,:;!?()[]{}<>\"'"
        )


        # Skip tiny tokens

        if len(word) < 4:

            continue


        # Skip numeric-heavy tokens

        if word.isdigit():

            continue


        cleaned_words.append(word)


    # =========================
    # WORD FREQUENCY
    # =========================

    frequency = {}


    for word in cleaned_words:

        frequency[word] = (

            frequency.get(word, 0) + 1
        )


    # =========================
    # SORT KEYWORDS
    # =========================

    sorted_keywords = sorted(

        frequency.items(),

        key=lambda x: x[1],

        reverse=True
    )


    # =========================
    # TOP KEYWORDS
    # =========================

    keywords = [

        word

        for word, _ in sorted_keywords[:25]
    ]


    return keywords


# =========================
# PROCESS SINGLE PDF
# =========================

def process_pdf_document(

    pdf_path,

    global_chunk_id
):

    pdf_file = os.path.basename(pdf_path)

    docs = load_pdf(pdf_path)

    document_chunks = []



    # =========================
    # PROCESS PAGES
    # =========================

    for page_num, doc in enumerate(docs):

        raw_text = doc.page_content

        cleaned_text = clean_text(raw_text)

        # =========================
        # EXTRACTION QUALITY CHECK
        # =========================

        word_count = len(cleaned_text.split())

        if word_count < 30:

            print(
                f"\nWARNING: Poor extraction detected"
                f"\nPDF: {pdf_file}"
                f"\nPage: {page_num + 1}"
                f"\nWords Extracted: {word_count}"
            )

        # Skip pages that are mostly code
        code_line_count = sum(
            1 for line in cleaned_text.split("\n")
            if re.search(r"(def |class |import |return |const |export |function |\btablename\b|Column\(|BaseModel)", line)
        )
        total_lines = max(len(cleaned_text.split("\n")), 1)
        code_ratio = code_line_count / total_lines

        if code_ratio > 0.3:
            print(f"\nSKIPPING CODE PAGE: {page_num+1} | code ratio: {round(code_ratio, 2)}")
            continue

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

                section_name,

                document_type = "pdf"
            )


            chunks = create_chunks(

                section_name.upper() + "\n\n" + section_content,

                metadata
            )


            # =========================
            # GLOBAL CHUNK IDS
            # =========================

            for chunk in chunks:

                chunk["chunk_id"] = global_chunk_id

                global_chunk_id += 1


            document_chunks.extend(chunks)


    # =========================
    # GENERATE FINAL DOCUMENT SUMMARY
    # =========================

    document_summary = build_document_summary(

            document_chunks
    )


    # =========================
    # EXTRACT DOCUMENT KEYWORDS
    # =========================

    document_keywords = extract_document_keywords(

        document_chunks
    )


    return (

        document_chunks,

        document_summary,

        document_keywords,

        global_chunk_id
    )


# =========================
# SAVE DATABASE FILES
# =========================

def save_database_files(

    vector_store,

    all_chunks,

    document_registry
):


    # =========================
    # SAVE FAISS INDEX
    # =========================

    vector_store.save_local(

        FAISS_INDEX_PATH
    )


    # =========================
    # SAVE CHUNKS
    # =========================

    with open(

        CHUNKS_PATH,

        "wb"
    ) as f:

        pickle.dump(

            all_chunks,

            f
        )


    # =========================
    # SAVE DOCUMENT REGISTRY
    # =========================

    with open(

        DOCUMENT_REGISTRY_PATH,

        "wb"
    ) as f:

        pickle.dump(

            document_registry,

            f
        )


# =========================
# INDEX SINGLE DOCUMENT
# =========================

def index_single_document(

    pdf_path
):


    # =========================
    # LOAD EXISTING CHUNKS
    # =========================

    if os.path.exists(CHUNKS_PATH):

        with open(CHUNKS_PATH, "rb") as f:

            all_chunks = pickle.load(f)

    else:

        all_chunks = []


    # =========================
    # LOAD DOCUMENT REGISTRY
    # =========================

    if os.path.exists(DOCUMENT_REGISTRY_PATH):

        with open(DOCUMENT_REGISTRY_PATH, "rb") as f:

            document_registry = pickle.load(f)

    else:

        document_registry = {}


    # =========================
    # LOAD OR CREATE VECTOR STORE
    # =========================

    if os.path.exists(FAISS_INDEX_PATH):

        vector_store = load_vector_store(

            embedding_model
        )

    else:

        vector_store = None


    # =========================
    # GLOBAL CHUNK ID
    # =========================

    if all_chunks:

        global_chunk_id = (

            max(

                chunk["chunk_id"]

                for chunk in all_chunks
            ) + 1
        )

    else:

        global_chunk_id = 0


    # =========================
    # PROCESS NEW PDF
    # =========================

    (

        document_chunks,

        document_summary,

        document_keywords,

        global_chunk_id

    ) = process_pdf_document(

        pdf_path,

        global_chunk_id
    )


    # =========================
    # APPEND CHUNKS
    # =========================

    all_chunks.extend(

        document_chunks
    )


    # =========================
    # UPDATE REGISTRY
    # =========================

    pdf_file = os.path.basename(

        pdf_path
    )


    document_registry[pdf_file] = {

        "summary": document_summary,

        "keywords": document_keywords,

        "sample_chunks": [

            chunk["content"][:500]

            for chunk in sorted(

                document_chunks,

                key=lambda x: len(x["content"]),

                reverse=True

            )[:15]
        ]
    }

    print("\n========================")
    print("DOCUMENT PROFILE DEBUG")
    print("========================")

    print("FILE:", pdf_file)

    print("\nSUMMARY:")
    print(document_summary[:500])

    print("\nKEYWORDS:")
    print(document_keywords[:20])

    print("\nSAMPLE CHUNKS:")
    for chunk in document_registry[pdf_file]["sample_chunks"][:3]:

        print("-", chunk[:200])


    # =========================
    # CREATE OR UPDATE VECTOR STORE
    # =========================

    if vector_store is None:

        vector_store = create_vector_store(

            document_chunks,

            embedding_model
        )

    else:

        vector_store = add_documents_to_vector_store(

            vector_store,

            document_chunks
        )


    # =========================
    # SAVE UPDATED DATABASE
    # =========================

    save_database_files(

        vector_store,

        all_chunks,

        document_registry
    )


    print(f"\nNEW DOCUMENT INDEXED: {pdf_file}")


# =========================
# INDEX DOCUMENTS
# =========================

def index_documents():


    # =========================
    # LOAD PDF FILES
    # =========================

    pdf_files = [

        file

        for file in os.listdir(DOCUMENTS_FOLDER)

        if file.endswith(".pdf")
    ]


    print("\nTOTAL PDF FILES:")

    print(len(pdf_files))


    # =========================
    # STORAGE
    # =========================

    all_chunks = []

    document_registry = {}

    global_chunk_id = 0


    # =========================
    # PROCESS EACH PDF
    # =========================

    for pdf_file in pdf_files:

        print(f"\nPROCESSING: {pdf_file}")


        pdf_path = os.path.join(

            DOCUMENTS_FOLDER,

            pdf_file
        )


        (

             document_chunks,
             document_summary,
             document_keywords,
             global_chunk_id
        ) = process_pdf_document(

            pdf_path,

            global_chunk_id
        )


        all_chunks.extend(

            document_chunks
        )


        document_registry[pdf_file] = {

            "summary": document_summary,

            "keywords": document_keywords,

            "sample_chunks": [

                chunk["content"][:500]

                for chunk in sorted(

                    document_chunks,

                    key=lambda x: len(x["content"]),

                    reverse=True

                )[:15]
            ]
        }

        print("\n========================")
        print("DOCUMENT PROFILE DEBUG")
        print("========================")

        print("FILE:", pdf_file)

        print("\nSUMMARY:")
        print(document_summary[:500])

        print("\nKEYWORDS:")
        print(document_keywords[:20])

        print("\nSAMPLE CHUNKS:")
        for chunk in document_registry[pdf_file]["sample_chunks"][:3]:

            print("-", chunk[:200])


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
    # SAVE DATABASE FILES
    # =========================

    save_database_files(

        vector_store,

        all_chunks,

        document_registry
    )


    print("\nDOCUMENT REGISTRY CREATED")

    print("\nMULTI-DOCUMENT INDEXING COMPLETED")

    print("\nTOTAL CHUNKS STORED:")
    print(len(all_chunks))

    print("\nTOTAL DOCUMENTS REGISTERED:")
    print(len(document_registry))

    return all_chunks