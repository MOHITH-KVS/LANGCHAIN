# GOAL OF THIS FILE

# This file should ONLY:

#| Responsibility              | Why                              |
#| --------------------------- | -------------------------------- |
#| Load RAG resources          | reusable backend engine          |
#| Execute retrieval pipeline  | centralized orchestration        |
#| Generate grounded answers   | production AI backend            |
#| Return structured responses | API-ready architecture           |


from langchain_community.vectorstores import FAISS

from langchain_core.documents import Document

from embeddings import embedding_model

from vector_store import (

    hybrid_retrieve
)

from reranker import rerank_chunks

from generation import generate_answer

from query_rewriter import rewrite_query

from document_router import route_documents

from engine.bm25_retriever import (

    create_bm25_index,

    bm25_search
)


import pickle

import tiktoken

import os


# =========================
# TOKENIZER
# =========================

tokenizer = tiktoken.get_encoding("cl100k_base")


# =========================
# RETRIEVAL SETTINGS
# =========================

MIN_RELEVANCE_SCORE = 0.3

MAX_CONTEXT_CHUNKS = 3

RELATIVE_SCORE_THRESHOLD = 0.30

MAX_CONTEXT_TOKENS = 1800

MIN_RERANK_SCORE = 0.1

MIN_CONTEXT_CHUNKS = 1


# =========================
# TOKEN COUNT FUNCTION
# =========================

def count_tokens(text):

    return len(

        tokenizer.encode(text)
    )


# =========================
# REMOVE DUPLICATE CHUNKS
# =========================

def remove_duplicate_chunks(reranked_chunks):

    unique_chunks = []

    seen_contents = set()


    for chunk, score in reranked_chunks:


        normalized_content = (

            chunk.page_content
            .strip()
            .lower()
        )


        # =========================
        # DUPLICATE CHECK
        # =========================

        if normalized_content in seen_contents:

            continue


        seen_contents.add(normalized_content)


        unique_chunks.append(

            (chunk, score)
        )


    return unique_chunks




# =========================
# CHAT MEMORY
# =========================

chat_history = []






# =========================
# MAIN CHATBOT FUNCTION
# =========================

def ask_question(

    question,

    document_name=None
):
    
        # =========================
    # LOAD SAVED CHUNKS
    # =========================

    if os.path.exists("chunks.pkl"):

        with open("chunks.pkl", "rb") as f:

            all_chunks = pickle.load(f)

    else:

        return {

            "answer": (

                "No documents are uploaded yet. "

                "Please upload PDFs first."
            ),

            "sources": []
        }


    # =========================
    # LOAD VECTOR DATABASE
    # =========================

    if os.path.exists("faiss_index"):

        vector_store = FAISS.load_local(

            "faiss_index",

            embedding_model,

            allow_dangerous_deserialization=True
        )

    else:

        return {

            "answer": "Vector database not found.",

            "sources": []
        }


    # =========================
    # CREATE BM25 INDEX
    # =========================

    bm25 = create_bm25_index(all_chunks)


    # =========================
    # CREATE METADATA INDEXES
    # =========================

    source_index = {}

    section_index = {}

    chunk_id_index = {}


    for chunk in all_chunks:

        source = chunk["metadata"]["source"]

        section = chunk["metadata"]["section"]

        chunk_id = chunk["chunk_id"]


        if source not in source_index:

            source_index[source] = []


        source_index[source].append(chunk)


        section_key = (source, section)


        if section_key not in section_index:

            section_index[section_key] = []


        section_index[section_key].append(chunk)


        chunk_id_index[chunk_id] = chunk


    # =========================
    # EMPTY QUESTION CHECK
    # =========================

    if not question.strip():

        return {

            "answer": "Please enter a valid question."
        }


    # =========================
    # EMPTY DATABASE CHECK
    # =========================

    if (

        vector_store is None

        or

        bm25 is None

        or

        len(all_chunks) == 0
    ):

        return {

            "answer": (

                "No documents are uploaded yet. "

                "Please upload PDFs first."
            ),

            "sources": []
        }

    # =========================
    # CONVERSATION MEMORY
    # =========================

    conversation_context = ""


    for chat in chat_history[-3:]:

        conversation_context += (

            f"\nUser: {chat['question']}\n"

            f"Assistant: {chat['answer']}\n"
        )


    # =========================
    # QUERY REWRITING
    # =========================

    rewritten_query = rewrite_query(

        question,

        conversation_context
    )


    # =========================
    # DOCUMENT ROUTING
    # =========================

    matched_documents = route_documents(

        rewritten_query,

        top_k=1
    )


    if len(matched_documents) == 0:

        return {

            "answer": "No relevant document found.",

            "sources": []
        }

    selected_document = matched_documents[0]["source"]

    document_score = matched_documents[0]["score"]


    # =========================
    # ROUTING CONFIDENCE CHECK
    # =========================

    if document_score < 0.30:

        selected_document = None


    # =========================
    # OPTIONAL MANUAL DOCUMENT
    # =========================

    if document_name is not None:

        selected_document = document_name

    print("\nSELECTED DOCUMENT:")
    print(selected_document)

    print("\nDOCUMENT MATCH SCORE:")
    print(document_score)


    # =========================
    # HYBRID RETRIEVAL
    # =========================

    retrieved_chunks = hybrid_retrieve(

        vector_store,

        bm25,

        all_chunks,

        rewritten_query,

        source_filter=selected_document,

        k=15
    )

    print("\nHYBRID RETRIEVAL RESULTS:\n")

    for chunk, score in retrieved_chunks[:5]:

        print("SECTION:")
        print(chunk.metadata)

        print("\nHYBRID SCORE:")
        print(score)

        print("\nCONTENT:")
        print(chunk.page_content[:300])

        print("\n" + "=" * 50)


    # =========================
    # RERANKING
    # =========================

    reranked_chunks = rerank_chunks(

        rewritten_query,

        [chunk for chunk, score in retrieved_chunks]
    )

    print("\nRERANKED RESULTS:\n")

    for item in reranked_chunks[:5]:

        print("\nDEBUG ITEM:")
        print(item)

        if isinstance(item, tuple):

            chunk = item[0]
            score = item[1]

        else:

            chunk = item
            score = "N/A"

        print("SECTION:")
        print(chunk.metadata)

        print("\nRERANK SCORE:")
        print(score)

        print("\nCONTENT:")
        print(chunk.page_content[:300])

        print("\n" + "=" * 50)


    


    # =========================
    # RELEVANCE FILTERING
    # =========================

    filtered_chunks = []

    for item in reranked_chunks:

        if isinstance(item, tuple):

            chunk = item[0]
            score = item[1]

        else:

            chunk = item
            score = 0

        if score >= MIN_RERANK_SCORE:

            filtered_chunks.append((chunk, score))


    # fallback safety
    if len(filtered_chunks) == 0:

        filtered_chunks = reranked_chunks[:3]


    # =========================
    # REMOVE DUPLICATES
    # =========================

    filtered_chunks = remove_duplicate_chunks(

        filtered_chunks
    )

    # =========================
    # DEBUG RETRIEVED CHUNKS
    # =========================

    print("\nDEBUG RETRIEVED CHUNKS:\n")

    for item in filtered_chunks[:3]:

        if isinstance(item, tuple):

            chunk = item[0]
            score = item[1]

        else:

            chunk = item
            score = "N/A"

        print("SECTION:")
        print(chunk.metadata)

        print("\nCONTENT:")
        print(chunk.page_content[:400])

        print("\nSCORE:")
        print(score)

        print("\n" + "=" * 50)


    # =========================
    # RETRIEVAL VALIDATION
    # =========================

    if len(filtered_chunks) == 0:

        return {

            "answer": "No relevant information found."
        }


    top_rerank_score = filtered_chunks[0][1]



    # =========================
    # RETRIEVAL CONFIDENCE CHECK
    # =========================

    if top_rerank_score < MIN_RERANK_SCORE:

        return {

            "answer": (

                "The retrieved information is not "

                "confident enough to answer reliably."
            ),

            "sources": []
        }


    if len(filtered_chunks) < MIN_CONTEXT_CHUNKS:

        return {

            "answer": (

                "Not enough relevant information "

                "was found in the documents."
            ),

            "sources": []
        }


    
    # =========================
    # STABLE CONTEXT PACKING
    # =========================

    expanded_chunks = []

    seen_chunks = set()

    used_sections = set()

    current_token_count = 0


    # =========================
    # TAKE TOP RERANKED CHUNKS
    # =========================

    top_chunks = filtered_chunks[:MAX_CONTEXT_CHUNKS]


    for chunk, score in top_chunks:


        target_section = chunk.metadata["section"]


        # =========================
        # DIVERSITY CONTROL
        # =========================

        section_key = (

            chunk.metadata["source"],

            target_section
        )


        # Avoid excessive duplicate sections

        if section_key in used_sections:

            continue


        used_sections.add(section_key)


        base_chunk_id = chunk.metadata["chunk_id"]


        # =========================
        # CONTEXT WINDOW
        # =========================

        nearby_chunk_ids = [

            base_chunk_id - 1,

            base_chunk_id,

            base_chunk_id + 1
        ]


        for nearby_id in nearby_chunk_ids:


            if nearby_id not in chunk_id_index:

                continue


            nearby_chunk = chunk_id_index[nearby_id]

            if (
                nearby_chunk["metadata"]["source"]
                !=
                chunk.metadata["source"]
            ):
                continue


            # =========================
            # SAME DOCUMENT CHECK
            # =========================

            if selected_document:

                if (

                    nearby_chunk["metadata"]["source"]

                    !=

                    selected_document
                ):

                    continue


            # =========================
            # SAME SECTION CHECK
            # =========================

            if (

                nearby_chunk["metadata"]["section"]

                !=

                target_section
            ):

                continue


            metadata = {

                **nearby_chunk["metadata"],

                "chunk_id": nearby_chunk["chunk_id"]
            }


            unique_key = (

                metadata["source"],

                metadata["page"],

                metadata["chunk_id"]
            )


            if unique_key in seen_chunks:

                continue


            content = nearby_chunk["content"]

            estimated_tokens = count_tokens(content)


            # =========================
            # TOKEN BUDGET CONTROL
            # =========================

            if (

                current_token_count + estimated_tokens

                >

                MAX_CONTEXT_TOKENS
            ):

                continue


            expanded_chunks.append(

                Document(

                    page_content=content,

                    metadata=metadata
                )
            )


            current_token_count += estimated_tokens

            seen_chunks.add(unique_key)


    # =========================
    # GENERATE ANSWER
    # =========================

    generation_result = generate_answer(

        question,

        [(chunk, 0) for chunk in expanded_chunks],

        conversation_context
    )


    # =========================
    # SAVE CHAT HISTORY
    # =========================

    chat_history.append({

        "question": question,

        "answer": generation_result["answer"],

        "sources": generation_result["sources"]
    })


    # =========================
    # RETURN RESPONSE
    # =========================

    return {

        "question": question,

        "rewritten_query": rewritten_query,

        "document": selected_document,

        "document_score": float(document_score)
        if document_score is not None
        else None,

        "rerank_score": float(top_rerank_score)
        if top_rerank_score is not None
        else None,

        "context_tokens": int(current_token_count),

        "answer": generation_result["answer"],

        "sources": generation_result["sources"]
    }

if __name__ == "__main__":

    response = ask_question(

        "Explain abstract"
    )

    print(response)