# GOAL OF THIS FILE

# This file should ONLY:

#| Responsibility              | Why                              |
#| --------------------------- | -------------------------------- |
#| Load RAG resources          | reusable backend engine          |
#| Execute retrieval pipeline  | centralized orchestration        |
#| Generate grounded answers   | production AI backend            |
#| Return structured responses | API-ready architecture           |


from time import time

from datetime import datetime

from langchain_community.vectorstores import FAISS

from langchain_core.documents import Document

from embeddings import embedding_model

from vector_store import (

    hybrid_retrieve
)

from reranker import rerank_chunks

from generation import generate_answer

from services.context_compressor import compress_context

from services.answer_validator import validate_context

from services.cache_manager import (
    get_cached_response,
    save_to_cache
)

from query_rewriter import rewrite_query

from document_router import route_documents

from services.logger import save_log


from resource_manager import (

    initialize_resources,

    get_vector_store,

    get_all_chunks,

    get_bm25_index
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

MIN_RELEVANCE_SCORE = 0.20

MAX_CONTEXT_CHUNKS = 15

RELATIVE_SCORE_THRESHOLD = 0.50

MAX_CONTEXT_TOKENS = 3500

MIN_RERANK_SCORE = 0.0

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
    
    import time

    start_time = time.time()
    
    
    # =========================
    # LOAD CACHED RESOURCES
    # =========================

    all_chunks = get_all_chunks()

    vector_store = get_vector_store()

    bm25 = get_bm25_index()


    # =========================
    # CACHE CHECK
    # =========================

    if document_name is None:

        cached_response = get_cached_response(
            question
        )

        if cached_response:
            print("\nCACHE HIT")
            return cached_response


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


    print("\nCHAT HISTORY LENGTH:")
    print(len(chat_history))

    print("\nCHAT HISTORY CONTENT:")
    print(chat_history)
    

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


    print("\n" + "="*60)
    print("ORIGINAL QUESTION:")
    print(question)

    print("\nREWRITTEN QUERY:")
    print(rewritten_query)
    print("="*60)


    # =========================
    # DOCUMENT ROUTING
    # =========================

    matched_documents = route_documents(

        rewritten_query,

        top_k=5
    )

    print("\nMATCHED DOCUMENTS")
    print(matched_documents)


    # =========================
    # ROUTER FALLBACK
    # =========================

    if len(matched_documents) == 0:

        print("\nROUTER FAILED - FALLING BACK TO GLOBAL SEARCH")

        selected_documents = None

        document_score = 0

    else:

        selected_documents = [

            matched_documents[0]["source"]
        ]

        document_score = matched_documents[0]["score"]

        if document_score < 0.35:

            selected_documents = None


    # =========================
    # OPTIONAL MANUAL DOCUMENT
    # =========================

    if document_name is not None:

        selected_documents = [document_name]




    print("\n" + "="*60)
    print("SELECTED DOCUMENTS:")
    print(selected_documents)
    print("="*60)

    # =========================
    # HYBRID RETRIEVAL
    # =========================

    combined_query = question + " " + rewritten_query + " " + question

    retrieved_chunks = hybrid_retrieve(

        vector_store,

        bm25,

        all_chunks,

        combined_query,

        source_filter=selected_documents,

        k=40
    )


            

    print("\nTOP RETRIEVED CHUNKS")

    for idx, (chunk, score) in enumerate(retrieved_chunks):

        chunk_id = chunk.metadata.get("chunk_id", "UNKNOWN")

        print(
            f"\nRANK {idx+1}"
        )

        print(
            "CHUNK ID:",
            chunk_id
        )

        print(
            "SOURCE:",
            chunk.metadata.get("source")
        )

        print(
            "SECTION:",
            chunk.metadata.get("section")
        )

        print(
            chunk.page_content[:200]
        )

        print("=" * 50)


    print("\nHYBRID RETRIEVAL RESULTS:\n")

    for chunk, score in retrieved_chunks[:5]:

        print("SECTION:")
        print(chunk.metadata)

        print("\nHYBRID SCORE:")
        print(score)

        print("\nCONTENT:")
        print(chunk.page_content[:300])

        print("\n" + "=" * 50)


    print("\nRETRIEVED CHUNKS COUNT:")
    print(len(retrieved_chunks))

    for chunk, score in retrieved_chunks[:3]:

        print("\nCHUNK:")
        print(chunk.page_content[:300])

        print("\nMETADATA:")
        print(chunk.metadata)


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

            if score >= MIN_RERANK_SCORE:

                filtered_chunks.append((chunk, score))


    # =========================
    # SORT ONLY BY RERANK SCORE
    # =========================

    filtered_chunks = sorted(

        filtered_chunks,

        key=lambda x: x[1],

        reverse=True
    )

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

            "answer": "No relevant information found.",
            "sources": []
        }


    top_rerank_score = filtered_chunks[0][1]


    # =========================
    # CONTEXT COMPRESSION
    # =========================

    filtered_chunks = compress_context(
        filtered_chunks
    )

    is_valid = validate_context(
        filtered_chunks
    )


    if not is_valid:

        return {
            "answer": (
                "The uploaded documents do not contain "
                "enough information to answer this question."
            ),
            "sources": []
        }

    print("\nCOMPRESSED CHUNKS:")
    print(len(filtered_chunks))



    # =========================
    # RETRIEVAL CONFIDENCE CHECK
    # =========================


    if len(filtered_chunks) < MIN_CONTEXT_CHUNKS:

        return {

            "answer": (

                "Not enough relevant information "

                "was found in the documents."
            ),

            "sources": []
        }


    # =========================
    # CONTEXT PACKING
    # =========================

    expanded_chunks = []

    seen_chunks = set()

    current_token_count = 0


    for chunk, score in filtered_chunks[:MAX_CONTEXT_CHUNKS]:

        if len(chunk.page_content.strip()) < 40:
            continue

        unique_key = (

            chunk.metadata["source"],
            chunk.metadata["page"],
            chunk.metadata["chunk_id"]
        )

        if unique_key in seen_chunks:

            continue

        estimated_tokens = count_tokens(

            chunk.page_content
        )

        if (

            current_token_count + estimated_tokens

            >

            MAX_CONTEXT_TOKENS
        ):

            continue

        print("\nFINAL CONTEXT CHUNK:")
        print(chunk.page_content[:500])

        print("\nFINAL SCORE:")
        print(score)

        print("\n" + "=" * 60)

        expanded_chunks.append(chunk)

        current_token_count += estimated_tokens

        seen_chunks.add(unique_key)


    # =========================
    # GENERATION INPUT
    # =========================

    generation_chunks = []


    for expanded_chunk in expanded_chunks:

        matching_score = 0.0


        for original_chunk, original_score in filtered_chunks:

            if (

                expanded_chunk.page_content

                ==

                original_chunk.page_content
            ):

                matching_score = original_score

                break


        generation_chunks.append(

            (expanded_chunk, matching_score)
        )

    

    # =========================
    # GENERATE ANSWER
    # =========================

    generation_result = generate_answer(

        question,

        generation_chunks,

        conversation_context
    )

    print("\nFINAL GENERATION CHUNKS:\n")

    for chunk, score in generation_chunks:

        print("SOURCE:")
        print(chunk.metadata.get("source"))

        print("\nPAGE:")
        print(chunk.metadata.get("page"))

        print("\nSCORE:")
        print(score)

        print("\nCONTENT:")
        print(chunk.page_content[:500])

        print("\n" + "=" * 80)


    # =========================
    # CLEAN FINAL ANSWER
    # =========================

    final_answer = generation_result["answer"]

    while final_answer.endswith(","):
        final_answer = final_answer[:-1].strip()

    # =========================
    # SAVE CHAT HISTORY
    # =========================

    chat_history.append({

        "question": question,

        "answer": final_answer,

        "sources": generation_result["sources"]
    })



    # =========================
    # SAVE LOG
    # =========================

    save_log({

        "timestamp": str(datetime.now()),

        "question": question,

        "rewritten_query": rewritten_query,

        "selected_documents": selected_documents,

        "document_score": document_score,

        "rerank_score": float(top_rerank_score)
        if top_rerank_score is not None
        else None,

        "context_tokens": current_token_count,

        "answer": final_answer,

        "sources": generation_result["sources"],

        "response_time": round(
            time.time() - start_time,
            2
        )
    })



    # =========================
    # RETURN RESPONSE
    # =========================

    final_response = {

        "question": question,

        "rewritten_query": rewritten_query,

        "documents": selected_documents,

        "document_score": float(document_score)
        if document_score is not None
        else None,

        "rerank_score": float(top_rerank_score)
        if top_rerank_score is not None
        else None,

        "context_tokens": int(current_token_count),

        "answer": final_answer,

        "sources": generation_result["sources"]

    }


    # =========================
    # SAVE TO CACHE
    # =========================

    save_to_cache(

        question,

        final_response
    )


    # =========================
    # RESPONSE TIME
    # =========================

    response_time = round(

        time.time() - start_time,

        2
    )


    # =========================
    # LOG ENTRY
    # =========================

    log_entry = {

        "timestamp": str(datetime.now()),

        "question": question,

        "rewritten_query": rewritten_query,

        "document_score": float(document_score)
        if document_score is not None
        else None,

        "rerank_score": float(top_rerank_score)
        if top_rerank_score is not None
        else None,

        "response_time": response_time,

        "answer": generation_result["answer"]
    }

    save_log(log_entry)


    return final_response

# =========================
# INITIALIZE RESOURCES
# =========================

initialize_resources()

if __name__ == "__main__":

    response = ask_question(

        "Explain abstract"
    )

    print(response)