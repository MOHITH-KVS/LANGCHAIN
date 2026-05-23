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

    create_bm25_index,

    hybrid_retrieve
)

from reranker import rerank_chunks

from generation import generate_answer

from query_rewriter import rewrite_query

from document_router import route_documents

import pickle

import tiktoken


# =========================
# TOKENIZER
# =========================

tokenizer = tiktoken.get_encoding("cl100k_base")


# =========================
# RETRIEVAL SETTINGS
# =========================

MIN_RELEVANCE_SCORE = 0.3

MAX_CONTEXT_CHUNKS = 5

RELATIVE_SCORE_THRESHOLD = 0.55

MAX_CONTEXT_TOKENS = 1800

# =========================
# TOKEN COUNT FUNCTION
# =========================

def count_tokens(text):

    return len(

        tokenizer.encode(text)
    )

# =========================
# LOAD SAVED CHUNKS
# =========================

with open("chunks.pkl", "rb") as f:

    all_chunks = pickle.load(f)


# =========================
# METADATA INDEXES
# =========================

source_index = {}

section_index = {}

chunk_id_index = {}


for chunk in all_chunks:

    source = chunk["metadata"]["source"]

    section = chunk["metadata"]["section"]

    chunk_id = chunk["chunk_id"]


    # SOURCE INDEX

    if source not in source_index:

        source_index[source] = []


    source_index[source].append(chunk)


    # SECTION INDEX

    section_key = (source, section)


    if section_key not in section_index:

        section_index[section_key] = []


    section_index[section_key].append(chunk)


    # CHUNK ID INDEX

    chunk_id_index[chunk_id] = chunk


# =========================
# LOAD VECTOR DATABASE
# =========================

vector_store = FAISS.load_local(

    "faiss_index",

    embedding_model,

    allow_dangerous_deserialization=True
)


# =========================
# CREATE BM25 INDEX
# =========================

bm25 = create_bm25_index(all_chunks)

# =========================
# CHAT MEMORY
# =========================

chat_history = []


# =========================
# MAIN CHATBOT FUNCTION
# =========================

def ask_question(question):


    # =========================
    # EMPTY QUESTION CHECK
    # =========================

    if not question.strip():

        return {

            "answer": "Please enter a valid question."
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


    selected_document = matched_documents[0]["source"]

    document_score = matched_documents[0]["score"]


    # =========================
    # ROUTING CONFIDENCE CHECK
    # =========================

    if document_score < 0.30:

        selected_document = None


    # =========================
    # HYBRID RETRIEVAL
    # =========================

    retrieved_chunks = hybrid_retrieve(

        vector_store,

        bm25,

        all_chunks,

        rewritten_query,

        source_filter=selected_document
    )


    # =========================
    # RERANKING
    # =========================

    reranked_chunks = rerank_chunks(


        rewritten_query,

        retrieved_chunks
    )

    # =========================
    # DEBUG RETRIEVED CHUNKS
    # =========================

    print("\nDEBUG RETRIEVED CHUNKS:\n")

    for chunk, score in reranked_chunks[:3]:

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

    if len(reranked_chunks) == 0:

        return {

            "answer": "No relevant information found."
        }


    top_rerank_score = reranked_chunks[0][1]


    
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

    top_chunks = reranked_chunks[:MAX_CONTEXT_CHUNKS]


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

    answer = generate_answer(

        question,

        [(chunk, 0) for chunk in expanded_chunks],

        conversation_context
    )


    # =========================
    # SAVE CHAT HISTORY
    # =========================

    chat_history.append({

        "question": question,

        "answer": answer
    })


    # =========================
    # RETURN RESPONSE
    # =========================

    return {

        "question": question,

        "rewritten_query": rewritten_query,

        "document": selected_document,

        "document_score": round(document_score, 4),

        "rerank_score": round(top_rerank_score, 4),

        "context_tokens": current_token_count,

        "answer": answer
    }

if __name__ == "__main__":

    response = ask_question(

        "Explain abstract"
    )

    print(response)