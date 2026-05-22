#GOAL OF THIS FILE

#This file should ONLY:

#| Responsibility              | Why                              |
#| --------------------------- | -------------------------------- |
#| Load saved vector database  | scalable retrieval system        |
#| Handle user queries         | chatbot interaction              |
#| Execute hierarchical RAG    | enterprise retrieval orchestration |
#| Perform document routing    | multi-document precision         |
#| Validate retrieval quality  | hallucination prevention         |
#| Compress retrieval context  | efficient grounded generation    |
#| Perform semantic packing    | coherent answer generation       |
#| Generate grounded answers   | accurate PDF responses           |
#| Create conversational loop  | interactive chatbot              |


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
# DEBUG MODE
# =========================

DEBUG = True


# =========================
# TOKENIZER
# =========================

tokenizer = tiktoken.get_encoding("cl100k_base")


# =========================
# RETRIEVAL SETTINGS
# =========================

MIN_RELEVANCE_SCORE = 2.0

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


print("\nTOTAL CHUNKS:")

print(len(all_chunks))


# =========================
# LOAD FAISS DATABASE
# =========================

vector_store = FAISS.load_local(

    "faiss_index",

    embedding_model,

    allow_dangerous_deserialization=True
)


print("\nFAISS DATABASE LOADED SUCCESSFULLY")


# =========================
# CREATE BM25 INDEX
# =========================

bm25 = create_bm25_index(all_chunks)


print("\nBM25 INDEX CREATED SUCCESSFULLY")


# =========================
# CHAT MEMORY
# =========================

chat_history = []


# =========================
# CHATBOT LOOP
# =========================

while True:

    question = input("\nAsk Question: ")


    # =========================
    # EMPTY QUESTION HANDLING
    # =========================

    if not question.strip():

        print("\nPlease enter a valid question.")

        continue


    # =========================
    # EXIT HANDLING
    # =========================

    if question.lower() in ["exit", "quit", "bye"]:

        print("\nChatbot session ended.")

        break


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


    print("\nREWRITTEN QUERY:")

    print(rewritten_query)


    # =========================
    # DOCUMENT ROUTING
    # =========================

    matched_documents = route_documents(

        rewritten_query,

        top_k=1
    )


    selected_document = matched_documents[0]["source"]

    document_score = matched_documents[0]["score"]


    print("\nSELECTED DOCUMENT:")

    print(selected_document)


    print("\nDOCUMENT MATCH SCORE:")

    print(round(document_score, 4))


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
    # RETRIEVAL VALIDATION
    # =========================

    if len(reranked_chunks) == 0:

        print("\nANSWER:\n")

        print(

            "No relevant information found in the uploaded documents."
        )

        continue


    top_rerank_score = reranked_chunks[0][1]


    print("\nTOP RERANK SCORE:")

    print(round(top_rerank_score, 4))


    # =========================
    # LOW CONFIDENCE DETECTION
    # =========================

    if top_rerank_score < MIN_RELEVANCE_SCORE:

        print("\nANSWER:\n")

        print(

            "The uploaded documents do not contain enough relevant information."
        )

        continue


    # =========================
    # SMART CONTEXT SELECTION
    # =========================

    selected_reranked_chunks = []


    for chunk, score in reranked_chunks:

        relative_score = score / top_rerank_score


        if score < MIN_RELEVANCE_SCORE:

            continue


        if relative_score < RELATIVE_SCORE_THRESHOLD:

            continue


        selected_reranked_chunks.append(

            (chunk, score)
        )


        if len(selected_reranked_chunks) >= MAX_CONTEXT_CHUNKS:

            break


    print("\nSELECTED HIGH-CONFIDENCE CHUNKS:")


    for chunk, score in selected_reranked_chunks:

        print(

            f"Page: {chunk.metadata['page']} | "

            f"Section: {chunk.metadata['section']} | "

            f"Score: {round(score, 2)}"
        )


    # =========================
    # SEMANTIC CONTEXT PACKING
    # =========================

    expanded_chunks = []

    seen_chunks = set()

    current_token_count = 0


    for chunk, score in selected_reranked_chunks:

        target_section = chunk.metadata["section"]


        same_section_chunks = [

            stored_chunk

            for stored_chunk in all_chunks

            if (

                stored_chunk["metadata"]["source"]

                ==

                selected_document
            )

            and

            (

                stored_chunk["metadata"]["section"]

                ==

                target_section
            )
        ]


        for stored_chunk in same_section_chunks:

            content = stored_chunk["content"]

            metadata = {

                **stored_chunk["metadata"],

                "chunk_id": stored_chunk["chunk_id"]
            }


            unique_key = (

                metadata["source"],

                metadata["page"],

                metadata["chunk_id"]
            )


            if unique_key in seen_chunks:

                continue


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


    print("\nTOTAL CONTEXT TOKENS:")

    print(current_token_count)


    # =========================
    # DEBUG MODE
    # =========================

    if DEBUG:

        print("\nDEBUG EXPANDED CHUNKS:\n")


        for chunk in expanded_chunks[:5]:

            print("SECTION:")

            print(chunk.metadata)


            print("\nCONTENT:")

            print(chunk.page_content[:500])


            print("\n" + "=" * 50)


    # =========================
    # GENERATE ANSWER
    # =========================

    answer = generate_answer(

        question,

        [(chunk, 0) for chunk in expanded_chunks],

        conversation_context
    )


    # =========================
    # PRINT ANSWER
    # =========================

    print("\nANSWER:\n")

    print(answer)


    # =========================
    # SAVE CHAT HISTORY
    # =========================

    chat_history.append({

        "question": question,

        "answer": answer
    })