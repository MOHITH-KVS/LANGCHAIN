#GOAL OF THIS FILE

#This file should ONLY:

#| Responsibility              | Why                              |
#| --------------------------- | -------------------------------- |
#| Load saved vector database  | scalable retrieval system        |
#| Handle user queries         | chatbot interaction              |
#| Execute retrieval flow      | document question answering      |
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

import pickle


# =========================
# DEBUG MODE
# =========================

DEBUG = True


# =========================
# LOAD SAVED CHUNKS
# =========================

with open("chunks.pkl", "rb") as f:

    all_chunks = pickle.load(f)


print("\nTOTAL CHUNKS:")
print(len(all_chunks))


# =========================
# LOAD SAVED FAISS DATABASE
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
    # HYBRID RETRIEVAL
    # =========================

    retrieved_chunks = hybrid_retrieve(

        vector_store,

        bm25,

        all_chunks,

        rewritten_query
    )


    # =========================
    # RERANKING
    # =========================

    reranked_chunks = rerank_chunks(

        rewritten_query,

        retrieved_chunks
    )


    # =========================
    # CONTEXT EXPANSION
    # =========================

    expanded_chunks = []

    seen_chunks = set()


    for chunk, score in reranked_chunks[:3]:

        current_chunk_id = chunk.metadata.get(

            "chunk_id",

            0
        )


        nearby_chunk_ids = [

            current_chunk_id - 1,

            current_chunk_id,

            current_chunk_id + 1
        ]


        for nearby_id in nearby_chunk_ids:

            for stored_chunk in all_chunks:

                if stored_chunk.get("chunk_id") == nearby_id:

                    content = stored_chunk["content"]

                    metadata = stored_chunk["metadata"]

                    unique_key = (

                        metadata["page"],

                        nearby_id
                    )


                    if unique_key not in seen_chunks:

                        expanded_chunks.append(

                            Document(

                                page_content=content,

                                metadata=metadata
                            )
                        )

                        seen_chunks.add(unique_key)


    # =========================
    # DEBUG RETRIEVAL
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