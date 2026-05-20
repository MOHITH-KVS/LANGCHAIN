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
    create_metadata,
    split_into_sections
)

from chunking import create_chunks

from embeddings import embedding_model

from vector_store import (
    create_vector_store,
    create_bm25_index,
    hybrid_retrieve
)

from reranker import rerank_chunks

from generation import generate_answer

from query_rewriter import rewrite_query


# =========================
# DEBUG MODE
# =========================

DEBUG = True


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
# CREATE BM25 INDEX
# =========================

bm25 = create_bm25_index(all_chunks)


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
    # DEBUG RETRIEVAL
    # =========================

    if DEBUG:

        print("\nDEBUG RETRIEVED CHUNKS:\n")

        for chunk, score in reranked_chunks[:3]:

            print("SECTION:")
            print(chunk.metadata)

            print("\nCONTENT:")
            print(chunk.page_content[:500])

            print("\nSCORE:")
            print(score)

            print("\n" + "=" * 50)


    # =========================
    # GENERATE ANSWER
    # =========================

    answer = generate_answer(

        question,

        reranked_chunks,

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