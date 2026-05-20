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

docs = load_pdf(
    "GVP-MAAA DOCUMENTATION (1).pdf"
)

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

vector_store = create_vector_store(
    all_chunks,
    embedding_model
)
bm25 = create_bm25_index(all_chunks)
while True:

    question = input("\nAsk Question: ")

    if question.lower() == "exit":
        break

    retrieved_chunks = hybrid_retrieve(

        vector_store,

        bm25,

        all_chunks,

        question
    )
    reranked_chunks = rerank_chunks(
        question,
        retrieved_chunks
    )

    print("\nDEBUG RETRIEVED CHUNKS:\n")

    for chunk, score in reranked_chunks[:3]:

        print("SECTION:")
        print(chunk.metadata)

        print("\nCONTENT:")
        print(chunk.page_content[:500])

        print("\nSCORE:")
        print(score)

        print("\n" + "="*50)
        answer = generate_answer(
            question,
            reranked_chunks
        )

    print("\nANSWER:\n")

    print(answer)

