#GOAL OF THIS FILE

#| Responsibility                    | Why                        |
#| --------------------------------- | -------------------------- |
#| Store embeddings                  | searchable semantic memory |
#| Create FAISS index                | fast vector search         |
#| Perform similarity search         | retrieve relevant chunks   |
#| Return top matching chunks        | contextual retrieval       |
#| Connect embeddings with retrieval | semantic RAG pipeline      |

from langchain_community.vectorstores import FAISS

from embeddings import embedding_model

from chunking import create_chunks

from document_processor import (
    load_pdf,
    clean_text,
    detect_section,
    create_metadata,
    is_useful_page
)

from reranker import rerank_chunks


# Function to create FAISS vector store
def create_vector_store(chunks, embedding_model):

    vector_store = FAISS.from_texts(

        texts=[chunk["content"] for chunk in chunks],

        embedding=embedding_model,

        metadatas=[chunk["metadata"] for chunk in chunks]
    )

    return vector_store


# Function to retrieve relevant chunks
def retrieve_chunks(vector_store, query):

    results = vector_store.similarity_search(

        query,

        k=10
    )

    return results


# Testing Section
if __name__ == "__main__":

    docs = load_pdf(
        "GVP-MAAA DOCUMENTATION (1).pdf"
    )

    all_chunks = []

    for page_num, doc in enumerate(docs):

        raw_text = doc.page_content

        if not is_useful_page(raw_text):
            continue

        cleaned_text = clean_text(raw_text)

        section = detect_section(cleaned_text)

        metadata = create_metadata(

            "GVP-MAAA DOCUMENTATION (1).pdf",

            page_num + 1,

            section
        )

        chunks = create_chunks(

            cleaned_text,

            metadata
        )

        all_chunks.extend(chunks)

    vector_store = create_vector_store(

        all_chunks,

        embedding_model
    )

    print("TOTAL CHUNKS:")

    print(len(all_chunks))

    test_queries = [

        "What is GVP-MAAA project?",

        "Explain the methodology",

        "What technologies are used?",

        "What is the objective of the project?",

        "Explain the conclusion"
    ]

    for query in test_queries:

        print(f"\nQUERY: {query}")

        results = retrieve_chunks(

            vector_store,

            query
        )

        ranked_results = rerank_chunks(

            query,

            results
        )

        print("\nRERANKED CHUNKS:\n")

        for result, score in ranked_results[:3]:

            print("RELEVANCE SCORE:")

            print(score)

            print("\nCHUNK:\n")

            print(result.page_content[:500])

            print("\nMETADATA:")

            print(result.metadata)

            print("\n" + "="*50 + "\n")