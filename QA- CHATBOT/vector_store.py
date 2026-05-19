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
    create_metadata
)

#function to create a FAISS vector store from chunks, returns the vector store
def create_vector_store(chunks, embedding_model):

    vector_store = FAISS.from_texts(
        texts=[chunk["content"] for chunk in chunks],

        embedding=embedding_model,

        metadatas=[chunk["metadata"] for chunk in chunks]
    )

    return vector_store

#function to perform similarity search on the vector store using a query, returns the top matching chunks
def retrieve_chunks(vector_store, query):

    results = vector_store.similarity_search(
        query,
        k=3
    )

    return results

if __name__ == "__main__":
        docs = load_pdf("GVP-MAAA DOCUMENTATION (1).pdf")

        raw_text = docs[0].page_content

        cleaned_text = clean_text(raw_text)

        section = detect_section(cleaned_text)

        metadata = create_metadata(
            "GVP-MAAA DOCUMENTATION (1).pdf",
            1,
            section
        )

        chunks = create_chunks(cleaned_text, metadata)

        vector_store = create_vector_store(
            chunks,
            embedding_model
        )

        print("VECTOR STORE CREATED SUCCESSFULLY")

        query = "What is GVP-MAAA project?"

        results = retrieve_chunks(
            vector_store,
            query
        )

        print("\nRETRIEVED CHUNKS:\n")

        for result in results:

            print(result.page_content)

            print("\nMETADATA:")
            print(result.metadata)

            print("\n" + "="*50 + "\n")