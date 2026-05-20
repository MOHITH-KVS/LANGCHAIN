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
    split_into_sections
)

from reranker import rerank_chunks
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

# Function to create FAISS vector store
def create_vector_store(chunks, embedding_model):

    vector_store = FAISS.from_texts(

        texts=[chunk["content"] for chunk in chunks],

        embedding=embedding_model,

        metadatas=[chunk["metadata"] for chunk in chunks]
    )

    return vector_store

# Function to check whether page is useful based on presence of keywords
def create_bm25_index(chunks):

    tokenized_chunks = [

        chunk["content"].lower().split()

        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)

    return bm25

# Function to retrieve relevant chunks based on query
def detect_query_section(query):

    query = query.lower()

    section_keywords = {

        "abstract": [
            "abstract",
            "summary"
        ],

        "introduction": [
            "introduction",
            "overview"
        ],

        "methodology": [
            "methodology",
            "implementation",
            "working"
        ],

        "architecture": [
            "architecture",
            "design",
            "system design"
        ],

        "technologies": [
            "technology",
            "tools",
            "tech stack"
        ],

        "conclusion": [
            "conclusion",
            "future work",
            "result"
        ]
    }

    for section, keywords in section_keywords.items():

        for keyword in keywords:

            if keyword in query:

                return section

    return None


# Function to retrieve relevant chunks
def retrieve_chunks(vector_store, query):

    target_section = detect_query_section(query)

    if target_section:

        results = vector_store.similarity_search(

            query,

            k=10,

            filter={
                "section": target_section
            }
        )

        if results:
            return results

    results = vector_store.similarity_search(

        query,

        k=10
    )

    return results

# Hybrid retrieval combining vector search and BM25 keyword search
def hybrid_retrieve(

    vector_store,

    bm25,

    chunks,

    query
):

    semantic_results = vector_store.similarity_search(

        query,

        k=5
    )

    tokenized_query = query.lower().split()

    bm25_scores = bm25.get_scores(tokenized_query)

    top_keyword_indices = sorted(

        range(len(bm25_scores)),

        key=lambda i: bm25_scores[i],

        reverse=True
    )[:5]

    keyword_results = [

        Document(

            page_content=chunks[i]["content"],

            metadata=chunks[i]["metadata"]
        )

        for i in top_keyword_indices
    ]

    combined_results = []

    seen = set()

    for doc in semantic_results:

        if doc.page_content not in seen:

            combined_results.append(doc)

            seen.add(doc.page_content)

    for doc in keyword_results:

        if doc.page_content not in seen:

            combined_results.append(doc)

            seen.add(doc.page_content)

    return combined_results


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