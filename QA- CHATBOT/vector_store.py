#GOAL OF THIS FILE

#This file should ONLY:

#| Responsibility                    | Why                                |
#| --------------------------------- | ---------------------------------- |
#| Store embeddings                  | searchable semantic memory         |
#| Create FAISS index                | fast vector search                 |
#| Perform hybrid retrieval          | semantic + keyword retrieval       |
#| Apply metadata filtering          | retrieval precision improvement    |
#| Return relevant chunks            | contextual retrieval               |
#| Connect retrieval with RAG        | industrial RAG pipeline            |


from langchain_community.vectorstores import FAISS

from rank_bm25 import BM25Okapi

from langchain_core.documents import Document

from embeddings import embedding_model


# =========================
# CREATE VECTOR STORE
# =========================

def create_vector_store(

    chunks,

    embedding_model
):

    vector_store = FAISS.from_texts(

        texts=[chunk["content"] for chunk in chunks],

        embedding=embedding_model,

        metadatas=[

            {

                **chunk["metadata"],

                "chunk_id": chunk["chunk_id"]
            }

            for chunk in chunks
        ]
    )

    return vector_store


# =========================
# CREATE BM25 INDEX
# =========================

def create_bm25_index(all_chunks):

    tokenized_chunks = [

        chunk["content"].split()

        for chunk in all_chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)

    return bm25


# =========================
# DETECT SECTION FILTER
# =========================

def detect_section_filter(query):

    query_lower = query.lower()

    section_keywords = [

        "abstract",

        "introduction",

        "methodology",

        "architecture",

        "conclusion",

        "technologies",

        "implementation",

        "results",

        "objectives"
    ]


    for section in section_keywords:

        if section in query_lower:

            return section

    return None


# =========================
# HYBRID RETRIEVAL
# =========================

def hybrid_retrieve(

    vector_store,

    bm25,

    all_chunks,

    query,

    k=5
):


    # =========================
    # DETECT SECTION FILTER
    # =========================

    section_filter = detect_section_filter(query)


    # =========================
    # FILTER CHUNKS
    # =========================

    if section_filter:

        filtered_chunks = [

            chunk

            for chunk in all_chunks

            if chunk["metadata"]["section"]

            ==

            section_filter
        ]

    else:

        filtered_chunks = all_chunks


    # =========================
    # FAISS RETRIEVAL
    # =========================

    semantic_results = vector_store.similarity_search(

        query,

        k=k
    )


    semantic_docs = []


    for doc in semantic_results:

        if section_filter:

            if doc.metadata["section"] == section_filter:

                semantic_docs.append(doc)

        else:

            semantic_docs.append(doc)


    # =========================
    # BM25 RETRIEVAL
    # =========================

    bm25_query = query.split()

    bm25_scores = bm25.get_scores(bm25_query)


    bm25_ranked = sorted(

        zip(filtered_chunks, bm25_scores),

        key=lambda x: x[1],

        reverse=True
    )


    bm25_docs = []


    for chunk, score in bm25_ranked[:k]:

        bm25_docs.append(

            Document(

                page_content=chunk["content"],

                metadata={

                    **chunk["metadata"],

                    "chunk_id": chunk["chunk_id"]
                }
            )
        )


    # =========================
    # MERGE RESULTS
    # =========================

    combined_results = []

    seen = set()


    for doc in semantic_docs + bm25_docs:

        unique_key = (

            doc.metadata.get("page", -1),

            doc.metadata.get("chunk_id", -1)
        )


        if unique_key not in seen:

            combined_results.append(doc)

            seen.add(unique_key)


    return combined_results[:k]