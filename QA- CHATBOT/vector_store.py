#GOAL OF THIS FILE

#This file should ONLY:

#| Responsibility                    | Why                                |
#| --------------------------------- | ---------------------------------- |
#| Store embeddings                  | searchable semantic memory         |
#| Create FAISS index                | fast vector search                 |
#| Perform hybrid retrieval          | semantic + keyword retrieval       |
#| Apply metadata filtering          | retrieval precision improvement    |
#| Support document routing          | hierarchical retrieval             |
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
# LOAD VECTOR STORE
# =========================

def load_vector_store(

    embedding_model,

    faiss_path="faiss_index"
):

    vector_store = FAISS.load_local(

        faiss_path,

        embedding_model,

        allow_dangerous_deserialization=True
    )

    return vector_store


# =========================
# ADD DOCUMENTS TO VECTOR STORE
# =========================

def add_documents_to_vector_store(

    vector_store,

    chunks
):

    texts = [

        chunk["content"]

        for chunk in chunks
    ]


    metadatas = [

        {

            **chunk["metadata"],

            "chunk_id": chunk["chunk_id"]
        }

        for chunk in chunks
    ]


    vector_store.add_texts(

        texts=texts,

        metadatas=metadatas
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

    source_filter=None,

    k=5
):


    # =========================
    # DETECT SECTION FILTER
    # =========================

    section_filter = detect_section_filter(query)


    # =========================
    # INITIAL FILTER
    # =========================

    filtered_chunks = all_chunks


    # =========================
    # SOURCE FILTER
    # =========================

    if source_filter:

        filtered_chunks = [

            chunk

            for chunk in filtered_chunks

            if chunk["metadata"]["source"]

            ==

            source_filter
        ]


    # =========================
    # SECTION FILTER
    # =========================

    if section_filter:

        filtered_chunks = [

            chunk

            for chunk in filtered_chunks

            if chunk["metadata"]["section"]

            ==

            section_filter
        ]


    # =========================
    # FAISS RETRIEVAL WITH SCORES
    # =========================

    semantic_results = vector_store.similarity_search_with_score(

        query,

        k=k * 5
    )


    semantic_docs = []


    # =========================
    # SEMANTIC SCORE FILTERING
    # =========================

    SEMANTIC_SCORE_THRESHOLD = 1.2


    for doc, score in semantic_results:

        source_match = True

        section_match = True


        if source_filter:

            source_match = (

                doc.metadata["source"]

                ==

                source_filter
            )


        if section_filter:

            section_match = (

                doc.metadata["section"]

                ==

                section_filter
            )


    # =========================
    # KEEP ONLY STRONG MATCHES
    # =========================

    if (

        source_match

        and

        section_match

        and

        score <= SEMANTIC_SCORE_THRESHOLD
    ):

        semantic_docs.append(doc)

    


    # =========================
    # BM25 RETRIEVAL
    # =========================

    bm25_query = query.split()


    filtered_texts = [

        chunk["content"].split()

        for chunk in filtered_chunks
    ]


    filtered_bm25 = BM25Okapi(filtered_texts)


    bm25_scores = filtered_bm25.get_scores(bm25_query)


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

    # =========================
    # RECIPROCAL RANK FUSION
    # =========================

    rrf_scores = {}

    RRF_K = 60


    # =========================
    # SEMANTIC RANKING
    # =========================

    for rank, doc in enumerate(semantic_docs):

        unique_key = (

            doc.metadata.get("source", ""),

            doc.metadata.get("page", -1),

            doc.metadata.get("chunk_id", -1)
        )


        score = 1 / (RRF_K + rank + 1)


        if unique_key not in rrf_scores:

            rrf_scores[unique_key] = {

                "doc": doc,

                "score": 0
            }


        rrf_scores[unique_key]["score"] += score


    # =========================
    # BM25 RANKING
    # =========================

    for rank, doc in enumerate(bm25_docs):

        unique_key = (

            doc.metadata.get("source", ""),

            doc.metadata.get("page", -1),

            doc.metadata.get("chunk_id", -1)
        )


        score = 1 / (RRF_K + rank + 1)


        if unique_key not in rrf_scores:

            rrf_scores[unique_key] = {

                "doc": doc,

                "score": 0
            }


        rrf_scores[unique_key]["score"] += score


    # =========================
    # FINAL RANKING
    # =========================

    ranked_results = sorted(

        rrf_scores.values(),

        key=lambda x: x["score"],

        reverse=True
    )


    combined_results = [

        item["doc"]

        for item in ranked_results
    ]


    return combined_results[:k]