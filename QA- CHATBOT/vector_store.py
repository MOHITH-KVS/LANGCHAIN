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

import re

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

def tokenize(text):

    text = text.lower()

    text = re.sub(r"[^a-z0-9\s]", " ", text)

    return text.split()


# =========================
# CREATE BM25 INDEX
# =========================

def create_bm25_index(all_chunks):

    tokenized_chunks = [

        tokenize(chunk["content"])

        for chunk in all_chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)

    return bm25



# =========================
# HYBRID RETRIEVAL
# =========================

def hybrid_retrieve(

    vector_store,

    bm25,

    all_chunks,

    query,

    source_filter=None,

    k=10
):




    # =========================
    # INITIAL FILTER
    # =========================

    filtered_chunks = all_chunks

    print("\nALL RESUME CHUNKS")

    for chunk in filtered_chunks:

        if chunk["metadata"]["source"] == "K.V.S MOHITH RESUME FINAL.pdf":

            print(
                chunk["chunk_id"],
                chunk["metadata"]["source"]
            )


    # =========================
    # SOURCE FILTER
    # =========================

    if source_filter:

        # =========================
        # HANDLE SINGLE STRING
        # =========================

        if isinstance(source_filter, str):

            source_filter = [source_filter]


        filtered_chunks = [

            chunk

            for chunk in filtered_chunks

            if chunk["metadata"]["source"]

            in

            source_filter
        ]



    print("\nSOURCE FILTER:", source_filter)
    print("FILTERED CHUNKS:", len(filtered_chunks))

    resume_chunks = [
        c for c in filtered_chunks
        if c["metadata"]["source"] == "K.V.S MOHITH RESUME FINAL.pdf"
    ]

    print("RESUME CHUNKS:", len(resume_chunks))


    # =========================
    # FAISS RETRIEVAL WITH SCORES
    # =========================

    all_semantic_results = vector_store.similarity_search_with_score(

        query,

        k=150
    )


    print("\nLOOKING FOR CHUNK 570")

    for doc, score in all_semantic_results:

        if doc.metadata.get("chunk_id") == 570:

            print("\nFOUND CHUNK 570")
            print("SCORE:", score)
            print(doc.page_content)

    print("\nSEMANTIC RESULTS")
    for doc, score in all_semantic_results:
        print(
            doc.metadata.get("source"),
            doc.metadata.get("chunk_id"),
            score
        )

    if doc.metadata.get("source") == "K.V.S MOHITH RESUME FINAL.pdf":

        print(
            "RESUME",
            doc.metadata.get("chunk_id"),
            score
        )

    semantic_results = []


    for doc, score in all_semantic_results:

        source_match = True


        # =========================
        # SOURCE FILTER
        # =========================

        if source_filter:

            source_match = (

                doc.metadata.get("source")

                in

                source_filter
            )




        # =========================
        # IMPORTANT:
        # LOWER FAISS DISTANCE = BETTER
        # CONVERT TO SIMILARITY
        # =========================

        similarity_score = 1 / (1 + score)


        # =========================
        # KEEP FILTERED RESULTS
        # =========================

        if (

            source_match

            and

            similarity_score >= 0.10
        ):

            semantic_results.append(

                (doc, similarity_score)
            )


    # =========================
    # SORT BY SIMILARITY
    # =========================

    semantic_results = sorted(

        semantic_results,

        key=lambda x: x[1],

        reverse=True
    )

    print("\nRESUME CHUNKS IN SEMANTIC RESULTS")

    for doc, score in semantic_results:

        if doc.metadata.get("source") == "K.V.S MOHITH RESUME FINAL.pdf":

            print(
                doc.metadata.get("chunk_id"),
                score
            )


    # =========================
    # EXTRACT FILTERED DOCUMENTS
    # =========================

    semantic_docs = [

        doc

        for doc, score in semantic_results[:100]
    ]
        


    # =========================
    # BM25 RETRIEVAL
    # =========================

    bm25_query = tokenize(query)


    filtered_texts = [

        tokenize(chunk["content"])

        for chunk in filtered_chunks
    ]


    if len(filtered_texts) == 0:

        return []


    filtered_bm25 = BM25Okapi(filtered_texts)


    bm25_scores = filtered_bm25.get_scores(bm25_query)


    bm25_ranked = sorted(

        zip(filtered_chunks, bm25_scores),

        key=lambda x: x[1],

        reverse=True
    )


    bm25_docs = []


    for chunk, score in bm25_ranked[:25]:

        bm25_docs.append(

            Document(

                page_content=chunk["content"],

                metadata={

                    **chunk["metadata"],

                    "chunk_id": chunk["chunk_id"]
                }
            )
        )


    print("\nRESUME CHUNKS IN BM25 RESULTS")

    for chunk, score in bm25_ranked[:50]:

        if chunk["metadata"]["source"] == "K.V.S MOHITH RESUME FINAL.pdf":

            print(
                chunk["chunk_id"],
                score
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

        (

            item["doc"],

            item["score"]
        )

        for item in ranked_results
    ]



    print("\nTOP HYBRID RESULTS")

    for rank, (doc, score) in enumerate(combined_results[:15], start=1):

        print("\nRANK:", rank)
        print("CHUNK ID:", doc.metadata.get("chunk_id"))
        print("SOURCE:", doc.metadata.get("source"))
        print("RRF SCORE:", score)
        print(doc.page_content[:200])

        print("\nCHECKING FOR RESUME CHUNKS")

    for doc, score in combined_results:

        if doc.metadata.get("source") == "K.V.S MOHITH RESUME FINAL.pdf":

            print("\nRESUME CHUNK FOUND")
            print("CHUNK ID:", doc.metadata.get("chunk_id"))
            print("RANK SCORE:", score)
            print(doc.page_content[:200])


    print("\n" + "="*80)
    print("ALL RESUME CHUNKS RETURNED BY RETRIEVAL")
    print("="*80)

    for doc, score in combined_results:

        if doc.metadata.get("source") == "K.V.S MOHITH RESUME FINAL.pdf":

            print("\nCHUNK ID:", doc.metadata.get("chunk_id"))
            print("SCORE:", score)

            print("CONTENT:")
            print(doc.page_content[:300])

        print("-"*80)


    return combined_results[:k]