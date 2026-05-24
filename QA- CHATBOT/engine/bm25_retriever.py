from rank_bm25 import BM25Okapi


# =========================
# TOKENIZE DOCUMENTS
# =========================

def tokenize(text):

    return text.lower().split()


# =========================
# CREATE BM25 INDEX
# =========================

def create_bm25_index(chunks):

    tokenized_chunks = [

        tokenize(chunk["text"])

        for chunk in chunks
    ]


    bm25 = BM25Okapi(tokenized_chunks)

    return bm25


# =========================
# BM25 SEARCH
# =========================

def bm25_search(

    bm25,

    chunks,

    query,

    top_k=5
):

    tokenized_query = tokenize(query)


    scores = bm25.get_scores(

        tokenized_query
    )


    scored_chunks = list(

        zip(chunks, scores)
    )


    scored_chunks = sorted(

        scored_chunks,

        key=lambda x: x[1],

        reverse=True
    )


    return scored_chunks[:top_k]