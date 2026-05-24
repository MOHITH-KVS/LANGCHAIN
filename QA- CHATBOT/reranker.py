#GOAL OF THIS FILE

#This file should ONLY:
#| Responsibility              | Why                         |
#| --------------------------- | --------------------------- |
#| Score retrieved chunks      | relevance estimation        |
#| Rerank chunks               | improve retrieval precision |
#| Select best chunks          | stronger context quality    |
#| Improve retrieval relevance | reduce noisy retrieval      |

from sentence_transformers import CrossEncoder

reranker_model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def rerank_chunks(query, retrieved_chunks):

    pairs = []

    documents = []


    # =========================
    # PREPARE CROSS-ENCODER INPUT
    # =========================

    for item in retrieved_chunks:

        if isinstance(item, tuple):

            chunk = item[0]

        else:

            chunk = item

        pairs.append(

            [query, chunk.page_content]
        )

        documents.append(chunk)


    # =========================
    # GET RERANK SCORES
    # =========================

    scores = reranker_model.predict(pairs)


    # =========================
    # ATTACH SCORES
    # =========================

    scored_chunks = [

        (doc, float(score))

        for doc, score in zip(documents, scores)
    ]


    # =========================
    # SORT BY RERANK SCORE
    # =========================

    ranked_chunks = sorted(

        scored_chunks,

        key=lambda x: x[1],

        reverse=True
    )


    return ranked_chunks