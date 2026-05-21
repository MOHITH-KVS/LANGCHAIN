#GOAL OF THIS FILE

#This file should ONLY:

#| Responsibility              | Why                              |
#| --------------------------- | -------------------------------- |
#| Test retrieval quality      | evaluate RAG performance         |
#| Validate retrieved sections | retrieval precision analysis     |
#| Measure retrieval accuracy  | industrial RAG evaluation        |
#| Automate RAG testing        | scalable quality assessment      |


from langchain_community.vectorstores import FAISS

from embeddings import embedding_model

from vector_store import (

    create_bm25_index,

    hybrid_retrieve
)

from reranker import rerank_chunks

import pickle


# =========================
# LOAD CHUNKS
# =========================

with open("chunks.pkl", "rb") as f:

    all_chunks = pickle.load(f)


# =========================
# LOAD FAISS
# =========================

vector_store = FAISS.load_local(

    "faiss_index",

    embedding_model,

    allow_dangerous_deserialization=True
)


# =========================
# BM25 INDEX
# =========================

bm25 = create_bm25_index(all_chunks)


# =========================
# TEST CASES
# =========================

test_cases = [

    {

        "question": "Explain abstract",

        "expected_section": "abstract"
    },

    {

        "question": "Explain methodology",

        "expected_section": "methodology"
    },

    {

        "question": "Explain conclusion",

        "expected_section": "conclusion"
    },

    {

        "question": "What technologies are used?",

        "expected_section": "technologies"
    },

    {

        "question": "Explain architecture",

        "expected_section": "architecture"
    }
]


# =========================
# EVALUATION
# =========================

total_tests = len(test_cases)

passed_tests = 0


print("\nRAG RETRIEVAL EVALUATION\n")


for test in test_cases:

    question = test["question"]

    expected_section = test["expected_section"]


    # =========================
    # RETRIEVAL
    # =========================

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


    # =========================
    # CHECK RETRIEVED SECTIONS
    # =========================

    retrieved_sections = []


    for chunk, score in reranked_chunks[:3]:

        section = chunk.metadata["section"]

        retrieved_sections.append(section)


    # =========================
    # PASS / FAIL
    # =========================

    if expected_section in retrieved_sections:

        result = "PASS"

        passed_tests += 1

    else:

        result = "FAIL"


    # =========================
    # PRINT RESULTS
    # =========================

    print(f"\nQUESTION: {question}")

    print(f"EXPECTED SECTION: {expected_section}")

    print(f"RETRIEVED SECTIONS: {retrieved_sections}")

    print(f"RESULT: {result}")

    print("\n" + "=" * 60)


# =========================
# FINAL SCORE
# =========================

accuracy = (passed_tests / total_tests) * 100


print("\nFINAL RETRIEVAL ACCURACY:")

print(f"{accuracy:.2f}%")