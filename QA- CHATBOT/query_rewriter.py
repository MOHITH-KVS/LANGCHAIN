# GOAL OF THIS FILE

# This file should ONLY:

#| Responsibility              | Why                              |
#| --------------------------- | -------------------------------- |
#| Rewrite user queries        | improve retrieval quality        |
#| Clarify vague questions     | stronger semantic retrieval      |
#| Add conversational context  | better follow-up understanding   |
#| Optimize retrieval queries  | improve RAG accuracy             |


from langchain_groq import ChatGroq

from dotenv import load_dotenv

import os

load_dotenv()


# =========================
# LLM
# =========================

llm = ChatGroq(

    groq_api_key=os.getenv("GROQ_API_KEY"),

    model_name="llama-3.3-70b-versatile",

    temperature=0
)


# =========================
# DOCUMENT SECTION KEYWORDS
# =========================

document_sections = [

    "abstract",

    "introduction",

    "methodology",

    "architecture",

    "conclusion",

    "technologies",

    "results",

    "implementation",

    "objectives"
]


# =========================
# QUERY REWRITING FUNCTION
# =========================

def rewrite_query(

    question,

    conversation_context
):

    question_lower = question.lower()


    # =========================
    # SECTION-AWARE REWRITING
    # =========================

    # Convert short section queries
    # into retrieval-friendly queries

    for section in document_sections:

        if section in question_lower:

            return f"Explain the {section} section of the document"


    # =========================
    # QUERY REWRITING PROMPT
    # =========================

    prompt = f"""

You are a query rewriting system for PDF document retrieval.

Your ONLY job is to improve retrieval quality for the uploaded document.

IMPORTANT RULES:

1. NEVER change the meaning of the user's question.
2. NEVER convert document section names into general concepts.
3. Preserve important keywords exactly.
4. Preserve names, headings, and technical terms.
5. Rewrite ONLY if the question is vague.
6. Keep rewritten query concise.
7. Keep rewritten query closely related to the document.
8. Do NOT answer the question.
9. Output ONLY the rewritten query.
10. If the user mentions a document section like abstract, conclusion, methodology, introduction, architecture, etc., preserve the exact section name clearly in the rewritten query.
11. Expand short queries into retrieval-friendly document-oriented questions.

PREVIOUS CONVERSATION:
{conversation_context}

USER QUESTION:
{question}

REWRITTEN QUERY:

"""


    # =========================
    # LLM REWRITING
    # =========================

    response = llm.invoke(prompt)

    rewritten_query = response.content.strip()


    # =========================
    # SAFETY CHECK
    # =========================

    important_terms = [

        "gvp-maaa",

        "abstract",

        "methodology",

        "architecture",

        "conclusion",

        "technologies"
    ]

    original_lower = question.lower()

    rewritten_lower = rewritten_query.lower()


    # =========================
    # FALLBACK SAFETY
    # =========================

    # If important terms disappear,
    # fallback to original question.

    for term in important_terms:

        if term in original_lower and term not in rewritten_lower:

            return question


    # =========================
    # FINAL QUERY
    # =========================

    return rewritten_query