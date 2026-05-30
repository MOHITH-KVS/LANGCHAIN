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



    # =========================
    # QUERY REWRITING PROMPT
    # =========================

    prompt = f"""
You are a search query optimizer for a document retrieval system.

Your ONLY job is to rewrite the user's question into a better search query.

RULES:
1. Keep ALL important keywords from the original question
2. Do NOT remove technical terms, names, or section names
3. Do NOT make the query generic - keep it specific
4. If the user asks about a section like abstract, introduction, methodology - keep that word AND expand with related terms
5. Do NOT answer the question
6. Return ONLY the rewritten query, nothing else
7. Keep it under 20 words

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
    # FINAL QUERY
    # =========================

    return rewritten_query