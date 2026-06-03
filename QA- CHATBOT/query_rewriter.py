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

    print("\nQUERY REWRITER EXECUTED")
    question_lower = question.lower()


    # =========================
    # SECTION-AWARE REWRITING
    # =========================

    for section in document_sections:

        if section in question_lower:

            question = (

                f"{question} "
                f"{section} section "
                f"{section} details "
                f"{section} information"
            )

            break



    # =========================
    # QUERY REWRITING PROMPT
    # =========================

    prompt = f"""
    You are a conversational search query rewriter for a RAG system.

    Your job is to convert the user's question into a standalone retrieval query.

    RULES:

    1. Use PREVIOUS CONVERSATION to understand the user's intent.

    2. Resolve references such as:
    - it
    - they
    - them
    - this
    - that
    - these
    - those

    3. Replace pronouns with the actual entity from the conversation.

    4. Preserve important names, technologies, projects, people, and document topics.

    5. If the question is already clear, keep it unchanged.

    6. Do NOT answer the question.

    7. Return ONLY the rewritten search query.

    8. Keep the query concise and retrieval-friendly.

    PREVIOUS CONVERSATION:
    {conversation_context}

    USER QUESTION:
    {question}

    STANDALONE SEARCH QUERY:
    """


    # =========================
    # LLM REWRITING
    # =========================

    response = llm.invoke(prompt)

    rewritten_query = response.content.strip()

    print("\n" + "=" * 60)
    print("ORIGINAL QUESTION:")
    print(question)

    print("\n" + "="*50)
    print("REWRITTEN QUERY ONLY")
    print(rewritten_query)
    print("="*50)

    print("=" * 60)



    # =========================
    # FINAL QUERY
    # =========================

    return rewritten_query