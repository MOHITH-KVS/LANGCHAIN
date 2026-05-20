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


llm = ChatGroq(

    groq_api_key=os.getenv("GROQ_API_KEY"),

    model_name="llama-3.3-70b-versatile"
)


def rewrite_query(

    question,

    conversation_context
):

    prompt = f"""

You are a query rewriting system for RAG retrieval.

Your job is to rewrite vague user questions into clear and retrieval-optimized queries.

IMPORTANT:
- Preserve original meaning.
- Use conversation context if necessary.
- Keep rewritten query concise.
- Do not answer the question.
- Only rewrite the query.

PREVIOUS CONVERSATION:
{conversation_context}

USER QUESTION:
{question}

REWRITTEN QUERY:

"""

    response = llm.invoke(prompt)

    return response.content.strip()