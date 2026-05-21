#GOAL OF THIS FILE

#This file should ONLY:

#| Responsibility              | Why                             |
#| --------------------------- | ------------------------------- |
#| Create grounded prompts     | reduce hallucinations           |
#| Send context to LLM         | contextual answer generation    |
#| Generate final answers      | conversational document QA      |
#| Restrict answers to context | industrial RAG reliability      |

#WHAT THIS FILE WILL DO

#It will:

#receive user query
#receive reranked chunks
#combine chunks into context
#build prompt
#send to LLM
#return grounded answer


from langchain_groq import ChatGroq

from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv

import os


load_dotenv()


# =========================
# LLM
# =========================

llm = ChatGroq(

    groq_api_key=os.getenv("GROQ_API_KEY"),

    model_name="llama-3.1-8b-instant",

    temperature=0
)


# =========================
# PROMPT TEMPLATE
# =========================

prompt_template = PromptTemplate(

    input_variables=[

        "context",

        "question",

        "conversation_context"
    ],

    template="""

You are an intelligent PDF Question Answering System.

Answer the user's question ONLY using the provided context.

IMPORTANT RULES:

1. Do NOT use outside knowledge.
2. Do NOT make assumptions.
3. Combine information from multiple retrieved chunks if needed.
4. Give complete and meaningful answers.
5. Explain clearly in well-structured sentences.
6. If information is unavailable, say:
   "The document does not contain enough information."


IMPORTANT ANSWERING RULES:

1. Do NOT omit names, entities, numbers, or list items from the retrieved context.
2. Preserve complete factual information from the context.
3. If multiple names or members are present, include ALL of them.
4. Do NOT summarize entity lists.
5. Answer ONLY from the retrieved context.
6. Do NOT ignore relevant retrieved content.
7. If the context contains a complete list, preserve the entire list.
8. Do NOT shorten factual information.


CONTEXT:
{context}


PREVIOUS CONVERSATION:
{conversation_context}


QUESTION:
{question}


ANSWER:

"""
)


# =========================
# GENERATE ANSWER FUNCTION
# =========================

def generate_answer(

    question,

    reranked_chunks,

    conversation_context
):


    # =========================
    # CREATE CONTEXT
    # =========================

    context = "\n\n".join(

        [

            chunk.page_content

            for chunk, score in reranked_chunks[:5]
        ]
    )


    # =========================
    # BUILD FINAL PROMPT
    # =========================

    final_prompt = prompt_template.format(

        context=context,

        question=question,

        conversation_context=conversation_context
    )


    # =========================
    # LLM GENERATION
    # =========================

    response = llm.invoke(final_prompt)

    answer = response.content


    # =========================
    # SOURCE CITATIONS
    # =========================

    sources = []


    for chunk, score in reranked_chunks[:5]:

        metadata = chunk.metadata

        source_text = (

            f"Page {metadata['page']} | "

            f"Section: {metadata['section']}"
        )


        if source_text not in sources:

            sources.append(source_text)


    citations = "\n".join(sources)


    # =========================
    # FINAL ANSWER
    # =========================

    final_answer = f"""

{answer}

SOURCES:
{citations}
"""


    return final_answer