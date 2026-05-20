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

llm = ChatGroq(

    groq_api_key=os.getenv("GROQ_API_KEY"),

    model_name="llama-3.1-8b-instant",

    temperature=0
)


prompt_template = PromptTemplate(

    input_variables=["context", "question", "conversation_context"],

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

CONTEXT:
{context}

PREVIOUS CONVERSATION:
{conversation_context}

QUESTION:
{question}

ANSWER:

"""
)

#function to generate answer from LLM using the prompt template, returns the generated answer
def generate_answer(question, reranked_chunks,conversation_context):

    context = "\n\n".join(

        [
            chunk.page_content[:700]

            for chunk, score in reranked_chunks[:2]
        ]
    )

    final_prompt = prompt_template.format(

        context=context,

        question=question,

        conversation_context=conversation_context
    )

    response = llm.invoke(final_prompt)

    answer = response.content

    sources = []

    for chunk, score in reranked_chunks[:2]:

        metadata = chunk.metadata

        source_text = (

            f"Page {metadata['page']} | "

            f"Section: {metadata['section']}"
        )

        if source_text not in sources:

            sources.append(source_text)

    citations = "\n".join(sources)

    final_answer = f"""

    {answer}

    SOURCES:
    {citations}
    """

    return final_answer