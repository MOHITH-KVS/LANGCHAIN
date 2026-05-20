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

    model_name="llama3-8b-8192"
)


prompt_template = PromptTemplate(

    input_variables=["context", "question"],

    template="""

You are an intelligent PDF question-answering assistant.

Answer the question ONLY from the provided context.

If the answer is not available in the context, say:

"I could not find relevant information in the document."

DO NOT make up information.

Context:
{context}

Question:
{question}

Answer:

"""
)

#function to generate answer from LLM using the prompt template, returns the generated answer
def generate_answer(question, reranked_chunks):

    context = "\n\n".join(

        [chunk.page_content for chunk, score in reranked_chunks[:3]]
    )

    final_prompt = prompt_template.format(

        context=context,

        question=question
    )

    response = llm.invoke(final_prompt)

    return response.content