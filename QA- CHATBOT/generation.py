# GOAL OF THIS FILE

# This file should ONLY:

#| Responsibility              | Why                             |
#| --------------------------- | ------------------------------- |
#| Create grounded prompts     | reduce hallucinations           |
#| Send context to LLM         | contextual answer generation    |
#| Generate final answers      | conversational document QA      |
#| Restrict answers to context | industrial RAG reliability      |


# WHAT THIS FILE WILL DO

# It will:
# receive user query
# receive retrieved chunks
# combine chunks into context
# build grounded prompt
# synthesize contextual answer
# generate structured response
# return final grounded answer


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

You are an advanced Retrieval-Augmented Generation (RAG) assistant.

Your task is to answer the user's question ONLY using the provided retrieved context.

You must generate answers that are:
- grounded
- accurate
- contextual
- clearly explained
- professionally written

You are NOT allowed to:
- use outside knowledge
- hallucinate information
- invent facts
- assume missing information


==================================================
ANSWERING BEHAVIOR
==================================================

1. Carefully analyze ALL retrieved context before answering.

2. Synthesize information naturally instead of copying raw chunks directly.

3. Combine related information from multiple chunks into one coherent explanation.

4. Keep factual information accurate and complete.

5. Preserve:
   - names
   - numbers
   - entities
   - technical details
   - lists
   exactly as present in the context.

6. If the question asks:
   - "Explain"
   - "Describe"
   - "What is"
   then provide a clear explanatory response.

7. If multiple points exist in the context:
   organize them clearly.

8. Do NOT generate robotic chunk-like responses.

9. Do NOT mention:
   "according to the context"
   or
   "the provided context says"

10. If the answer is unavailable in the retrieved context, respond ONLY with:
"The document does not contain enough information."


==================================================
RESPONSE STYLE
==================================================

- Write naturally and professionally.
- Use complete sentences.
- Keep explanations concise but meaningful.
- Avoid unnecessary repetition.
- Prefer synthesized explanations over raw extraction.


==================================================
CONTEXT
==================================================

{context}


==================================================
PREVIOUS CONVERSATION
==================================================

{conversation_context}


==================================================
QUESTION
==================================================

{question}


==================================================
ANSWER
==================================================

"""
)


# =========================
# GROUNDED CONTEXT BUILDER
# =========================

def build_grounded_context(chunks):

    formatted_chunks = []

    for i, item in enumerate(chunks, 1):

        if isinstance(item, tuple):

            chunk = item[0]

        else:

            chunk = item

        source = chunk.metadata.get(
            "source",
            "Unknown"
        )

        page = chunk.metadata.get(
            "page",
            "N/A"
        )

        section = chunk.metadata.get(
            "section",
            "General"
        )

        formatted_chunk = f"""
[Chunk {i}]
Source: {source}
Page: {page}
Section: {section}

Content:
{chunk.page_content.strip()}
"""

        formatted_chunks.append(formatted_chunk)

    return "\n\n".join(formatted_chunks)



# =========================
# GENERATE ANSWER FUNCTION
# =========================

def generate_answer(

    question,

    reranked_chunks,

    conversation_context
):


    # =========================
    # EMPTY CONTEXT CHECK
    # =========================

    if not reranked_chunks:

        return "The document does not contain enough information."



    # =========================
    # CREATE GROUNDED CONTEXT
    # =========================

    context = build_grounded_context(

        reranked_chunks[:5]
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

    answer = (
        response.content
        .replace("\\n", "\n")
        .strip()
    )


    # =========================
    # SOURCE CITATIONS
    # =========================

    sources = []

    for item in reranked_chunks[:5]:

        if isinstance(item, tuple):

            chunk = item[0]

        else:

            chunk = item

        metadata = chunk.metadata

        source_name = metadata.get(
            "source",
            "Unknown Source"
        )

        page = metadata.get(
            "page",
            "Unknown"
        )

        section = metadata.get(
            "section",
            "general"
        )

        source_text = (
            f"{source_name} | "
            f"Page {page} | "
            f"Section: {section}"
        )

        if source_text not in sources:

            sources.append(source_text)

    # =========================
    # REMOVE SOURCES FOR UNKNOWN ANSWERS
    # =========================

    refusal_phrases = [

        "does not contain enough information",

        "not enough information",

        "information is not available",

        "cannot be found in the provided context"
    ]

    if any(

        phrase in answer.lower()

        for phrase in refusal_phrases
    ):

        sources = []

        

    # =========================
    # FINAL ANSWER
    # =========================

        return {

            "answer": answer,

            "sources": sources
        }