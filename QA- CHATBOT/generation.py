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

import re

from config import (
    TEMPERATURE,
    GENERATION_MODEL
)


load_dotenv()


# =========================
# LLM
# =========================

llm = ChatGroq(

    groq_api_key=os.getenv("GROQ_API_KEY"),

    model_name=GENERATION_MODEL,

    temperature=TEMPERATURE,

    max_tokens=4096
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
    You are a document extraction assistant.

    Your ONLY job is to extract and present information from the RETRIEVED CONTEXT below.

    STRICT RULES:
    1. Use ONLY the RETRIEVED CONTEXT. Never use outside knowledge.
    2. NEVER skip, summarize, or omit any item from the context.
    3. If the question asks for projects, list ONLY items explicitly listed under project sections. Do NOT include internship experiences, hackathon participations, or leadership roles as projects unless the question specifically asks for them.
    4. If the question asks for skills, list EVERY skill found across ALL chunks.
    5. If the question asks for certifications, list EVERY certification found across ALL chunks.
    6. Scan EVERY chunk completely before writing your answer.
    7. Do NOT say "and more" or "etc." - always list everything explicitly.
    8. If the answer is not in the context, say only: "The document does not contain this information."
    9. Do NOT mention chunks, retrieval, embeddings, or context in your answer.
    10. Format your answer clearly with bullet points when listing items.

    RETRIEVED CONTEXT:
    {context}

    PREVIOUS CONVERSATION:
    {conversation_context}

    QUESTION:
    {question}

    ANSWER (include ALL items found in context, do not skip any):
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
# CONTEXT CLEANER
# =========================

def clean_retrieved_text(text):

    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove repeated bullets/symbols
    text = re.sub(r'[•●▪■]+', '•', text)

    # Fix broken spacing around punctuation
    text = re.sub(r'\s+([,:.;])', r'\1', text)

    # Normalize skill separators
    text = re.sub(r'\s*:\s*', ': ', text)

    # Remove duplicate consecutive words
    text = re.sub(
        r'\b(\w+)( \1\b)+',
        r'\1',
        text,
        flags=re.IGNORECASE
    )

    return text.strip()





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

        return {

            "answer": "The document does not contain enough information.",

            "sources": []
        }


    # =========================
    # REMOVE DUPLICATE CHUNKS
    # =========================

    def remove_duplicate_chunks(chunks):

        unique_chunks = []

        seen_contents = set()


        for item in chunks:

            if isinstance(item, tuple):

                chunk = item[0]

            else:

                chunk = item


            # =========================
            # NORMALIZE TEXT
            # =========================

            normalized_text = re.sub(

                r'\s+',

                ' ',

                chunk.page_content.lower()
            ).strip()


            # =========================
            # SKIP DUPLICATES
            # =========================

            if normalized_text in seen_contents:

                continue


            seen_contents.add(normalized_text)

            unique_chunks.append(chunk)


        return unique_chunks


    # =========================
    # REMOVE DUPLICATE CHUNKS
    # =========================

    unique_chunks = remove_duplicate_chunks(

        reranked_chunks[:15]
    )

    if len(unique_chunks) == 0:

        return {

            "answer": "The document does not contain enough information.",

            "sources": []
        }


    # =========================
    # CREATE GROUNDED CONTEXT
    # =========================

    context = build_grounded_context(

        unique_chunks
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

    try:

        response = llm.invoke(final_prompt)

        answer = (

            response.content
            .replace("\\n", "\n")
            .strip()
        )

        # Remove trailing comma or comma+space at end of answer
        if answer.endswith(","):
            answer = answer[:-1].strip()

        if answer.endswith(",\n"):
            answer = answer[:-2].strip()

    except Exception as e:

        return {

            "answer": f"LLM generation failed: {str(e)}",

            "sources": []
        }


    # =========================
    # SOURCE CITATIONS
    # =========================

    sources = []

    for item in reranked_chunks[:10]:

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