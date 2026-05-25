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

You are an industrial-grade Retrieval-Augmented Generation (RAG) assistant.

Your responsibility is to answer the user's question ONLY using the retrieved context.

==================================================
STRICT RULES
==================================================

1. NEVER use outside knowledge.

2. NEVER hallucinate or invent facts.

3. NEVER assume missing information.

4. If the answer is not available in the context, respond ONLY with:
"The document does not contain enough information."

5. Preserve:
   - names
   - numbers
   - technical terms
   - skills
   - technologies
   - entities
   exactly as written.

6. Keep answers:
   - professional
   - concise
   - well-structured
   - human-readable

7. Use bullet points whenever appropriate.

8. If the context contains lists, skills, tools, technologies, or features:
   organize them clearly using bullets.

9. Do NOT mention:
   - chunks
   - retrieval
   - embeddings
   - vector databases
   - provided context

10. Synthesize information naturally instead of copying raw chunk text.


==================================================
RETRIEVED CONTEXT
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
# CONTEXT COMPRESSION
# =========================

def compress_context(question, chunks):

    compressed_chunks = []

    query_words = set(

        re.findall(r'\w+', question.lower())
    )

    for item in chunks:

        if isinstance(item, tuple):

            chunk = item[0]

        else:

            chunk = item

        text = clean_retrieved_text(

            chunk.page_content
        )

        sentences = re.split(

            r'''
            (?<=[.!?])\s+            # sentence endings
            |
            \n{2,}                   # paragraph breaks
            |
            (?<=:)\s+                # section labels
            |
            (?<=;)\s+                # semicolon-separated structures
            |
            (?<=•)\s*                # bullet points
            |
            (?<=-)\s+(?=[A-Z0-9])    # list items
            ''',

            text,

            flags=re.VERBOSE
        )

        relevant_sentences = []

        for sentence in sentences:

            sentence_words = set(

                sentence.lower().split()
            )

            overlap = query_words.intersection(

                sentence_words
            )

            if len(overlap) >= 3:

                relevant_sentences.append(sentence)

        # fallback
        if not relevant_sentences:

            relevant_sentences = sentences[:2]

        compressed_text = " ".join(

        relevant_sentences
    )


        # =========================
        # COMPRESSION DEBUG LOGS
        # =========================

        #print("\n" + "="*50)

        #print("ORIGINAL CHUNK:\n")

        #print(text[:700])

        #print("\n" + "-"*50)

        #print("COMPRESSED CHUNK:\n")

        #print(compressed_text[:700])

        #print("\n" + "-"*50)

        #print(

        #    f"ORIGINAL LENGTH: {len(text)} characters"
        #)

        #print(

        #    f"COMPRESSED LENGTH: {len(compressed_text)} characters"
        #)

        #reduction = (

        #    len(text) - len(compressed_text)
        #)

        #print(

        #    f"REDUCED BY: {reduction} characters"
        #)

        #print("="*50 + "\n")


        chunk.page_content = compressed_text

        compressed_chunks.append(chunk)

    return compressed_chunks



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

        reranked_chunks[:5]
    )


    # =========================
    # CONTEXT COMPRESSION
    # =========================

    compressed_chunks = compress_context(

        question,

        unique_chunks
    )


    # =========================
    # CREATE GROUNDED CONTEXT
    # =========================

    context = build_grounded_context(

        compressed_chunks
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