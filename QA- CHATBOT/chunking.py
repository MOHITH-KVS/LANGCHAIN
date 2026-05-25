# GOAL OF THIS FILE

# This file should ONLY:

# | Responsibility              | Why                              |
# | --------------------------- | -------------------------------- |
# | Split cleaned text          | semantic document chunking       |
# | Preserve contextual meaning | stronger retrieval quality       |
# | Create meaningful chunks    | improve embedding quality        |
# | Maintain chunk metadata     | contextual retrieval             |
# | Generate chunk structures   | industrial RAG pipeline          |


import re


# =========================
# CHUNK SETTINGS
# =========================

MAX_CHUNK_LENGTH = 1000

MIN_CHUNK_LENGTH = 300


# =========================
# CREATE CHUNKS FUNCTION
# =========================

def create_chunks(

    text,

    metadata
):


    # =========================
    # SMART STRUCTURE SPLITTING
    # =========================
    # Better for:
    # - resumes
    # - industrial docs
    # - reports
    # - OCR extracted PDFs
    # =========================

    sentences = re.split(

        r'\n+|(?<=:)|(?<=\.)',

        text
    )


    chunks = []

    current_chunk = ""

    chunk_id = 0


    # =========================
    # SEMANTIC GROUPING
    # =========================

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:

            continue


        # =========================
        # ADD TO CURRENT CHUNK
        # =========================

        if (

            len(current_chunk) + len(sentence)

            < MAX_CHUNK_LENGTH
        ):

            current_chunk += " " + sentence


        # =========================
        # CREATE NEW CHUNK
        # =========================

        else:

            # Avoid tiny chunks

            if len(current_chunk.strip()) >= MIN_CHUNK_LENGTH:

                chunk_data = {

                    "content": current_chunk.strip(),

                    "metadata": metadata,

                    "chunk_id": chunk_id
                }

                chunks.append(chunk_data)

                chunk_id += 1


            # Start new chunk

            current_chunk = sentence


    # =========================
    # FINAL CHUNK
    # =========================

    if len(current_chunk.strip()) >= MIN_CHUNK_LENGTH:

        chunk_data = {

            "content": current_chunk.strip(),

            "metadata": metadata,

            "chunk_id": chunk_id
        }

        chunks.append(chunk_data)


    return chunks