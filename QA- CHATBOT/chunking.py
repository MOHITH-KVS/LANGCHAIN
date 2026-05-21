#GOAL OF THIS FILE

#This file should ONLY:

#| Responsibility              | Why                              |
#| --------------------------- | -------------------------------- |
#| Split cleaned text          | semantic document chunking       |
#| Preserve contextual meaning | stronger retrieval quality       |
#| Create meaningful chunks    | improve embedding quality        |
#| Maintain chunk metadata     | contextual retrieval             |
#| Generate chunk structures   | industrial RAG pipeline          |


from nltk.tokenize import sent_tokenize


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
    # SPLIT INTO SENTENCES
    # =========================

    sentences = sent_tokenize(text)


    chunks = []

    current_chunk = ""

    chunk_id = 0


    # =========================
    # SEMANTIC GROUPING
    # =========================

    for sentence in sentences:

        # Add sentence if chunk size is safe

        if len(current_chunk) + len(sentence) < MAX_CHUNK_LENGTH:

            current_chunk += " " + sentence


        # Otherwise create chunk

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

    if current_chunk.strip():

        chunk_data = {

            "content": current_chunk.strip(),

            "metadata": metadata,

            "chunk_id": chunk_id
        }

        chunks.append(chunk_data)


    return chunks