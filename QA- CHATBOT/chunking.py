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

MAX_CHUNK_LENGTH = 700

MIN_CHUNK_LENGTH = 50

CHUNK_OVERLAP = 120


# =========================
# HEADING DETECTION
# =========================

def is_heading(line):

    line = line.strip()


    # =========================
    # EMPTY CHECK
    # =========================

    if not line:

        return False


    # =========================
    # VERY LONG LINES
    # =========================

    if len(line) > 80:

        return False


    # =========================
    # ALL CAPS HEADINGS
    # =========================

    if line.isupper():

        return True


    # =========================
    # SHORT HEADING WITH :
    # =========================

    if (

        line.endswith(":")

        and

        len(line.split()) <= 6
    ):

        return True


    # =========================
    # COMMON HEADING PATTERNS
    # =========================

    heading_patterns = [

        r"^skills$",
        r"^technical skills$",
        r"^education$",
        r"^projects$",
        r"^experience$",
        r"^work experience$",
        r"^certifications$",
        r"^summary$",
        r"^profile$",
        r"^soft skills$",
        r"^achievements$",
        r"^internships$",
        r"^languages$",
        r"^hobbies$",
        r"^strengths$",
        r"^objective$",
        r"^career objective$",
        r"^professional summary$"
    ]


    line_lower = line.lower()


    for pattern in heading_patterns:

        if re.match(pattern, line_lower):

            return True


    return False


# =========================
# CLEAN TEXT
# =========================

def clean_line(line):

    line = line.strip()


    # Remove excessive spaces

    line = re.sub(

        r"\s+",

        " ",

        line
    )


    return line


# =========================
# SPLIT HUGE BLOCKS
# =========================

def split_large_block(block):

    sentences = re.split(

        r'(?<=[.!?])\s+',

        block
    )


    chunks = []

    current_chunk = ""


    for sentence in sentences:

        sentence = sentence.strip()


        if not sentence:

            continue


        # =========================
        # APPEND TO CURRENT CHUNK
        # =========================

        if (

            len(current_chunk) + len(sentence)

            < MAX_CHUNK_LENGTH
        ):

            current_chunk += " " + sentence


        # =========================
        # SAVE CURRENT CHUNK
        # =========================

        else:

            if len(current_chunk.strip()) >= MIN_CHUNK_LENGTH:

                chunks.append(

                    current_chunk.strip()
                )


            overlap_text = (

                current_chunk[-CHUNK_OVERLAP:]
            )

            current_chunk = (

                overlap_text + " " + sentence
            )


    # =========================
    # FINAL CHUNK
    # =========================

    if len(current_chunk.strip()) >= MIN_CHUNK_LENGTH:

        chunks.append(

            current_chunk.strip()
        )


    return chunks


# =========================
# CREATE CHUNKS
# =========================

def create_chunks(

    text,

    metadata
):


    # =========================
    # NORMALIZE TEXT
    # =========================

    text = text.replace(

        "\r",

        "\n"
    )


    lines = text.split("\n")


    # =========================
    # SEMANTIC BLOCKING
    # =========================

    semantic_blocks = []

    current_block = []


    for line in lines:

        line = clean_line(line)


        # =========================
        # EMPTY LINE
        # =========================

        if not line:

            continue


        # =========================
        # NEW HEADING
        # =========================

        if is_heading(line):


            # Save previous block

            if current_block:

                semantic_blocks.append(

                    "\n".join(current_block)
                )

                current_block = []


            current_block.append(line)

            continue


        # =========================
        # NORMAL CONTENT
        # =========================

        current_block.append(line)


    # =========================
    # FINAL BLOCK
    # =========================

    if current_block:

        semantic_blocks.append(

            "\n".join(current_block)
        )


    # =========================
    # CREATE FINAL CHUNKS
    # =========================

    chunks = []


    for block in semantic_blocks:

        block = block.strip()


        if not block:

            continue


        # =========================
        # SMALL BLOCK
        # =========================

        if len(block) <= MAX_CHUNK_LENGTH:

            final_chunks = [block]


        # =========================
        # LARGE BLOCK SPLITTING
        # =========================

        else:

            final_chunks = split_large_block(block)


        # =========================
        # CREATE CHUNK OBJECTS
        # =========================

        for chunk_text in final_chunks:


            chunk_text = chunk_text.strip()


            if len(chunk_text) < MIN_CHUNK_LENGTH:

                continue


            chunk_data = {

                "content": chunk_text,

                "metadata": metadata
            }


            chunks.append(chunk_data)


    return chunks