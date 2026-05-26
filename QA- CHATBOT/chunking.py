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
# HEADING SCORE DETECTION
# =========================

def detect_heading_score(line):

    score = 0

    line = clean_line(line)

    if not line:

        return 0


    words = line.split()

    word_count = len(words)


    # =========================
    # SHORT LINES
    # =========================

    if word_count <= 6:

        score += 3


    # =========================
    # VERY SHORT HEADINGS
    # =========================

    if word_count <= 3:

        score += 2


    # =========================
    # NO SENTENCE ENDING
    # =========================

    if not line.endswith((".", "?", "!")):

        score += 2


    # =========================
    # TITLE CASE
    # =========================

    title_case_ratio = sum(

        1 for word in words

        if word[:1].isupper()

    ) / max(len(words), 1)


    if title_case_ratio >= 0.8:

        score += 2


    # =========================
    # ALL CAPS
    # =========================

    uppercase_ratio = sum(

        1 for c in line if c.isupper()

    ) / max(len(line), 1)


    if uppercase_ratio > 0.6:

        score += 3


    # =========================
    # NUMBERED HEADINGS
    # =========================

    if re.match(r"^\d+(\.\d+)*", line):

        score += 2


    # =========================
    # ROMAN HEADINGS
    # =========================

    if re.match(r"^[IVXLC]+\.", line):

        score += 2


    # =========================
    # BULLET LINES PENALTY
    # =========================

    if line.startswith(("-", "•", "*")):

        score -= 8


    # =========================
    # SYMBOL HEAVY PENALTY
    # =========================

    symbol_count = len(

        re.findall(r"[:()\[\],]", line)
    )

    if symbol_count >= 2:

        score -= 5


    # =========================
    # LONG LINE PENALTY
    # =========================

    if len(line) > 80:

        score -= 5


    # =========================
    # NUMBER DENSITY PENALTY
    # =========================

    number_count = len(

        re.findall(r"\d", line)
    )

    if number_count >= 5:

        score -= 3


    return score


# =========================
# FINAL HEADING DETECTION
# =========================

def is_heading(line):

    score = detect_heading_score(line)

    return score >= 6



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

    current_heading = "general"

    current_content = []


    # =========================
    # STRUCTURE-AWARE PARSING
    # =========================

    for line in lines:

        line = clean_line(line)


        # =========================
        # EMPTY LINE
        # =========================

        if not line:

            continue


        # =========================
        # NEW HEADING DETECTED
        # =========================

        if is_heading(line):

            heading_candidate = line.lower().strip()


            # Ignore noisy headings

            invalid_heading = (

                heading_candidate.startswith("-")
                or heading_candidate.startswith("•")
                or len(heading_candidate) < 3
                or heading_candidate.count(":") > 1
                or heading_candidate.count("(") > 1
                or heading_candidate in [":", "-", "--"]

            )


            if not invalid_heading:

                current_heading = heading_candidate


                if current_content:

                    semantic_blocks.append({

                        "section": current_heading,

                        "content": "\n".join(current_content)

                    })

                    current_content = []


                continue


        # =========================
        # NORMAL CONTENT
        # =========================

        current_content.append(line)


    # =========================
    # FINAL SECTION
    # =========================

    if current_content:

        semantic_blocks.append({

            "section": current_heading,

            "content": "\n".join(current_content)
        })


    # =========================
    # CREATE FINAL CHUNKS
    # =========================

    chunks = []


    for block_data in semantic_blocks:

        section_name = block_data["section"]

        block = block_data["content"]


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


            # =========================
            # SECTION-AWARE METADATA
            # =========================

            updated_metadata = metadata.copy()

            updated_metadata["section"] = section_name


            chunk_data = {

                "content": chunk_text,

                "metadata": updated_metadata
            }


            chunks.append(chunk_data)


    return chunks