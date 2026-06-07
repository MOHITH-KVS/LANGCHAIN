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

from config import (
    MAX_CHUNK_LENGTH,
    MIN_CHUNK_LENGTH
)

# =========================
# CHUNK SETTINGS
# =========================

CHUNK_OVERLAP = 120


# =========================
# HEADING SCORE DETECTION
# =========================

def detect_heading_score(line):

    score = 0

    line = clean_line(line)

    # Reject GPA / percentage style lines

    if re.search(r"\d+\.\d+", line):
        return -99
    
    if re.search(r"cgpa|gpa|percentage|%", line.lower()):
        return -99

    if not line:

        return 0


    # REJECT CODE LINES
    code_patterns = [
        r"^\s*(def |class |import |from |return )",
        r"=\s*(Column|List|Optional|None|True|False|\[)",
        r":\s*(str|int|float|bool|list|dict|Optional|List)",
        r"https?://",
        r"nullable\s*=",
        r"primary_key\s*=",
        r"[a-z]_[a-z]",
    ]
    for pattern in code_patterns:
        if re.search(pattern, line):
            return -99

    

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
    # SINGLE WORD HEADING BOOST
    # =========================

    if word_count == 1:

        score += 3

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

    if symbol_count >= 4:

        score -= 3


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

    return score >= 4



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

        # =========================
        # SKIP VERY SMALL FRAGMENTS
        # =========================

        if len(sentence) < 40:

            current_chunk += " " + sentence

            continue


        if not sentence:

            continue


        # =========================
        # APPEND TO CURRENT CHUNK
        # =========================

        if (
            len(current_chunk) + len(sentence)
            <= MAX_CHUNK_LENGTH
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

    prev_line_was_bullet = False
    prev_line_was_heading = False

    for line in lines:

        line = clean_line(line)

        if not line:
            prev_line_was_bullet = False
            continue

        is_bullet = line.startswith(("-", "•", "*"))

        detected_as_heading = (
            is_heading(line)
            and not prev_line_was_bullet
            and not prev_line_was_heading
        )

        if detected_as_heading:

            heading_candidate = re.sub(
                r"[^a-zA-Z0-9\s]",
                "",
                line.lower()
            ).strip()

            invalid_heading = (
                heading_candidate.startswith("-")
                or heading_candidate.startswith("•")
                or len(heading_candidate) < 3
                or heading_candidate.count(":") > 1
                or heading_candidate.count("(") > 1
                or heading_candidate in [":", "-", "--"]
                or "_" in heading_candidate
                or re.search(r"\b(none|true|false|null|str|int|float|bool|list|dict|optional)\b", heading_candidate)
            )

            if not invalid_heading:

                if current_content:
                    semantic_blocks.append({
                        "section": current_heading,
                        "content": "\n".join(current_content)
                    })
                    current_content = []

                current_heading = heading_candidate
                prev_line_was_bullet = False
                prev_line_was_heading = True
                continue

        current_content.append(line)
        prev_line_was_bullet = is_bullet
        prev_line_was_heading = False


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


            if len(chunk_text.strip()) < 20:

                print("\nSHORT CHUNK REMOVED")
                print("SECTION:", section_name)
                print("LENGTH:", len(chunk_text))
                print(chunk_text)
                print("=" * 50)


                continue

                


            # =========================
            # SECTION-AWARE METADATA
            # =========================

            updated_metadata = metadata.copy()

            updated_metadata["section"] = section_name


            enriched_chunk = f"{section_name.upper()}\n\n{chunk_text}"

            chunk_data = {

                "content": enriched_chunk,

                "metadata": updated_metadata
            }


            chunks.append(chunk_data)


    return chunks