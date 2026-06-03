# GOAL OF THIS FILE

# This file should ONLY:

# | Responsibility     | Why                       |
# | ------------------ | ------------------------- |
# | Load PDFs          | extract text              |
# | Clean text         | remove noise              |
# | Detect sections    | preserve structure        |
# | Add metadata       | smarter retrieval         |
# | Filter noisy pages | improve retrieval quality |


from langchain_community.document_loaders import PyMuPDFLoader

import re


# =========================================================
# LOAD PDF
# =========================================================

def load_pdf(file_path):

    loader = PyMuPDFLoader(file_path)

    documents = loader.load()

    print("\nRAW PAGE 1:")
    print(documents[0].page_content)

    print("\nFIRST PAGE RAW TEXT:")
    

    print("\n" + "=" * 80)
    print("PDF EXTRACTION DEBUG")
    print("=" * 80)

    for i, doc in enumerate(documents):

        print(f"\nPAGE {i + 1}")

        print("-" * 50)

        print(
            doc.page_content[:2000]
            .encode("ascii", errors="ignore")
            .decode()
        )

    print("\n" + "=" * 80)

    return documents


# =========================================================
# CLEAN TEXT
# =========================================================

def clean_text(text):

    # =====================================================
    # BASIC NORMALIZATION
    # =====================================================

    text = text.replace("\r", "\n")

    text = re.sub(r'\t+', ' ', text)

    text = re.sub(r' +', ' ', text)


    # =====================================================
    # REMOVE WEIRD BULLETS / SYMBOLS
    # =====================================================

    text = re.sub(r'[•●▪■]', '-', text)


    # =====================================================
    # REMOVE PAGE NUMBER ARTIFACTS
    # =====================================================

    text = re.sub(

        r'\bPage\s+\d+\b',

        '',

        text,

        flags=re.IGNORECASE
    )

    text = re.sub(r'\b\d+\s*/\s*\d+\b', '', text)


    

    # =====================================================
    # CLEAN MULTIPLE NEWLINES
    # =====================================================

    text = re.sub(r'\n{3,}', '\n\n', text)


    # =====================================================
    # FINAL CLEAN
    # =====================================================

    text = text.strip()

    return text


# =========================================================
# DETECT SECTION
# =========================================================

def detect_section(text):

    text_lower = text.lower()


    section_keywords = {

        "skills": [

            "skills",

            "technical skills",

            "soft skills"
        ],

        "education": [

            "education"
        ],

        "experience": [

            "experience",

            "internships"
        ],

        "projects": [

            "projects"
        ],

        "certifications": [
            "certifications",
            "certificates",
            "certificate",
            "workshops",
            "courses"
        ],

        "achievements": [

            "achievements"
        ],

        "languages": [

            "languages"
        ],

        "hobbies": [

            "hobbies"
        ],

        "declaration": [

            "declaration"
        ]
    }


    for section, keywords in section_keywords.items():

        for keyword in keywords:

            if re.search(

                rf'\b{re.escape(keyword)}\b',

                text_lower
            ):

                return section


    return "general"


# =========================================================
# SPLIT INTO SECTIONS
# =========================================================

def split_into_sections(text):

    lines = text.split("\n")

    sections = []

    current_section = "general"

    current_content = []


    for line in lines:

        clean_line = line.strip()


        if not clean_line:

            continue


        # =====================================================
        # STRUCTURAL HEADING SCORE
        # =====================================================

        words = clean_line.split()

        word_count = len(words)


        uppercase_ratio = sum(

            1 for c in clean_line if c.isupper()

        ) / max(len(clean_line), 1)


        title_case_ratio = sum(

            1 for word in words

            if word[:1].isupper()

        ) / max(len(words), 1)


        has_terminal_punctuation = clean_line.endswith(

            (".", ",", ";", "?")
        )


        starts_with_bullet = clean_line.startswith(

            ("-", "•", "*")
        )

        contains_many_symbols = len(

            re.findall(r"[:()\-]", clean_line)

        ) >= 2


        contains_numbers = bool(

            re.search(r"\d", clean_line)
        )


        # =====================================================
        # HEADING SCORE
        # =====================================================

        heading_score = 0


        # Short lines are likely headings

        if word_count <= 6:

            heading_score += 1


        # Uppercase headings

        if uppercase_ratio > 0.6:

            heading_score += 2


        # Title case headings

        if title_case_ratio > 0.85:

            heading_score += 1


        # Headings usually don't end with punctuation

        if not has_terminal_punctuation:

            heading_score += 1


        # Bullet lines are almost never headings

        if starts_with_bullet:

            heading_score -= 8


        # Symbol-heavy lines are usually list items

        if contains_many_symbols:

            heading_score -= 5


        # Number-heavy lines are usually content

        if contains_numbers:

            heading_score -= 1


        # Long lines are usually content

        if len(clean_line) > 80:

            heading_score -= 3


        # =====================================================
        # INDUSTRIAL HEADING DETECTION
        # =====================================================

        # Reject code lines before heading detection
        code_signals = (
            re.search(r"[=\[\]{}()<>]", clean_line) is not None
            or re.search(r"(def |class |import |return |const |export |function )", clean_line) is not None
            or re.search(r"https?://", clean_line) is not None
            or re.search(r"[a-z][A-Z]", clean_line) is not None
            or "_" in clean_line
            or clean_line.startswith("@")
        )

        is_heading = (
            not code_signals
            and word_count <= 6
            and not starts_with_bullet
            and not has_terminal_punctuation
            and not contains_many_symbols
            and not contains_numbers
            and len(clean_line) < 50
            and title_case_ratio > 0.6
        )


        # =====================================================
        # SAVE PREVIOUS SECTION
        # =====================================================

        if is_heading:

            if current_content:

                sections.append({

                    "section": current_section,

                    "content": "\n".join(current_content)
                })


            detected_section = detect_section(clean_line)

            if detected_section != "general":
                current_section = detected_section
            else:
                current_section = clean_line.lower().replace(":", "").strip()

            current_content = []


        else:

            current_content.append(clean_line)


    # =====================================================
    # FINAL SECTION
    # =====================================================

    if current_content:

        sections.append({

            "section": current_section,

            "content": "\n".join(current_content)
        })

    print("\n========== SECTION DEBUG ==========")

    for sec in sections:
        print("SECTION FOUND:", sec["section"])

        preview = (
            sec["content"][:150]
            .encode("ascii", errors="ignore")
            .decode()
        )

        print(preview)

    print("===================================")


    return sections


# =========================================================
# CREATE METADATA
# =========================================================

def create_metadata(

    file_path,

    page_num,

    section,

    document_type = "pdf"
):

    metadata = {

        "source": file_path,

        "page": page_num,

        "section": section,

        "document_type": document_type
    }

    return metadata


# =========================================================
# TESTING
# =========================================================

if __name__ == "__main__":

    docs = load_pdf(

        "GVP-MAAA DOCUMENTATION (1).pdf"
    )


    raw_text = docs[0].page_content


    cleaned_text = clean_text(raw_text)


    section = detect_section(cleaned_text)


    metadata = create_metadata(

        "GVP-MAAA DOCUMENTATION (1).pdf",

        1,

        section
    )


    split_sections = split_into_sections(

        cleaned_text
    )


    print("\n==============================")

    print("CLEANED TEXT")

    print("==============================\n")

    print(cleaned_text[:2000])


    print("\n==============================")

    print("SECTION DETECTED")

    print("==============================\n")

    print(section)


    print("\n==============================")

    print("METADATA")

    print("==============================\n")

    print(metadata)


    print("\n==============================")

    print("SECTIONS")

    print("==============================\n")

    for sec in split_sections:

        print(f"\nSECTION: {sec['section']}")

        print(sec["content"][:500])