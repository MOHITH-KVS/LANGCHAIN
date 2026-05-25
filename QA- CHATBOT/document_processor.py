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
    # REMOVE BROKEN NUMBERING
    # =====================================================

    text = re.sub(r'\b\d+(\.\d+)+', '', text)


    # =====================================================
    # INDUSTRIAL HEADING PRESERVATION
    # =====================================================

    headings = [

        "CAREER OBJECTIVE",

        "SUMMARY",

        "SKILLS",

        "TECHNICAL SKILLS",

        "SOFT SKILLS",

        "ADDITIONAL SKILLS",

        "EDUCATION",

        "EXPERIENCE",

        "INTERNSHIPS",

        "PROJECTS",

        "CERTIFICATIONS",

        "ACHIEVEMENTS",

        "LANGUAGES",

        "HOBBIES",

        "DECLARATION",

        "CONTACT",

        "PROFILE",

        "LINKS"
    ]


    for heading in headings:

        text = re.sub(

            rf'\b{heading}\b',

            f'\n\n{heading}\n',

            text,

            flags=re.IGNORECASE
        )




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

            "certifications"
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


        # ==========================================
        # INDUSTRIAL DYNAMIC HEADING DETECTION
        # ==========================================

        words = clean_line.split()

        uppercase_ratio = sum(
            1 for c in clean_line if c.isupper()
        ) / max(len(clean_line), 1)

        is_short = len(words) <= 8

        has_no_fullstop = not clean_line.endswith(".")

        valid_words = [

            word for word in words

            if word[0].isalnum()
        ]


        title_case_ratio = sum(

            1 for word in valid_words

            if word[:1].isupper()

        ) / max(len(valid_words), 1)

        starts_with_bullet = clean_line.startswith(("-", "•", "*"))

        is_heading = (

            (
                clean_line.isupper()
                and uppercase_ratio > 0.6
                and is_short
                and not starts_with_bullet
            )

            or

            (
                title_case_ratio > 0.7
                and is_short
                and not starts_with_bullet
                and has_no_fullstop
            )

            or

            clean_line.endswith(":")
        )


        # ==========================================
        # SAVE PREVIOUS SECTION
        # ==========================================

        if is_heading:

            if current_content:

                sections.append({

                    "section": current_section,

                    "content": "\n".join(current_content)
                })


            current_section = clean_line.lower().replace(":", "").strip()

            current_content = []


        else:

            current_content.append(clean_line)


    # ==========================================
    # FINAL SECTION
    # ==========================================

    if current_content:

        sections.append({

            "section": current_section,

            "content": "\n".join(current_content)
        })


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