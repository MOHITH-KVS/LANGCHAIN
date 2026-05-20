#GOAL OF THIS FILE

#This file should ONLY:

#| Responsibility     | Why                       |
#| ------------------ | ------------------------- |
#| Load PDFs          | extract text              |
#| Clean text         | remove noise              |
#| Detect sections    | preserve structure        |
#| Add metadata       | smarter retrieval         |
#| Filter noisy pages | improve retrieval quality |


from langchain_community.document_loaders import PyMuPDFLoader
import re

#function to load pdf and return documents
def load_pdf(file_path):
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    return documents

#function to clean text by removing multiple new lines, spaces, tabs, and unnecessary blank lines
import re


def clean_text(text):

    # Remove extra spaces and new lines
    text = re.sub(r'\s+', ' ', text)

    # Remove standalone page numbers
    text = re.sub(r'\b\d+\b(?=\s+[A-Z])', '', text)

    # Remove repeated numbering patterns
    # Example: 1.1Introduction -> Introduction
    text = re.sub(r'\b\d+(\.\d+)+', '', text)

    # Remove multiple dots/symbols
    text = re.sub(r'[•●▪■]', ' ', text)

    # Remove repeated spaces again
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

#function to detect sections based on keywords in the text, returns the section name or "general" if no specific section is detected
def detect_section(text):

    text_lower = text.lower()

    section_keywords = {

        "abstract": [
            "abstract",
            "project summary"
        ],

        "introduction": [
            "introduction",
            "overview"
        ],

        "methodology": [
            "methodology",
            "proposed system",
            "implementation"
        ],

        "architecture": [
            "architecture",
            "system architecture",
            "design"
        ],

        "technologies": [
            "technologies",
            "tools used",
            "tech stack"
        ],

        "conclusion": [
            "conclusion",
            "future work",
            "results"
        ]
    }

    for section, keywords in section_keywords.items():

        for keyword in keywords:

            if keyword in text_lower:

                return section

    return "general"


import re

def split_into_sections(text):

    section_patterns = {

        "abstract": r"\bABSTRACT\b",

        "introduction": r"\bINTRODUCTION\b",

        "methodology": r"\bMETHODOLOGY\b|\bIMPLEMENTATION\b",

        "architecture": r"\bARCHITECTURE\b|\bSYSTEM DESIGN\b",

        "technologies": r"\bTECHNOLOGIES\b|\bTOOLS USED\b",

        "conclusion": r"\bCONCLUSION\b|\bFUTURE WORK\b"
    }

    matches = []

    for section, pattern in section_patterns.items():

        for match in re.finditer(pattern, text, re.IGNORECASE):

            matches.append({

                "section": section,

                "start": match.start()
            })

    matches = sorted(matches, key=lambda x: x["start"])

    sections = []

    for i in range(len(matches)):

        start = matches[i]["start"]

        end = matches[i + 1]["start"] if i + 1 < len(matches) else len(text)

        section_name = matches[i]["section"]

        section_content = text[start:end]

        sections.append({

            "section": section_name,

            "content": section_content
        })

    if not sections:

        sections.append({

            "section": "general",

            "content": text
        })

    return sections

#function to create metadata for each document, including source file path, page number, and detected section
def create_metadata(file_path, page_num, section):

    metadata = {
        "source": file_path,
        "page": page_num,
        "section": section
    }

    return metadata




if __name__ == "__main__":

    docs = load_pdf("GVP-MAAA DOCUMENTATION (1).pdf")

    raw_text = docs[0].page_content

    cleaned_text = clean_text(raw_text)

    section = detect_section(cleaned_text)

    metadata = create_metadata(
        "GVP-MAAA DOCUMENTATION (1).pdf",
        1,
        section
    )

    print(cleaned_text[:500])

    print("\nSECTION DETECTED:")
    print(section)

    print("\nMETADATA:")
    print(metadata)