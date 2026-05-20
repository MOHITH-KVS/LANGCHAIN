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

    text_upper = text.upper()

    if "ABSTRACT" in text_upper:
        return "abstract"

    elif "INTRODUCTION" in text_upper:
        return "introduction"

    elif "LITERATURE SURVEY" in text_upper:
        return "literature_survey"

    elif "METHODOLOGY" in text_upper:
        return "methodology"

    elif "CONCLUSION" in text_upper:
        return "conclusion"

    else:
        return "general"

#function to create metadata for each document, including source file path, page number, and detected section
def create_metadata(file_path, page_num, section):

    metadata = {
        "source": file_path,
        "page": page_num,
        "section": section
    }

    return metadata

#FUNCTION TO FILTER NOISY PAGES BASED ON KEYWORDS, RETURNS TRUE IF PAGE IS USEFUL AND FALSE IF IT IS NOISY
def is_useful_page(text):

    noisy_keywords = [
        "certificate",
        "acknowledgement",
        "table of contents",
        "index",
        "submitted by",
        "approved by",
        "department of",
        "college for degree",
        "visakhapatnam"
    ]

    text_lower = text.lower()

    for keyword in noisy_keywords:

        if keyword in text_lower:
            return False

    return True




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