#GOAL OF THIS FILE

#This file should ONLY:

#Responsibility	            Why
#Load PDFs	                get text
#Clean text	                remove noise
#Detect sections	        preserve structure
#Add metadata	            improve retrieval

from langchain_community.document_loaders import PyMuPDFLoader
import re

#function to load pdf and return documents
def load_pdf(file_path):
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    return documents

#function to clean text by removing multiple new lines, spaces, tabs, and unnecessary blank lines
def clean_text(text):

    # remove multiple new lines
    text = re.sub(r'\n+', '\n', text)

    # replace isolated line breaks with spaces
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

    # remove multiple spaces
    text = re.sub(r' +', ' ', text)

    # remove tabs
    text = re.sub(r'\t+', ' ', text)

    # remove unnecessary blank lines
    text = text.strip()

    return text

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