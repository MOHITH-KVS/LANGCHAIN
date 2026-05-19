#GOAL OF THIS FILE
#This file should ONLY:
#| Responsibility                | Why                              |
#| ----------------------------- | -------------------------------- |
#| Split cleaned text            | create manageable semantic units |
#| Preserve semantic meaning     | improve embeddings               |
#| Maintain chunk overlap        | preserve continuity              |
#| Attach metadata to chunks     | smarter retrieval                |
#| Prepare chunks for embeddings | improve vector search            |

#WHAT WE WILL BUILD

#We will:

#take cleaned text
#split semantically
#preserve overlap
#preserve metadata
#create structured chunks

from langchain_text_splitters import RecursiveCharacterTextSplitter
from document_processor import (
    load_pdf,
    clean_text,
    detect_section,
    create_metadata
)



#function to create chunks from cleaned text using RecursiveCharacterTextSplitter, returns a list of chunks
def create_chunks(text, metadata):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    split_chunks = splitter.split_text(text)

    chunks = []

    for chunk in split_chunks:

        chunk_data = {
            "content": chunk,
            "metadata": metadata
        }

        chunks.append(chunk_data)

    return chunks

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

        chunks = create_chunks(cleaned_text, metadata)

        print(chunks[0])

        print("\nTOTAL CHUNKS:")
        print(len(chunks))