#GOAL OF THIS FILE
#This file should ONLY:
#| Responsibility                | Why                              |
#| ----------------------------- | -------------------------------- |
#| Split cleaned text            | create manageable semantic units |
#| Preserve semantic meaning     | improve embeddings               |
#| Maintain chunk overlap        | preserve continuity              |
#| Attach metadata to chunks     | smarter retrieval                |
#| Prepare chunks for embeddings | improve vector search            |
#|Filter weak chunks             | improve retrieval quality        |


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


# Function to check whether chunk is meaningful
def is_meaningful_chunk(text):

    text = text.strip()

    # Remove very tiny chunks
    if len(text) < 80:
        return False

    # Remove chunks with too few words
    if len(text.split()) < 15:
        return False

    return True


# Function to create chunks from cleaned text
def create_chunks(text, metadata):

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=500,

        chunk_overlap=100,

        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    split_chunks = splitter.split_text(text)

    chunks = []

    for chunk in split_chunks:

        enhanced_chunk = f"""

    SECTION: {metadata['section']}

    {chunk}

    """

        chunk_data = {

            "content": enhanced_chunk,

            "metadata": metadata
        }

        chunks.append(chunk_data)

    return chunks


# Testing Section
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

    chunks = create_chunks(
        cleaned_text,
        metadata
    )

    print("FIRST CHUNK:\n")

    print(chunks[0])

    print("\nTOTAL CHUNKS:")

    print(len(chunks))