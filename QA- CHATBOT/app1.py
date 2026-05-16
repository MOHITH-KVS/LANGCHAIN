from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
#from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever

import os
import re

# =========================================================
# LOAD ENV
# =========================================================

load_dotenv()

# =========================================================
# API KEYS
# =========================================================

api_keys = [
    os.getenv("GOOGLE_API_KEY"),
    os.getenv("GOOGLE_API_KEY_1"),
    os.getenv("GOOGLE_API_KEY_2"),
    os.getenv("GOOGLE_API_KEY_3")
]

groq_api_keys = [
    os.getenv("GROQ_API_KEY"),
    os.getenv("GROQ_API_KEY_1")
]

# =========================================================
# MODELS
# =========================================================

# Gemini -> Final Answer Priority

gemini_model_names = [
    "gemini-2.5-flash-lite"
]

# Groq -> Query Rewrite + Re-ranking + Fallback

groq_model_names = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile"
]

# =========================================================
# MODEL STORAGE
# =========================================================

gemini_models = []
groq_models = []
all_answer_models = []

# =========================================================
# LOAD GEMINI MODELS
# =========================================================

for api_key in api_keys:

    if api_key:

        for model_name in gemini_model_names:

            try:

                chat = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=api_key
                )

                gemini_models.append({
                    "chat": chat,
                    "model": model_name,
                    "api_key": api_key
                })

                print(f"Loaded Gemini: {model_name}")

            except Exception as e:

                print("Gemini Load Failed:", e)

# =========================================================
# LOAD GROQ MODELS
# =========================================================

for api_key in groq_api_keys:

    if api_key:

        for model_name in groq_model_names:

            try:

                chat = ChatGroq(
                    model=model_name,
                    api_key=api_key
                )

                groq_models.append({
                    "chat": chat,
                    "model": model_name,
                    "api_key": api_key
                })

                print(f"Loaded Groq: {model_name}")

            except Exception as e:

                print("Groq Load Failed:", e)

# =========================================================
# FINAL ANSWER MODELS
# Gemini First -> Groq Fallback
# =========================================================

all_answer_models = gemini_models + groq_models


# =========================================================
# SECTION DETECTOR
# =========================================================

def detect_section(text):

    text = text.lower()

    possible_sections = [
        "abstract",
        "introduction",
        "literature survey",
        "methodology",
        "system design",
        "implementation",
        "results",
        "conclusion",
        "references"
    ]

    for section in possible_sections:

        if section in text[:300]:

            return section

    return "unknown"



# =========================================================
# LOAD PDFs
# =========================================================

files = [
    "GVP-MAAA DOCUMENTATION (1).pdf",
    "GVP-MAAA PAPER.pdf"
]

documents = []

for file in files:

    try:

        loader = PyMuPDFLoader(file)

        docs = loader.load()

        for doc in docs:

            text = doc.page_content.lower()

            # =========================================
            # SECTION DETECTION
            # =========================================

            if "abstract" in text[:1000]:

                doc.metadata["section"] = "abstract"

            elif "introduction" in text[:1000]:

                doc.metadata["section"] = "introduction"

            elif "conclusion" in text[:1000]:

                doc.metadata["section"] = "conclusion"

            elif "literature survey" in text[:1000]:

                doc.metadata["section"] = "literature survey"

            else:

                doc.metadata["section"] = "general"

                print(f"Pages in current PDF: {len(docs)}")

        # ================================================
        # ADD METADATA
        # ================================================

        for i, doc in enumerate(docs):

            doc.metadata["source"] = file
            doc.metadata["page"] = i + 1

            documents.append(doc)

        print(f"Loaded PDF: {file}")

        for i in range(min(5, len(docs))):

            print(
                f"Page {i} Section:",
                docs[i].metadata.get("section")
            )

    except Exception as e:

        print(f"Failed PDF: {file}")
        print(e)

# =========================================================
# TEXT SPLITTING
# FIXED HEADER SPLITTING ISSUE
# =========================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=300,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " "
    ]
)

print("TOTAL DOCUMENTS:", len(documents))

split_docs = text_splitter.split_documents(documents)

# =========================================================
# BM25 RETRIEVER
# =========================================================

bm25_retriever = BM25Retriever.from_documents(
    split_docs
)

bm25_retriever.k = 10

print("BM25 Retriever Ready")

# =========================================================
# ADD METADATA
# =========================================================

for doc in split_docs:

    section_name = detect_section(doc.page_content)

    doc.metadata["section"] = section_name

    # page number

    if "page" not in doc.metadata:

        doc.metadata["page"] = "unknown"

    # source file

    if "source" not in doc.metadata:

        doc.metadata["source"] = "unknown"

print("\n========== DEBUG CHUNKS ==========\n")

for i, doc in enumerate(split_docs[:20]):

    print(f"\nCHUNK {i}")
    print(doc.page_content[:500])

    print("\n-------------------------")

# =========================================================
# SECTION DETECTION
# =========================================================

section_keywords = [
    "abstract",
    "introduction",
    "conclusion",
    "methodology",
    "results",
    "literature survey",
    "architecture",
    "objective",
    "summary"
]

for doc in split_docs:

    text = doc.page_content.lower()

    detected_section = "general"

    for section in section_keywords:

        if section in text[:500]:

            detected_section = section
            break

    # ============================================
    # STORE SECTION IN METADATA
    # ============================================

    doc.metadata["section"] = detected_section

print(f"TOTAL SPLIT DOCS: {len(split_docs)}")
print("\n========== METADATA DEBUG ==========\n")

for i, doc in enumerate(split_docs[:10]):

    print(f"CHUNK {i}")

    print("SECTION:", doc.metadata.get("section"))

    print("PAGE:", doc.metadata.get("page"))

    print("SOURCE:", doc.metadata.get("source"))

    print(doc.page_content[:300])

    print("\n-------------------------\n")
# =========================================================
# EMBEDDINGS
# =========================================================

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# =========================================================
# VECTOR DATABASE
# =========================================================

db = FAISS.from_documents(split_docs, embeddings)

print("FAISS Database Ready")

# =========================================================
# USER STYLE
# =========================================================

style = input("Style (teacher/funny/strict/friendly): ")

# =========================================================
# MEMORY
# =========================================================

chat_history = []

# =========================================================
# CHAT LOOP
# =========================================================

while True:

    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        break

    # =====================================================
    # STORE USER INPUT
    # =====================================================

    chat_history.append(f"You: {user_input}")

    # =====================================================
    # CONTEXT MEMORY
    # =====================================================

    context = "\n".join(chat_history[-10:])

    # =====================================================
    # IMPORTANT QUERY CHECK
    # =====================================================

    direct_keywords = [
        "abstract",
        "introduction",
        "conclusion",
        "objective",
        "aim",
        "methodology",
        "architecture",
        "literature survey",
        "results",
        "summary"
    ]

    use_direct_query = False

    for word in direct_keywords:

        if word in user_input.lower():

            use_direct_query = True
            break

    # =====================================================
    # QUERY REWRITE
    # =====================================================

    better_query = None

    if use_direct_query:

        better_query = user_input

        print("\n====================================")
        print("DIRECT QUERY MODE ENABLED")
        print("QUERY:", better_query)
        print("====================================\n")

    else:

        search_query = f"""
        You are an advanced conversational query rewriter for RAG systems.

        Your job:
        Convert the user's latest question into a COMPLETE standalone search query.

        IMPORTANT RULES:
        - Preserve original meaning exactly
        - Use conversation history for references like:
        "it", "that", "they", "this section"
        - Expand short follow-up questions into full standalone queries
        - Keep important technical keywords
        - Do NOT hallucinate
        - Do NOT answer the question
        - Do NOT explain
        - Return ONLY optimized search query
        - Keep query concise but complete

        Conversation History:
        {context}

        Latest User Question:
        {user_input}

        Standalone Search Query:
        """

        for item in groq_models:

            try:

                better_query = item["chat"].invoke(
                    search_query
                ).content.strip()

                print("\n====================================")
                print("QUERY REWRITE MODEL:", item["model"])
                print("IMPROVED QUERY:", better_query)
                print("====================================\n")

                break

            except Exception as e:

                print("Query Rewrite Failed:", item["model"])
                print("Error:", e)

    # =====================================================
    # FALLBACK
    # =====================================================

    if not better_query:

        better_query = user_input

        print("Using Original Query")

    # =====================================================
    # IMPORTANT CLEAN QUERY
    # =====================================================

    better_query = better_query.lower().strip()

    # =====================================================
    # HYBRID RETRIEVAL
    # Semantic + BM25
    # =====================================================

    # ============================================
    # SEMANTIC SEARCH (FAISS)
    # ============================================

    semantic_docs = db.max_marginal_relevance_search(
        better_query,
        k=10,
        fetch_k=30,
        lambda_mult=0.7
    )

    # ============================================
    # BM25 KEYWORD SEARCH
    # ============================================

    bm25_docs = bm25_retriever.invoke(
        better_query
    )

    # ============================================
    # COMBINE RESULTS
    # ============================================

    all_docs = semantic_docs + bm25_docs

    # ============================================
    # REMOVE DUPLICATES
    # ============================================

    unique_docs = []
    seen_content = set()

    for doc in all_docs:

        content = doc.page_content.strip()

        if content not in seen_content:

            unique_docs.append(doc)

            seen_content.add(content)

    # ============================================
    # FINAL DOCS
    # ============================================

    docs = unique_docs[:15]

    print("\n====================================")
    print("HYBRID RETRIEVAL ENABLED")
    print("Semantic Docs:", len(semantic_docs))
    print("BM25 Docs:", len(bm25_docs))
    print("Combined Docs:", len(docs))
    print("====================================\n")

    # =====================================================
    # METADATA FILTERING
    # =====================================================

    query_lower = user_input.lower()

    filtered_by_metadata = []

    for doc in docs:

        section = doc.metadata.get(
            "section",
            ""
        ).lower()

        # Abstract queries

        if "abstract" in query_lower:

            if section == "abstract":

                filtered_by_metadata.append(doc)

        # Introduction queries

        elif "introduction" in query_lower:

            if section == "introduction":

                filtered_by_metadata.append(doc)

        # Conclusion queries

        elif "conclusion" in query_lower:

            if section == "conclusion":

                filtered_by_metadata.append(doc)

    # Fallback if nothing found

    if len(filtered_by_metadata) > 0:

        docs = filtered_by_metadata

    # =====================================================
    # KEYWORD EXTRACTION
    # =====================================================

    keywords = re.findall(
        r'\b\w+\b',
        user_input.lower()
    )

    stop_words = {
        "what",
        "is",
        "the",
        "of",
        "a",
        "an",
        "explain",
        "tell",
        "me",
        "about",
        "give"
    }

    keywords = [
        word for word in keywords
        if word not in stop_words
    ]

    print("\nKEYWORDS:", keywords)

    # =====================================================
    # STRONG KEYWORD FILTERING
    # =====================================================

    filtered_docs = []

    for doc in docs:

        chunk = doc.page_content.lower()

        keyword_matches = 0

        for keyword in keywords:

            if keyword in chunk:

                keyword_matches += 1

        # =================================================
        # STRICT FILTERING
        # =================================================

        if keyword_matches >= 1:

            filtered_docs.append(doc)

    # =====================================================
    # HEADER BOOSTING
    # =====================================================

    boosted_docs = []

    for doc in filtered_docs:

        chunk = doc.page_content.lower()

        heading_match = False

        for keyword in keywords:

            if keyword in chunk[:250]:

                heading_match = True
                break

        if heading_match:

            boosted_docs.insert(0, doc)

        else:

            boosted_docs.append(doc)

    filtered_docs = boosted_docs

    # =====================================================
    # FALLBACK IF NOTHING FOUND
    # =====================================================

    if len(filtered_docs) == 0:

        filtered_docs = docs

    print(f"\nSemantic Results: {len(docs)}")
    print(f"Filtered Results: {len(filtered_docs)}")

    # =====================================================
    # FINAL DOCS
    # =====================================================

    docs = filtered_docs[:8]

    # =====================================================
    # RERANKING
    # =====================================================

    best_chunk = ""
    best_score = 0
    best_doc = None

    top_chunks = []
    scored_docs = []

    for doc in docs:

        chunk = doc.page_content

        lower_chunk = chunk.lower().strip()

        # =================================================
        # RERANK PROMPT
        # =================================================

        score_prompt = f"""
        You are a document relevance evaluator.

        Question:
        {user_input}

        Chunk:
        {chunk}

        Instructions:
        - Give score 10 if chunk directly answers the question
        - Prefer chunks with exact section headings
        - Prefer chunks containing direct answer
        - Penalize unrelated sections like certificate,
        acknowledgements, table of contents
        - Return ONLY one number
        - Return ONLY 1 to 10
        - Do NOT explain
        - Do NOT write text
        - Do NOT write /10

        Score:
        """

        current_score = 0

        # =================================================
        # LLM RERANKING
        # =================================================

        for item in groq_models:

            try:

                score_response = item["chat"].invoke(
                    score_prompt
                ).content.strip()

                match = re.search(
                    r'\b([1-9]|10)\b',
                    score_response
                )

                if match:

                    current_score = int(match.group(1))

                else:

                    current_score = 0

                print("--------------------------------")
                print("RERANK MODEL:", item["model"])
                print("SCORE:", current_score)
                print("--------------------------------")

                break

            except Exception as e:

                print("RERANK FAILED:", e)

                continue

        # =================================================
        # SECTION BOOSTING
        # =================================================

        clean_chunk = lower_chunk.strip()

        # ABSTRACT BOOST

        if "abstract" in user_input.lower():

            if clean_chunk.startswith("abstract"):

                current_score += 15

        # INTRODUCTION BOOST

        if "introduction" in user_input.lower():

            if "introduction" in clean_chunk[:120]:

                current_score += 15

        # CONCLUSION BOOST

        if "conclusion" in user_input.lower():

            if "conclusion" in clean_chunk[:120]:

                current_score += 15

        # METHODOLOGY BOOST

        if "methodology" in user_input.lower():

            if "methodology" in clean_chunk[:150]:

                current_score += 15

        # RESULTS BOOST

        if "results" in user_input.lower():

            if "results" in clean_chunk[:150]:

                current_score += 15

        # ARCHITECTURE BOOST

        if "architecture" in user_input.lower():

            if "architecture" in clean_chunk[:150]:

                current_score += 15

        # =================================================
        # KEYWORD BOOSTING
        # =================================================

        keyword_matches = 0

        for keyword in keywords:

            if keyword in clean_chunk:

                keyword_matches += 1

        current_score += keyword_matches

        # =================================================
        # NEGATIVE BOOSTING
        # =================================================

        bad_sections = [
            "certificate",
            "acknowledgement",
            "table of contents",
            "list of figures",
            "list of tables",
            "declaration"
        ]

        for bad_word in bad_sections:

            if bad_word in clean_chunk[:250]:

                current_score -= 10

        # =================================================
        # STORE SCORED DOCS
        # =================================================

        scored_docs.append({

            "score": current_score,
            "content": chunk,
            "metadata": doc.metadata

        })

        # =================================================
        # STORE TOP CHUNKS
        # =================================================

        top_chunks.append({

            "score": current_score,
            "chunk": chunk,
            "doc": doc

        })

        # =================================================
        # BEST DOC SELECTION
        # =================================================

        if current_score > best_score:

            best_score = current_score
            best_chunk = chunk
            best_doc = doc

    # =====================================================
    # SORT TOP CHUNKS
    # =====================================================

    top_chunks = sorted(
        top_chunks,
        key=lambda x: x["score"],
        reverse=True
    )

    # =====================================================
    # SELECT TOP CHUNKS
    # =====================================================

    selected_chunks = top_chunks[:5]

    # =====================================================
    # REMOVE DUPLICATE / OVERLAPPING CHUNKS
    # =====================================================

    unique_selected_chunks = []

    seen_snippets = set()

    for item in selected_chunks:

        chunk = item["chunk"]

        # ============================================
        # CREATE SMALL SIGNATURE
        # ============================================

        signature = chunk[:300].lower().strip()

        # ============================================
        # CHECK DUPLICATES
        # ============================================

        if signature not in seen_snippets:

            unique_selected_chunks.append(item)

            seen_snippets.add(signature)

    # =====================================================
    # FINAL UNIQUE CHUNKS
    # =====================================================

    unique_selected_chunks = unique_selected_chunks[:3]

    # =====================================================
    # BUILD FINAL CONTEXT
    # =====================================================

    combined_context = ""

    for item in unique_selected_chunks:

        combined_context += "\n\n"

        combined_context += item["chunk"]

    # =====================================================
    # DEBUG UNIQUE CHUNKS
    # =====================================================

    print("\n============== FINAL UNIQUE CHUNKS ==============\n")

    for i, item in enumerate(unique_selected_chunks):

        print(f"\n########## UNIQUE CHUNK {i+1} ##########\n")

        print("FINAL SCORE:", item["score"])

        if hasattr(item["doc"], "metadata"):

            print("METADATA:", item["doc"].metadata)

        print("\n")

        print(item["chunk"][:1500])

        print("\n========================================\n")

    # =====================================================
    # BEST DOC DEBUG
    # =====================================================

    if best_doc:

        print("\n============== BEST DOC ==============\n")

        print("BEST SCORE:", best_score)

        if hasattr(best_doc, "metadata"):

            print("BEST DOC METADATA:", best_doc.metadata)

        print("\n======================================\n")

    # =====================================================
    # FINAL PROMPT
    # =====================================================

    final_input = f"""
    You are a {style} assistant.

    RULES:
    - Answer ONLY using DATA
    - Be detailed
    - Preserve headings
    - Never hallucinate
    - If answer exists in DATA, do not say "I don't know"

    DATA:
    {combined_context}

    Conversation:
    {context}

    Final Answer:
    """

    # =====================================================
    # FINAL ANSWER
    # =====================================================

    response = None

    for item in all_answer_models:

        try:

            response = item["chat"].invoke(final_input)

            print("\n================================")
            print("ANSWER MODEL:", item["model"])
            print("================================\n")

            break

        except Exception as e:

            print("Model Failed:", item["model"])
            print("Error:", e)

    # =====================================================
    # OUTPUT
    # =====================================================

    if response:

        print("Bot:", response.content)

        chat_history.append(
            f"Bot: {response.content}"
        )

    else:

        print("All models failed.")