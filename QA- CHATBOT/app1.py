from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
# LOAD PDFs
# =========================================================

files = [
    "GVP-MAAA DOCUMENTATION (1).pdf",
    "GVP-MAAA PAPER.pdf"
]

documents = []

for file in files:

    try:

        loader = PyPDFLoader(file)

        docs = loader.load()

        documents.extend(docs)

        print(f"Loaded PDF: {file}")

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

split_docs = text_splitter.split_documents(documents)

print(f"Total Chunks: {len(split_docs)}")

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
        You are a search query optimizer.

        RULES:
        - Keep the meaning EXACTLY SAME
        - Do NOT add extra concepts
        - Do NOT hallucinate
        - Do NOT explain
        - Return SHORT query only
        - Preserve important keywords

        Conversation:
        {context}

        Question:
        {user_input}

        Optimized Search Query:
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
    # SIMILARITY SEARCH
    # FIXED MMR ISSUE
    # =====================================================

    docs = db.similarity_search(
        better_query,
        k=10
    )

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

    for doc in docs:

        chunk = doc.page_content

        score_prompt = f"""
        You are a document relevance evaluator.

        Question:
        {user_input}

        Chunk:
        {chunk}

        Instructions:
        - Give score 10 if chunk directly answers the question
        - Prefer heading matches
        - Prefer exact keyword matches
        - Return ONLY one number
        - Return ONLY 1 to 10
        - Do NOT explain
        - Do NOT write text
        - Do NOT write /10

        Score:
        """

        current_score = 0

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

            except Exception:

                continue

        # =================================================
        # EXTRA MANUAL BOOST
        # =================================================

        lower_chunk = chunk.lower()

        for keyword in keywords:

            if keyword in lower_chunk[:300]:

                current_score += 2

        # =================================================
        # SECTION TITLE BOOST
        # =================================================

        if "abstract" in user_input.lower():

            if "abstract" in lower_chunk[:500]:

                current_score += 5

        if "introduction" in user_input.lower():

            if "introduction" in lower_chunk[:500]:

                current_score += 5

        if "conclusion" in user_input.lower():

            if "conclusion" in lower_chunk[:500]:

                current_score += 5

        if current_score > best_score:

            best_score = current_score
            best_chunk = chunk

    # =====================================================
    # DEBUG BEST CHUNK
    # =====================================================

    print("\n============== BEST CHUNK ==============\n")

    print(best_chunk[:2500])

    print("\n========================================\n")

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
    {best_chunk}

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