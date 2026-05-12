
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

load_dotenv()

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
#chat = ChatGoogleGenerativeAI(
#    model="gemini-2.5-flash-lite",
#    google_api_key=os.getenv("GOOGLE_API_KEY")
#)

models = [
    "gemini-2.5-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-pro"
]
groq_models = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile"
]

gemini_models = []

groq_chat_models = []

for api_key in api_keys:

    for model_name in models:

        chat = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key
        )

        gemini_models.append({
            "chat": chat,
            "model": model_name,
            "api_key": api_key
        })

for api_key in groq_api_keys:

    for model_name in groq_models:

        chat = ChatGroq(
            model=model_name,
            api_key=api_key
        )

        groq_chat_models.append({
            "chat": chat,
            "model": model_name,
            "api_key": api_key
        })

#prompt_template = ChatPromptTemplate.from_template(
#    "Explain like a {style}: {question}"
#  )

prompt_template = ChatPromptTemplate.from_template(
    "You are a {style} assistant.\n\nConversation:\n{question}\n\nAnswer the last user question clearly."
)

style = input("Style (teacher/funny/strict/friendly): ") 

#implementing the rag - phase 6
#loader = TextLoader("data.txt")
#documents = loader.load()

#implementing the rag with pdf data - phase 8
#loader = PyPDFLoader("GVP-MAAA DOCUMENTATION (1).pdf")
#documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200
)

#documents = text_splitter.split_documents(documents)
#multi pdf data loading - phase 8
files = ["GVP-MAAA DOCUMENTATION (1).pdf", "GVP-MAAA PAPER.pdf"]  # add your PDFs
documents = []
for file in files:
    loader = PyPDFLoader(file)
    docs = loader.load()
    documents.extend(docs)

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

db = FAISS.from_documents(documents, embeddings)


# ✅ MEMORY ADDED HERE - this will store the conversation history and pass it to the model for better context understanding
#implementing the memory phase in langchain - fifth phase
chat_history = []
while True:
    user_input = input("You: ")

    if user_input == "exit":
        break
    
    #implementing the prompt phase in langchain - second phase
    #prompt = "Answer in a funny way." + user_input
    #response = chat.invoke([HumanMessage(content=prompt)])

    #implementing the prompt phase in langchain - third phase
    #prompt_template = ChatPromptTemplate.from_template(
    #"Explain in simple terms: {question}"
    #)
    #final_prompt = prompt_template.format_messages(question=user_input)
    #response = chat.invoke(final_prompt)

    #print("Bot:", response.content)

    #implementing  the prompt phase in langchain usign dynamic prompt templates - fourth phase
    
     # ✅ ADD USER INPUT TO MEMORY - phase 5
    chat_history.append(f"You: {user_input}")

    # ✅ CREATE CONTEXT FROM MEMORY - phase 5
    #context = "\n".join(chat_history)
    
      # 🔍 RAG: SEARCH DATA - implementing the rag - phase 6
    #docs = db.similarity_search(user_input, k=3)
    #increasing the chunks size and implemeting the semantic search for better results - phase 9
    #$docs = db.similarity_search(user_input, k=5)
    #retrieved_data = "\n\n".join([doc.page_content for doc in docs[:3]])

    # ✅ MEMORY CONTEXT - phase 6
    context = "\n".join(chat_history)

    # 🔥 QUERY REWRITING - phase 10

    search_query = f"""
    Convert this user question into a clear search query.

    Conversation:
    {context}

    Question:
    {user_input}

    Only return the improved search query.
    """

    better_query = None

    for item in groq_chat_models:

        try:
            better_query = item["chat"].invoke(search_query).content

            print("Query Rewrite Model:", item["model"])
            print("Using API Key:", item["api_key"][:15], "...")
            print("Improved Query:", better_query)

            break

        except Exception as e:

            print("Query Rewrite Failed:", item["model"])

            print("Error:", e)

        print("Improved Query:", better_query)

    # 🔍 SEARCH USING BETTER QUERY
    if better_query:

        docs = db.similarity_search(better_query, k=5)
        best_chunk = ""

        best_score = 0

        for doc in docs:

            chunk = doc.page_content

            score_prompt = f"""
            Question:
            {user_input}

            Chunk:
            {chunk}

            Give relevance score from 1 to 10 only.
            """

            score = 0

            for item in groq_chat_models:

                try:

                    score_response = item["chat"].invoke(score_prompt).content

                    score = int(''.join(filter(str.isdigit, score_response)))

                    break

                except:

                    continue

            if score > best_score:

                best_score = score

                best_chunk = chunk

    else:

        print("Could not generate search query.")

        continue

    
        
    



    #final_prompt = prompt_template.format_messages(
    #    style=style,
    #   question=user_input
    #)
    #response = chat.invoke(final_prompt)
    #print("Bot:", response.content)

    # ✅ INCORPORATE CONTEXT INTO PROMPT - phase 5
    #final_prompt = prompt_template.format_messages(
    #    style=style,
    #    question=context
    #)
    #response = chat.invoke(final_prompt)

    # ✅ ADD BOT RESPONSE TO MEMORY - phase 5
    #chat_history.append(f"Bot: {response.content}")

    #print("Bot:", response.content)

    #implementingusing the rag and memory together - phase 7
     # ✅ FINAL INPUT (RAG + MEMORY + STYLE)
    final_input = f"""
    You are a {style} assistant.

    Answer ONLY from the given data.
    If answer is not in data, say "I don't know".

    DATA:
    {best_chunk}

    Conversation:
    {context}

    Answer clearly:
    """

    response = None

    for item in gemini_models:

        try:
            response = item["chat"].invoke(final_input)

            print("Using Model:", item["model"])

            break

        except Exception as e:

            print("Using Model:", item["model"])

            print("Using API Key:", item["api_key"][:15], "...")

            print("Error:", e)

    if response:

        print("Bot:", response.content)

        # ✅ ADD BOT RESPONSE TO MEMORY
        chat_history.append(f"Bot: {response.content}")

    else:

        print("All models failed.")