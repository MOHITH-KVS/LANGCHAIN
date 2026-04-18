
from langchain_google_genai import ChatGoogleGenerativeAI
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

chat = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

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
loader = PyPDFLoader("GVP-MAAA DOCUMENTATION (1).pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

documents = text_splitter.split_documents(documents)

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
    docs = db.similarity_search(user_input, k=3)
    retrieved_data = "\n\n".join([doc.page_content for doc in docs[:3]])
    
     # ✅ MEMORY CONTEXT - phase 6
    context = "\n".join(chat_history)

    

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
    {retrieved_data}

    Conversation:
    {context}

    Answer clearly:
    """

    response = chat.invoke(final_input)

    print("Bot:", response.content)

    # ✅ ADD BOT RESPONSE TO MEMORY
    chat_history.append(f"Bot: {response.content}")