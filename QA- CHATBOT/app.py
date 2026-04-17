from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

chat = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
while True:
    user_input = input("You: ")

    if user_input == "exit":
        break
    
    #implementing the prompt phase in langchain - second phase
    prompt = "Answer in a funny way." + user_input
    response = chat.invoke([HumanMessage(content=prompt)])

    print("Bot:", response.content)