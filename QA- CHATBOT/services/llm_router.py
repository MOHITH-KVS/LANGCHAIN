import os

from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_openai import ChatOpenAI


load_dotenv()


# =========================
# PRIMARY LLM
# =========================

groq_llm = ChatGroq(

    groq_api_key=os.getenv("GROQ_API_KEY"),

    model_name="llama-3.3-70b-versatile",

    temperature=0
)


# =========================
# BACKUP LLM
# =========================

gemini_llm = ChatGoogleGenerativeAI(

    google_api_key=os.getenv("GOOGLE_API_KEY_1"),

    model="gemini-2.0-flash",

    temperature=0
)



# =========================
# FINAL BACKUP LLM
# =========================

deepseek_llm = ChatOpenAI(

    api_key=os.getenv("OPEN_ROUTER_API_KEY"),

    base_url="https://openrouter.ai/api/v1",

    model="deepseek/deepseek-chat-v3-0324",

    default_headers={
        "HTTP-Referer": "http://localhost",
        "X-Title": "QA Chatbot"
    },

    temperature=0
)


# =========================
# FAILOVER INVOCATION
# =========================

def invoke_llm(prompt):

    try:

        print("\nUSING GROQ")

        return groq_llm.invoke(prompt)

    except Exception as groq_error:

        print("\nGROQ FAILED")

        print(str(groq_error))

        try:

            print("\nSWITCHING TO GEMINI")

            return gemini_llm.invoke(prompt)

        except Exception as gemini_error:

            print("\nGEMINI FAILED")

            print(str(gemini_error))

            try:

                print("\nSWITCHING TO DEEPSEEK")

                return deepseek_llm.invoke(prompt)

            except Exception as deepseek_error:

                print("\nDEEPSEEK FAILED")

                print(str(deepseek_error))

                raise Exception(

                    f"All LLM providers failed.\n"
                    f"Groq Error: {groq_error}\n"
                    f"Gemini Error: {gemini_error}\n"
                    f"DeepSeek Error: {deepseek_error}"
                )