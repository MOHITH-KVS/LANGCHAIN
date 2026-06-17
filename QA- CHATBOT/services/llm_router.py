# =============================================================
# services/llm_router.py  —  MEMORY-OPTIMIZED VERSION
# =============================================================
#
# WHY THIS CHANGED:
#
#   BEFORE: All 3 LLM clients (Groq, Gemini, DeepSeek/OpenRouter)
#           were created at import time — the moment this file
#           loads, ALL THREE client objects exist in memory at once,
#           each with their own internal HTTP clients, retry logic,
#           and schema validators (LangChain wraps these heavily).
#
#   AFTER:  Each client is created ONLY when actually needed
#           (lazy loading). In the normal case, only Groq is ever
#           created since it's the primary and rarely fails.
#           Gemini and DeepSeek clients are created on-demand,
#           only if Groq actually fails.
#
#   This significantly reduces baseline memory usage since you're
#   not holding 3 heavy client objects in RAM at all times —
#   only the ones actually used in a given request.
#
# Your chatbot_engine.py and generation.py call invoke_llm(prompt)
# exactly the same way — NOTHING else needs to change.
#
# =============================================================

import os
from dotenv import load_dotenv

load_dotenv()

# =========================
# LAZY CLIENT CACHE
# =========================
# These stay None until first actually used.
# Once created, they're reused (not recreated every request).

_groq_llm = None
_gemini_llm = None
_deepseek_llm = None


def get_groq_llm():
    global _groq_llm
    if _groq_llm is None:
        from langchain_groq import ChatGroq
        _groq_llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.3-70b-versatile",
            temperature=0
        )
    return _groq_llm


def get_gemini_llm():
    global _gemini_llm
    if _gemini_llm is None:
        from langchain_google_genai import ChatGoogleGenerativeAI
        _gemini_llm = ChatGoogleGenerativeAI(
            google_api_key=os.getenv("GOOGLE_API_KEY_1"),
            model="gemini-2.0-flash",
            temperature=0
        )
    return _gemini_llm


def get_deepseek_llm():
    global _deepseek_llm
    if _deepseek_llm is None:
        from langchain_openai import ChatOpenAI
        _deepseek_llm = ChatOpenAI(
            api_key=os.getenv("OPEN_ROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            model="deepseek/deepseek-chat-v3-0324",
            default_headers={
                "HTTP-Referer": "http://localhost",
                "X-Title": "QA Chatbot"
            },
            temperature=0
        )
    return _deepseek_llm


# =========================
# FAILOVER INVOCATION  (same logic as before, just lazy-loaded clients)
# =========================

def invoke_llm(prompt):

    try:
        print("\nUSING GROQ")
        groq_llm = get_groq_llm()
        return groq_llm.invoke(prompt)

    except Exception as groq_error:
        print("\nGROQ FAILED")
        print(str(groq_error))

        try:
            print("\nSWITCHING TO GEMINI")
            gemini_llm = get_gemini_llm()
            return gemini_llm.invoke(prompt)

        except Exception as gemini_error:
            print("\nGEMINI FAILED")
            print(str(gemini_error))

            try:
                print("\nSWITCHING TO DEEPSEEK")
                deepseek_llm = get_deepseek_llm()
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