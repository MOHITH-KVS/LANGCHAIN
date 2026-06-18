# =============================================================
# services/llm_router.py  —  LIGHTWEIGHT VERSION (no heavy SDKs)
# =============================================================
#
# WHY THIS CHANGED:
#
#   BEFORE: Used langchain-google-genai and langchain-openai
#           packages. Each of these pulls in a large dependency
#           tree just to import (Google's SDK includes grpc and
#           protobuf machinery; OpenAI's client includes its own
#           heavy schema validation layer). Combined with
#           langchain-groq, this added significant baseline
#           memory just from imports, before handling any request.
#
#   AFTER:  Groq still uses its own lightweight official `groq`
#           Python package (already small, kept as-is since it's
#           your primary/most-used provider).
#           Gemini and OpenRouter now use plain HTTP calls via
#           `requests` (which you already have installed and use
#           elsewhere in this project) instead of LangChain
#           wrapper packages.
#
#   SAME EXACT BEHAVIOR: 3-tier failover (Groq -> Gemini -> 
#   OpenRouter/DeepSeek), same prompt handling, same return
#   shape your generation.py expects (an object with `.content`).
#
#   YOU CAN NOW REMOVE these from requirements.txt:
#     langchain-google-genai
#     langchain-openai
#
# =============================================================

import os
import requests
from dotenv import load_dotenv

load_dotenv()


# =============================================================
# SIMPLE RESPONSE WRAPPER
# =============================================================
# Your generation.py likely does something like: response.content
# (that's how LangChain's chat models return text). This tiny class
# mimics that same `.content` attribute so NOTHING else in your
# project needs to change.

class SimpleLLMResponse:
    def __init__(self, content):
        self.content = content


# =============================================================
# GROQ  (unchanged — already lightweight, kept as native SDK)
# =============================================================

def call_groq(prompt):
    from groq import Groq

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    text = response.choices[0].message.content
    return SimpleLLMResponse(text)


# =============================================================
# GEMINI  (now via direct REST call, no langchain-google-genai)
# =============================================================

def call_gemini(prompt):
    api_key = os.getenv("GOOGLE_API_KEY_1")
    model = "gemini-2.0-flash"

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key
    }

    body = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0
        }
    }

    response = requests.post(url, headers=headers, json=body, timeout=30)

    if response.status_code != 200:
        raise Exception(f"Gemini API error: {response.status_code} - {response.text}")

    data = response.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"]

    return SimpleLLMResponse(text)


# =============================================================
# OPENROUTER / DEEPSEEK  (now via direct REST call, no langchain-openai)
# =============================================================

def call_deepseek(prompt):
    api_key = os.getenv("OPEN_ROUTER_API_KEY")

    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "QA Chatbot"
    }

    body = {
        "model": "deepseek/deepseek-chat-v3-0324",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0
    }

    response = requests.post(url, headers=headers, json=body, timeout=30)

    if response.status_code != 200:
        raise Exception(f"OpenRouter API error: {response.status_code} - {response.text}")

    data = response.json()
    text = data["choices"][0]["message"]["content"]

    return SimpleLLMResponse(text)


# =============================================================
# FAILOVER INVOCATION  (same 3-tier logic as before)
# =============================================================

def invoke_llm(prompt):

    try:
        print("\nUSING GROQ")
        return call_groq(prompt)

    except Exception as groq_error:
        print("\nGROQ FAILED")
        print(str(groq_error))

        try:
            print("\nSWITCHING TO GEMINI")
            return call_gemini(prompt)

        except Exception as gemini_error:
            print("\nGEMINI FAILED")
            print(str(gemini_error))

            try:
                print("\nSWITCHING TO DEEPSEEK")
                return call_deepseek(prompt)

            except Exception as deepseek_error:
                print("\nDEEPSEEK FAILED")
                print(str(deepseek_error))

                raise Exception(
                    f"All LLM providers failed.\n"
                    f"Groq Error: {groq_error}\n"
                    f"Gemini Error: {gemini_error}\n"
                    f"DeepSeek Error: {deepseek_error}"
                )