import os
from dotenv import load_dotenv
from openai import OpenAI
from google import genai
from google.genai import types

load_dotenv()

# 1. ІНІЦІАЛІЗАЦІЯ КЛІЄНТІВ

openai_key = os.getenv("OPENAI_API_KEY")
gemini_key = os.getenv("GEMINI_API_KEY")

if not openai_key:
    raise ValueError("Ключ OPENAI_API_KEY не знайдено!")
if not gemini_key:
    raise ValueError("Ключ GEMINI_API_KEY не знайдено!")

openai_client = OpenAI(api_key=openai_key)
gemini_client = genai.Client(api_key=gemini_key)

# 2. ПАРАМЕТРИ МОДЕЛЕЙ

# Налаштування для аналізатора
analyzer_config = {
    "model": "gpt-4o-mini",
    "temperature": 0.0,
    "top_p": 1.0,
    "seed": 13
}

# Налаштування для генератора
generator_model = "gemini-2.5-flash"

generator_config = {
    "temperature": 1.0,
    "safety_settings": [
        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),
        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE)
    ]
}