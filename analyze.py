import os
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("Ключ OPENAI_API_KEY не знайдено!")

client = OpenAI(api_key=api_key)

with open("prompts.json", "r", encoding="utf-8") as f:
    PROMPTS = json.load(f)["analysis"]

class AnalysisResult(BaseModel):
    reasoning: str = Field(
        description="Детальне обґрунтування. ЧОМУ саме така оцінка та задоволеність? Які конкретно репліки чи дії доводять наявність/відсутність помилок?")
    intent: str = Field(
        description="Визначена тема звернення (наприклад: технічні помилки, проблеми з оплатою, доступ до акаунту тощо)")
    satisfaction: Literal["satisfied", "neutral", "unsatisfied"] = Field(description="Оцінка задоволеності клієнта")
    quality_score: int = Field(description="Оцінка якості роботи оператора від 1 до 5")
    agent_mistakes: list[Literal[
        "ignored_question",
        "incorrect_info",
        "rude_tone",
        "no_resolution",
        "unnecessary_escalation"
    ]] = Field(description="Список знайдених помилок. Порожній масив [], якщо помилок немає.")


def analyze_dialogue(chat_data: dict) -> dict:
    chat_str = json.dumps(chat_data, ensure_ascii=False, indent=2)

    user_prompt = PROMPTS["user_prompt"].replace("{chat_str}", chat_str)

    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": PROMPTS["system_instruction"]},
            {"role": "user", "content": user_prompt}
        ],
        response_format=AnalysisResult,
        temperature=0.0,
        top_p=1.0,
        seed=13
    )

    return response.choices[0].message.parsed.model_dump()


def process_chat(item):
    index, chat = item
    chat_id = f"chat_{index + 1}"

    print(f"Analysing {chat_id}...")

    max_attempt = 3
    for attempt in range(max_attempt):
        try:
            analysis = analyze_dialogue(chat)
            return {
                "chat_id": chat_id,
                "original_data": chat,
                "analysis": analysis
            }
        except Exception as e:
            print(f"ERROR {chat_id} (Attempt {attempt + 1}): {e}")
            if attempt < max_attempt - 1:
                time.sleep(2)
            else:
                print(f"Skip {chat_id}")
                return None
    return None


def main():
    dataset_file_name = "dataset.json"

    try:
        with open(dataset_file_name, "r", encoding="utf-8") as f:
            chats = json.load(f)
    except FileNotFoundError:
        print(f"FIle {dataset_file_name} not founDDDDDDD :(")
        return

    start_time = time.time()
    results = []
    tasks = list(enumerate(chats))

    with ThreadPoolExecutor(max_workers=15) as executor:
        processed_results = list(executor.map(process_chat, tasks))

    for res in processed_results:
        if res:
            results.append(res)

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Analysis of {len(results)} chats completed.")
    print(f"Time wasted: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()