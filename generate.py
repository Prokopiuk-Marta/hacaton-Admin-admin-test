import os
import json
import random
import time
from dotenv import load_dotenv
from typing import Literal
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor
from google import genai
from google.genai import types

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Ключ GEMINI_API_KEY не знайдено!")

client = genai.Client(api_key=api_key)

# 1. Завантажуємо зовнішні знання
with open("scenarios.json", "r", encoding="utf-8") as f:
    cfg = json.load(f)


# 2. Сувора модель Pydantic для Gemini
class Message(BaseModel):
    role: Literal["Клієнт", "Оператор"] = Field(description="Тільки 'Клієнт' або 'Оператор'")
    text: str = Field(description="Текст репліки")


class DialogueResponse(BaseModel):
    dialogue: list[Message]


def generate_dialogue(intent, scenario_key):
    # Додаємо рандомності для унікальності
    agent_name = random.choice(cfg["agents_names"])
    client_profile = random.choice(cfg["client_profiles"])
    scenario_desc = cfg["scenarios"][scenario_key]

    system_instruction = f"""
    Ти — сценарист реалістичних діалогів для техпідтримки "КАРИБО".
    Твоя головна задача — СТВОРИТИ НОВИЙ, УНІКАЛЬНИЙ ДІАЛОГ. 
    ЗАБОРОНЕНО копіювати дослівно приклади з опису сценарію. Вигадай свої причини проблеми!

    ПРАВИЛА РОЛЕЙ:
    1. Клієнт: {client_profile}. Помилки мають бути природними.
    2. Оператор ({agent_name}): Дотримується скриптів, але не як робот. Ім'я використовує тільки при привітанні.

    ЗАБОРОНИ:
    - Не використовуй фразу "Ясно, дякую" як затичку.
    - Уникай ідеально-штучних діалогів. Це має бути реальне життя.
    """

    prompt = f"""
    ТЕМА ЗВЕРНЕННЯ: {intent}
    СЦЕНАРІЙ (Підтекст, який треба розіграти): {scenario_desc}

    СТРУКТУРА:
    1. Починає клієнт (описує проблему).
    2. Оператор представляється ("Вітаю, я {agent_name}...") і діє згідно зі сценарієм.
    3. Довжина: 6-10 реплік (зроби діалог змістовним).
    """

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=1.0,  # Підняв температуру для більшої креативності ШІ
        response_mime_type="application/json",
        response_schema=DialogueResponse,
    )

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=config
            )

            # Додаємо метадані вручну, щоб потім знати, що ми згенерували
            parsed_data = json.loads(response.text)
            parsed_data["metadata"] = {
                "intent": intent,
                "scenario": scenario_key,
                "profile": client_profile
            }
            return parsed_data

        except Exception as e:
            print(f"⚠️ Помилка: {intent} + {scenario_key} (Спроба {attempt + 1}): {e}")
            time.sleep(2)

    return None


if __name__ == "__main__":
    start_time = time.time()
    print("🚀 Генерація датасету розпочата...")

    tasks = []
    for intent in cfg["intents"]:
        for scenario in cfg["scenarios"].keys():
            tasks.append((intent, scenario))

    random.shuffle(tasks)
    tasks = tasks[:25]  # Генеруємо 25 штук

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(lambda x: generate_dialogue(x[0], x[1]), tasks))

    dataset = [res for res in results if res]

    with open("dataset.json", "w", encoding="utf-8") as file:
        json.dump(dataset, file, ensure_ascii=False, indent=4)

    print(f"✅ Готово! Згенеровано {len(dataset)} діалогів. Час: {time.time() - start_time:.2f}с")