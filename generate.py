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

class Message(BaseModel):
    role: Literal["Клієнт", "Оператор"] = Field(
        description="Роль учасника: суворо 'Клієнт' або 'Оператор'."
    )
    text: str = Field(
        description="Текст повідомлення: імена використовуй ТІЛЬКИ тут."
    )


class DialogueResponse(BaseModel):
    dialogue: list[Message]

intents = [
    "проблеми з оплатою",
    "технічні помилки",
    "доступ до акаунту",
    "питання по тарифу",
    "повернення коштів",
    "зміна особистих даних"
]

scenario_instructions = {
    "successful_case": """
        Ситуація: Оператор діє професійно, швидко знаходить рішення.
        Клієнт: Спочатку стурбований, але в кінці щиро дякує і задоволений сервісом.
        Результат: Проблема вирішена повністю. 
    """,
    "hidden_dissatisfaction": """
        Ситуація: Оператор відповідає шаблонно або не заглиблюється в суть.
        Клієнт: Формально ввічливий, але використовує холодні фрази ('зрозуміло', 'ну добре', 'я почув').
        ВАЖЛИВО: Клієнт НЕ каже прямо, що він незадоволений, але з контексту видно, що він просто здався.
        Фінал: Клієнт припиняє діалог, не отримавши реальної допомоги, але не влаштовує скандал.
    """,
    "conflict_case": """
        Ситуація: Клієнт з самого початку агресивний, використовує капс, знаки оклику, можливо легкі образи.
        Оператор: Намагається заспокоїти, але клієнт не слухає аргументів.
        Результат: Клієнт кидає слухавку або погрожує піти до конкурентів. Проблема не вирішена через емоції.
    """,
    "agent_mistake_tone": """
        Ситуація: Клієнт ставить звичайне питання.
        Оператор: Відповідає зверхньо, грубо або використовує пасивно-агресивний тон (rude_tone).
        Клієнт: Шокований такою поведінкою, робить зауваження.
    """,
    "agent_mistake_logic": """
        Ситуація: Клієнт описує проблему.
        Оператор: Дає пораду, яка абсолютно не стосується теми (incorrect_info) або ігнорує пряме запитання (ignored_question).
        Клієнт: Вказує на те, що оператор його не слухає.
    """,
    "unsolvable_policy": """
        Ситуація: Клієнт вимагає те, що заборонено правилами (наприклад, повернення коштів через рік або доступ до чужого акаунту).
        Оператор: Ввічливо, але твердо ВІДМОВЛЯЄ, посилаючись на правила. Це НЕ помилка оператора.
        Клієнт: Залишається незадоволеним фактом відмови, але претензій до роботи оператора не має (або змирився).
        Результат: Якісна робота підтримки при негативному результаті для клієнта.
    """
}

agents_names = ["Максим", "Олена", "Дмитро", "Аліна", "Сергій"]

def generate_dialogue(intent, scenario_key):
    detailed_instruction = scenario_instructions[scenario_key]
    agent_name = random.choice(agents_names)

    system_instruction = f"""
    Ти — сценарист реалістичних діалогів між оператором техпідтримки та клієнтом.
    Твоя задача: написати діалог між клієнтом та сапорт-агентом компанії "КАРИБО".
    
    ФОРМАТУВАННЯ:
    Для поля 'role' використовуй тільки два значення: "Клієнт" або "Оператор". 
    Ім'я оператора ({agent_name}) використовуй ТІЛЬКИ в самому тексті повідомлення (у полі 'text'), 
    наприклад при привітанні.

    ПЕРСОНАЛІЗАЦІЯ:
    1. Клієнт: Жива людина. Може писати з помилками, використовувати суржик, сленг, пропускати коми.
    2. Оператор ({agent_name}): Дотримується скриптів, але не роботизовано. "Роботизовано" - означає,
    що не проявляє емпатії, пише суху інформацію та не показує клієнту, шо він для нього важливий.

    ЗАБОРОНИ:
    - Не використовуй фразу "Ясно, дякую" у кожному діалозі та коли діалог закінчується, намагайся менше 
    використовувати це
    - Не роби діалоги занадто "роботизованими", не забувай що клієнт та оператор - живі люди.
    """

    prompt = f"""
    ТЕМА ЗВЕРНЕННЯ: {intent}
    СЦЕНАРІЙ: {scenario_key}
    ДЕТАЛІ СЦЕНАРІЮ: {detailed_instruction}

    СТРУКТУРА:
    1. Початок: '{agent_name}, підтримка КАРИБО. Підкажіть як до Вас можна звертатись. 
    Чим можу допомогти?' (або варіації).
    2. Довжина: 5-9 реплік (зроби діалог змістовним).
    3. Контекст: Це чат у месенджері.
    """

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=1.0,
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

            parsed_data = json.loads(response.text)
            return parsed_data

        except Exception as e:
            print(f"ПОМИЛКА: {intent} + {scenario_key} (Спроба {attempt + 1}): {e}")
            time.sleep(2)

    print(f"не відбудеться генерація {intent} + {scenario_key}. Пропускаємо.")
    return None

if __name__ == "__main__":
    start_time = time.time()
    print(" Генерація датасету...")

    tasks = []

    for intent in intents:
        for scenario in scenario_instructions.keys():
            tasks.append((intent, scenario))

    random.shuffle(tasks)

    tasks = tasks[:25]

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(lambda x: generate_dialogue(x[0], x[1]), tasks))

    dataset = [res for res in results if res]

    with open("dataset.json", "w", encoding="utf-8") as file:
        json.dump(dataset, file, ensure_ascii=False, indent=4)

    print(f"Готово! Згенеровано {len(dataset)} діалогів. Час: {time.time() - start_time:.2f}с")