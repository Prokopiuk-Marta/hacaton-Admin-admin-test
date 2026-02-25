import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
import time

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("Ключ OpenAI_api_key не знайдено!")

client = OpenAI(api_key=api_key)


class Message(BaseModel):
    role: str
    text: str


class DialogueResponse(BaseModel):
    dialogue: list[Message]


intents = [
    "проблеми з оплатою",
    "технічні помилки",
    "доступ до акаунту",
    "питання по тарифу",
    "повернення коштів"
]

scenario_instructions = {
    "уcпішний кейс": "оператор професійний, клієнт задоволений, проблему вирішено.",

    "прихована незадоволеність": """оператор відписує, клієнт формально дякує, 
    але його проблема НЕ вирішена. також може бути клієнт не в настрої, але його проблему вирішили, однак він все ще залишився
    незадоволеним""",

    "конфліктний кейс": """Клієнт агресивний, нервовий, використовує знаки оклику, оператор не може його заспокоїти, через шо клієнт,
    починає безпідставно обзивати оператора, погрожувати.""",

    "помилка оператора": """оператор припускається логічної або тональної помилки: 
    грубить (rude_tone), ігнорує питання (ignored_question) або дає невірну інфу (incorrect_info)"""
}


def generate_dialogue(intent, scenario):
    detailed_instruction = scenario_instructions[scenario]

    system_instruction = """Ти — спеціаліст з генерації навчальних даних. 
    Твоя задача: створити реалістичний чат підтримки українською мовою.
    Клієнт: реальна людина, може робити помилки, писати без великих літер, використовувати суржик.
    Оператор: працівник компанії "КАРИБО"."""

    prompt = f"""
        Згенеруй діалог на тему: {intent}.
        СЦЕНАРІЙ: {detailed_instruction}

        ПРАВИЛА:
        1. Агент завжди починає: 'Вітаю, на зв'язку Максим, оператор техпідтримки КАРИБО'.
        2. Довжина: 4-7 реплік.
        3. Якщо сценарій 'прихована незадоволеність', клієнт має завершити фразою на кшталт 'ясно дякую' при не вирішеній проблемі.
        """
    for attempt in range(3):
        try:
            response = client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt}
                ],
                response_format=DialogueResponse,
                temperature=0.8,
                seed=13
            )

            result = response.choices[0].message.parsed.model_dump()
            result["intent"] = intent
            result["scenario_type"] = scenario

            return result

        except Exception as e:
            print(f"ПОМИЛКА: {intent} + {scenario} (Спроба {attempt + 1}): {e}")
            time.sleep(2)

    print(f"Остаточний провал для {intent} + {scenario}. Пропускаємо.")
    return None


if __name__ == "__main__":
    start_time = time.time()
    print("🚀 Генерація датасету...")

    tasks = []
    for intent in intents:
        for scenario in scenario_instructions.keys():
            tasks.append((intent, scenario))

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(lambda x: generate_dialogue(x[0], x[1]), tasks))

    dataset = [res for res in results if res]

    with open("dataset.json", "w", encoding="utf-8") as file:
        json.dump(dataset, file, ensure_ascii=False, indent=4)

    print(f"Готово! Час: {time.time() - start_time:.2f}с")