import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
import time

load_dotenv()
api_key = os.getenv("OpenAI_api_key")

if not api_key:
    raise ValueError("Ключ OpenAI_api_key не знайдено! Перевір файл .env")

client = OpenAI(api_key=api_key)


class Message(BaseModel):
    role: str
    text: str


# Розширена модель для аналізатора (важливо для "стиковки" команд)
class DialogueResponse(BaseModel):
    dialogue: list[Message]
    ground_truth_satisfaction: str  # satisfied / neutral / unsatisfied
    has_agent_error: bool
    intended_error_type: str  # напр. ignored_question або none


intents = [
    "проблеми з оплатою",
    "технічні помилки",
    "доступ до акаунту",
    "питання по тарифу",
    "повернення коштів"
]

# Сценарії чітко за вимогами зі скріншота
scenario_instructions = {
    "упішний кейс": "Агент професійний, клієнт задоволений, проблему вирішено.",

    "прихована незадоволеність": """ВАЖЛИВО: Агент дає відписку. Клієнт формально дякує, 
    але його проблема НЕ вирішена. Це вимога ТЗ.""",

    "конфліктний кейс": "Клієнт агресивний, нервовий, використовує знаки оклику. Агент не може його заспокоїти.",

    "помилка агента": """Агент припускається логічної або тональної помилки: 
    грубить (rude_tone), ігнорує питання (ignored_question) або дає невірну інфу (incorrect_info)."""
}


def generate_dialogue(intent, scenario):
    detailed_instruction = scenario_instructions[scenario]

    system_instruction = """Ти — спеціаліст з генерації навчальних даних. 
    Твоя задача: створити реалістичний чат підтримки українською мовою.
    Клієнт: реальна людина, може робити помилки, писати без великих літер, використовувати суржик.
    Агент: працівник компанії КАРИБО."""

    prompt = f"""
        Згенеруй діалог на тему: {intent}.
        СЦЕНАРІЙ: {detailed_instruction}

        ПРАВИЛА:
        1. Агент завжди починає: 'Вітаю, на зв'язку Максим, оператор техпідтримки КАРИБО'.
        2. Довжина: 4-7 реплік.
        3. Якщо сценарій 'прихована незадоволеність', клієнт має завершити фразою на кшталт 'ясно дякую' при не вирішеній проблемі.
        """

    try:
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            response_format=DialogueResponse,
            temperature=0.8,  # Для відповідності вимозі "Різноманітність"
        )

        result = response.choices[0].message.parsed.model_dump()
        # Додаємо мітки для ваших друзів-аналітиків
        result["intent"] = intent
        result["scenario_type"] = scenario

        return result

    except Exception as e:
        print(f"❌ Помилка на {intent} + {scenario}: {e}")
        return None


if __name__ == "__main__":
    start_time = time.time()
    print("🚀 Генерація датасету за вимогами SKELAR...")

    tasks = []
    for intent in intents:
        for scenario in scenario_instructions.keys():
            tasks.append((intent, scenario))

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(lambda x: generate_dialogue(x[0], x[1]), tasks))

    dataset = [res for res in results if res]

    with open("dataset.json", "w", encoding="utf-8") as file:
        json.dump(dataset, file, ensure_ascii=False, indent=4)

    print(f"🎉 Готово! Збережено {len(dataset)} діалогів. Час: {time.time() - start_time:.2f}с")