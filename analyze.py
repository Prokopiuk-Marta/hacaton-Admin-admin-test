import json
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Завантажуємо ключ
load_dotenv()
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# 1. Задаємо системну інструкцію напряму в модель (це працює набагато сильніше, ніж звичайний текст)
system_instruction = """
Ти — інспектор з контролю якості. Твоя єдина мета — повернути JSON з 4 полями. 
ТИ ЗОБОВ'ЯЗАНИЙ ПОВЕРНУТИ ВСІ 4 ПОЛЯ, БЕЗ ВИНЯТКІВ!

Шаблон, який ти маєш суворо заповнити:
{
  "intent": "вибери одне: проблеми з оплатою, технічні помилки, доступ до акаунту, питання по тарифу, повернення коштів, other",
  "satisfaction": "вибери одне: satisfied, neutral, unsatisfied",
  "quality_score": 5, 
  "agent_mistakes": ["ignored_question", "incorrect_info", "rude_tone", "no_resolution", "unnecessary_escalation"]
}

Правила:
- quality_score: ціле число від 1 до 5.
- agent_mistakes: якщо помилок немає, поверни порожній масив [].
- satisfaction: обов'язково шукай приховану незадоволеність (наприклад, клієнт каже "дякую", але його проблему не вирішили). У такому разі став "unsatisfied".
"""

# 2. Ініціалізуємо модель з правильною інструкцією та налаштуваннями
model = genai.GenerativeModel(
    model_name='gemini-2.5-pro',
    system_instruction=system_instruction, # Вбудовуємо інструкцію прямо в "мозок" моделі
    generation_config=genai.GenerationConfig(
        response_mime_type="application/json", # Гарантуємо, що це буде JSON
        temperature=0.0 # Робимо результати стабільними
    )
)

def analyze_dialogue(dialogue_text: str) -> dict:
    # Тепер ми передаємо лише сам діалог
    response = model.generate_content(f"Діалог для аналізу:\n{dialogue_text}")
    return json.loads(response.text)

def main():
    try:
        with open("dataset.json", "r", encoding="utf-8") as f:
            chats = json.load(f)
    except FileNotFoundError:
        print("Файл dataset.json не знайдено.")
        return

    results = []

    for chat in chats:
        print(f"Аналізую чат ID: {chat.get('id', 'Невідомо')}...")
        dialogue_text = json.dumps(chat.get('dialogue', chat), ensure_ascii=False, indent=2)

        try:
            analysis = analyze_dialogue(dialogue_text)
            chat_result = {
                "chat_id": chat.get('id', 'Невідомо'),
                "analysis": analysis
            }
            results.append(chat_result)
        except Exception as e:
            print(f"Помилка при аналізі чату {chat.get('id')}: {e}")

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print("✅ Аналіз завершено! Файл results.json оновлено.")

if __name__ == "__main__":
    main()