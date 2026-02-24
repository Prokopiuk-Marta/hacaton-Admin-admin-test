import json
import time
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

load_dotenv()
client = genai.Client()

class AnalysisResult(BaseModel):
    intent: str = Field(description="""Проблеми з оплатою, технічні помилки, доступ до акаунту, питання по тарифу, 
    повернення коштів або other""")
    satisfaction: str = Field(description="Одне з: satisfied, neutral, unsatisfied")
    quality_score: int = Field(description="Оцінка від 1 до 5. 1 - дуже погано, 5 - чудово")
    agent_mistakes: list[str] = Field(description="""
    Список помилок (ignored_question, incorrect_info, rude_tone, no_resolution,
     unnecessary_escalation). Якщо помилок немає, поверни порожній масив []""")

system_instruction = """
Ти - головний інспектор з контролю якості служби підтримки компанії "КАРИБО".
Твоє завдання — глибоко та об'єктивно проаналізувати діалог між клієнтом та агентом. Не роби поверхневих висновків. Спирайся на 
критерії

 Критерії оцінювання:

1. Рівень задоволеності клієнта:
- "satisfied": проблема повністю вирішена, клієнт щиро вдячний.
- "neutral": проблему вирішено, але після довгих чи плутаних пояснень. АБО проблему не вирішено, але агент професійно все пояснив 
і клієнт спокійно це прийняв
- "unsatisfied": проблема клієнта не була вирішена, але окрім цього агент міг ввести себе нетактовно, грубіянити або ж відмовився 
допомогти.

2. Оцінка якості роботи агента від 1 до 5:
- 5: швидке розуміння суті проблеми, чіткий алгоритм вирішення проблеми, ідеальна робота.
- 4: проблема вирішена, але агент ставив зайві запитання, але при цьому клієнт задоволений, не вирішили проблему, оскільки це не
стосується даної компанії
- 3: агент був ввічливим, але рішення не знайшов; або знайшов, але дуже довго клієнт чекав; або не врахував стан клієнта
наприклад, клієнт боїться, а агент просто кидає сухі інструкції.
- 2: агент дає шаблонні відповіді, які не стосуються конкретної проблеми клієнта, ігнорує частину запитань.
- 1: звинувачення клієнта ("ви самі щось натиснули"), обзивання, суперечка, розголошення особистих даних.

3. Типізація помилок - уважно шукай ці патерни:
- "ignored_question": клієнт запитав про одне (наприклад, "а що з роутером?"), а агент відповів про інше або проігнорував.
- "incorrect_info": агент дає абсурдні або технічно неграмотні поради.
- "rude_tone": зверхність, сарказм, відсутність базового привітання/прощання, знецінення проблеми клієнта.
- "no_resolution": діалог завершився, але клієнту не сказали, що робити далі (навіть якщо це заявка на майстра).
- "unnecessary_escalation": агент міг допомогти сам, але відправив клієнта до іншого відділу чи запропонував викликати майстра 
на платній основі без потреби.
"""

def analyze_dialogue(dialogue_text: str) -> dict:
    response = client.models.generate_content(
        model='gemini-2.5-pro',
        contents=f"Діалог для аналізу:\n{dialogue_text}",
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=AnalysisResult,
            temperature=0.0,
            seed=13
        )
    )
    return json.loads(response.text)


def process_chat(item):
    index, chat = item
    chat_id = f"chat_{index + 1}"
    intent = chat.get('intent', 'Невідомо')

    print(f"Аналізую {chat_id} (Тема: {intent})...")
    dialogue_text = json.dumps(chat.get('dialogue', chat), ensure_ascii=False, indent=2)
    max_attempt = 3
    for attempt in range(max_attempt):
        try:
            analysis = analyze_dialogue(dialogue_text)
            return {
                "chat_id": chat_id,
                "original_intent": intent,
                "analysis": analysis
            }
        except Exception as e:
            print(f"Помилка для {chat_id} (Спроба {attempt + 1} з {max_attempt}): {e}")
            if attempt < max_attempt - 1:
                time.sleep(2)
            else:
                print(f" - кейс {chat_id}. Пропускаємо його.")
                return None


def main():
    try:
        with open("dataset.json", "r", encoding="utf-8") as f:
            chats = json.load(f)
    except FileNotFoundError:
        print("Файл dataset.json не знайдено.")
        return
    start_time = time.time()
    results = []

    tasks = list(enumerate(chats))

    with ThreadPoolExecutor(max_workers=5) as executor:
        processed_results = list(executor.map(process_chat, tasks))

    for res in processed_results:
        if res:
            results.append(res)

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print("Аналіз завершено!")
    time_end = time.time()
    print(f"time: {time_end - start_time}s")


if __name__ == "__main__":
    main()
