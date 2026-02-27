import os
import sys
import json
import random
import time
import threading
from dotenv import load_dotenv
from typing import Literal
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor

from google import genai
from google.genai import types
from LISTS_DATA import (
    intents,
    scenario_keys,
    mistakes_agent,
    agents_names,
    client_names
)
with open("prompts.json", "r", encoding="utf-8") as f:
    PROMPTS = json.load(f)["generation"]

class SpinnerTimer:
    """
    Класс, який запускає таймер в окремому потоці, поки виконується код всередині 'with'.
    """

    def __enter__(self):
        self.stop_event = threading.Event()
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._animate, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_event.set()
        self.thread.join()
        sys.stdout.write("\rGeneration completed. Check dataset file.             \n")
        sys.stdout.flush()

    def _animate(self):
        spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        while not self.stop_event.is_set():
            elapsed = int(time.time() - self.start_time)
            mins, secs = divmod(elapsed, 60)

            spin = spinner_chars[elapsed % len(spinner_chars)]

            sys.stdout.write(f"\r{spin} Generating dataset... [{mins:02d}:{secs:02d}]")
            sys.stdout.flush()
            time.sleep(0.1)

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("KEY GEMINI_API_KEY not found!")

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

def generate_dialogue(intent, scenario_key):
    current_agent = random.choice(agents_names)
    current_client = random.choice(client_names)

    sys_instruction = PROMPTS["system_instruction"]
    sys_instruction = sys_instruction.replace("{current_agent}", current_agent)
    sys_instruction = sys_instruction.replace("{current_client}", current_client)
    sys_instruction = sys_instruction.replace("{mistakes_agent}", str(mistakes_agent))

    def load_examples():
        try:
            with open("examples.json", "r", encoding="utf-8") as f:
                data = json.load(f)

            examples_text = ""
            for i, ex in enumerate(data):
                examples_text += f"\n--- ПРИКЛАД {i + 1} ---\n"
                if 'dialogue' in ex:
                    examples_text += f"ВХІДНИЙ ДІАЛОГ:\n{ex['dialogue']}\n"
            return examples_text
        except FileNotFoundError:
            return ""

    examples_str = load_examples()

    user_prompt = PROMPTS["user_prompt"]
    user_prompt = user_prompt.replace("{intent}", intent)
    user_prompt = user_prompt.replace("{scenario_key}", scenario_key)
    user_prompt = user_prompt.replace("{examples_str}", examples_str)

    custom_safety_settings = [
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        )
    ]

    config = types.GenerateContentConfig(
        system_instruction=sys_instruction,
        temperature=1.0,
        response_mime_type="application/json",
        response_schema=DialogueResponse,
        safety_settings=custom_safety_settings
    )

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=user_prompt,
                config=config
            )
            parsed_data = json.loads(response.text)
            if "dialogue" in parsed_data:
                merged_dialogue = []
                for msg in parsed_data["dialogue"]:
                    if merged_dialogue and merged_dialogue[-1]["role"] == msg["role"]:
                        merged_dialogue[-1]["text"] += " " + msg["text"]
                    else:
                        merged_dialogue.append(msg)
                parsed_data["dialogue"] = merged_dialogue
            return parsed_data

        except Exception as e:
            print(f"ERROR: {intent} + {scenario_key} (Attempt {attempt + 1}): {e}")
            time.sleep(2)

    print(f"Don't available {intent} + {scenario_key}. Skip")
    return None

if __name__ == "__main__":
    start_time = time.time()


    tasks = []

    for intent in intents:
        for scenario in scenario_keys:
            tasks.append((intent, scenario))

    random.shuffle(tasks)

    tasks = tasks[:20]
    with SpinnerTimer():
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(lambda x: generate_dialogue(x[0], x[1]), tasks))

    dataset = [res for res in results if res]

    with open("dataset.json", "w", encoding="utf-8") as file:
        json.dump(dataset, file, ensure_ascii=False, indent=4)
