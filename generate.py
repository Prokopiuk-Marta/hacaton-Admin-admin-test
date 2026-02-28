import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from google.genai import types
from pydantic import BaseModel, Field

from config import gemini_client, generator_config, generator_model
from lists_data import (agents_names, client_names, intents, mistakes_agent,
                        scenario_keys)
from utils import SpinnerTimer

with open("prompts.json", "r", encoding="utf-8") as f:
    prompts = json.load(f)["generation"]


class Message(BaseModel):
    role: Literal["Клієнт", "Оператор"] = Field(
        description="Роль учасника: суворо 'Клієнт' або 'Оператор'."
    )
    text: str = Field(description="Текст повідомлення: імена використовуй ТІЛЬКИ тут.")


class DialogueResponse(BaseModel):
    dialogue: list[Message]


def generate_dialogue(intent, scenario_key):
    current_agent = random.choice(agents_names)
    current_client = random.choice(client_names)

    sys_instruction = prompts["system_instruction"]
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
                if "dialogue" in ex:
                    examples_text += f"ВХІДНИЙ ДІАЛОГ:\n{ex['dialogue']}\n"
            return examples_text
        except FileNotFoundError:
            return ""

    examples_str = load_examples()

    user_prompt = prompts["user_prompt"]
    user_prompt = user_prompt.replace("{intent}", intent)
    user_prompt = user_prompt.replace("{scenario_key}", scenario_key)
    user_prompt = user_prompt.replace("{examples_str}", examples_str)

    config = types.GenerateContentConfig(
        system_instruction=sys_instruction,
        response_mime_type="application/json",
        response_schema=DialogueResponse,
        temperature=generator_config["temperature"],
        safety_settings=generator_config["safety_settings"],
    )

    for attempt in range(3):
        try:
            response = gemini_client.models.generate_content(
                model=generator_model, contents=user_prompt, config=config
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
