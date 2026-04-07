import os
import whisper
import pandas as pd
import torch

import torch, platform, sys

# Очистка кэша видеопамяти
#torch.cuda.empty_cache()

# Опционально: ограничьте использование памяти
torch.cuda.set_per_process_memory_fraction(0.8)  # использовать не более 80% памяти


print("PyTorch: ", torch.__version__)
print("CUDA available: ", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA device: ", torch.cuda.get_device_name(0))
    ##!nvidia-smi
else:
    print("Работаем на CPU - будет медленнее, это нормально.")

model = whisper.load_model("medium")

video_file = "audio.mp3"

result = model.transcribe(video_file, language = "ru")
print("Транскрипция:\n", result["text"])

transcribed = result["text"]


import requests, json, re
MODEL = "llama3"
SYSTEM_PROMPT = """
Ответь на русском. Ты - ии-ассистент, исполняющий роль экзаменатора. Твоя задача: проверить правильность ответа


Ответ выдай строго в следующем формате, без лишних слов!!!:
json={
        "file_id": string | null,
        "category": "обычный" | "подозрительный" | "мошеннический",
        "summary": string,
        "examples": [{"ts": "MM:SS", "quote": string, "note": string}],
        "indicators": [string],
        "risk_score": number
    }
    Дай только json. В summary и examples пиши на русском.
"""

prompt = f"{SYSTEM_PROMPT}\n\file_id: call10.wav\n\Транскрипт для анализа:\n{transcribed.strip()}"

resp = requests.post(
    "http://127.0.0.1:11434/api/generate",
    json={
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2}
    }
)

print(resp.json()["response"])