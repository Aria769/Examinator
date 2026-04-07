!apt-get update && apt-get install -y libsndfile1 ffmpeg
!pip install Cython packaging
#!pip install git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]

# Установить через conda когда возможно
#!conda install -c conda-forge numba cudatoolkit=11.8
!pip install nemo_toolkit['asr']
# !pip uninstall numba numba-cuda cuda-python -y
# !pip install "numba<0.62.0" "numba-cuda<0.12.0" "cuda-python<13.0.0"
# !pip install nemo_toolkit['asr'] --upgrade

!pip install huggingface_hub
#!pip install cuda-python

# !pip install --upgrade pip
!pip install git+https://github.com/openai/whisper.git
!pip install pandas

!pkill -f ollama

!curl -fsSL https:/ollama.com/install.sh | sh
!nohup ollama serve > /dev/null 2>&1 &
!sleep 10


!ollama pull llama3


import os
import whisper
import pandas as pd
import torch

import torch, platform, sys

print("PyTorch: ", torch.__version__)
print("CUDA available: ", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA device: ", torch.cuda.get_device_name(0))
    ##!nvidia-smi
else:
    print("Работаем на CPU - будет медленнее, это нормально.")

model = whisper.load_model("large")

from google.colab import files
uploaded = files.upload()

result = model.transcribe(video_file, language = "ru")
print("Транскрипция:\n", result["text"])

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

prompt = f"{SYSTEM_PROMPT}\n\file_id: call10.wav\n\Транскрипт для анализа:\n{TRANSCRIPT_FULL.strip()}\nДиаризация голоса:\n{DIARIZED_FULL.strip()}"

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