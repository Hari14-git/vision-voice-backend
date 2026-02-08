from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os

from brain_of_the_doctor import encode_image, analyze_image_with_query
from voice_of_the_user import transcribe_with_groq
from voice_of_the_doctor import text_to_speech_with_gtts

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = """
You are a helpful AI assistant. The user will share an image and ask a question about it (or about any topic).
Answer clearly and accurately based on what you see in the image and the user's question. You can answer questions about any field: documents, objects, nature, diagrams, screenshots, photos, etc.
Use simple, clear language. Structure your answer with short paragraphs. Do not use bullet points, numbers, symbols, or emojis unless the user's question clearly calls for them.
Do not mention that you are an AI or that you are analyzing an image. Respond naturally as a knowledgeable assistant.
"""

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/analyze")
async def analyze(
    audio: UploadFile = File(...),
    image: UploadFile = File(...)
):
    audio_path = f"/tmp/{audio.filename}"
    image_path = f"/tmp/{image.filename}"

    with open(audio_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)

    with open(image_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    speech_text = transcribe_with_groq(audio_path)

    response_text = analyze_image_with_query(
        query=SYSTEM_PROMPT + " " + speech_text,
        encoded_image=encode_image(image_path),
        model="meta-llama/llama-4-maverick-17b-128e-instruct"
    )

    voice_path = text_to_speech_with_gtts(response_text)

    with open(voice_path, "rb") as f:
        audio_base64 = f.read().encode("base64")

    return {
        "speech_text": speech_text,
        "response_text": response_text,
        "audio_base64": audio_base64
    }
