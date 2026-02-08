import logging
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO
from groq import Groq
import os

logging.basicConfig(level=logging.INFO)

def record_audio(file_path="user_voice.wav", boost_db=10, timeout=10):
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        logging.info("Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        logging.info("Start speaking...")
        audio_data = recognizer.listen(source, timeout=timeout)

    wav_data = audio_data.get_wav_data()
    audio_segment = AudioSegment.from_wav(BytesIO(wav_data))
    audio_segment = audio_segment + boost_db
    audio_segment.export(file_path, format="wav")

    return file_path

def transcribe_with_groq(audio_filepath, stt_model="whisper-large-v3"):
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    with open(audio_filepath, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model=stt_model,
            file=audio_file,
            language="en"
        )

    return transcription.text
