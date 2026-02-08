from gtts import gTTS
import tempfile

def text_to_speech_with_gtts(text):
    tts = gTTS(text=text, lang="en", slow=False)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    return temp_file.name
