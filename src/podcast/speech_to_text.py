import os 
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

client = Groq(os.getenv("GROQ_API_KEY"))
model = 'whisper-large-v3'

def audio_to_text(filepath, translate_to_english):
    with open(filepath,'rb') as file:
        if translate_to_english:
            translation = client.audio.translations.create(
                file=(filepath, file.read()),
                model='whisper-large-v3'
            )
            return translation.text 
        else:
            transcription = client.audio.transcriptions.create(
                file=(filepath, file.read()),
                model='whisper-large-v3'
            )
            return transcription.text
    