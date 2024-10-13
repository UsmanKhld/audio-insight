import pyaudio 
import wave 
import time
import os 
from groq import Groq
import numpy as np
import streamlit as st
from podcast.speech_to_text import audio_to_text
from datetime import datetime
from podcast.embedding import store_embeddings
from langchain.docstore.document import Document

# Audio stream parameters
CHUNK = 8192
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
DURATION = 5

# List to store the chunkz of text
transcription_chunks = list()

# Number of samples for 10 seconds
NUM_SAMPLES = RATE * DURATION  # 160,000 samples for 10 seconds
NUM_BYTES = NUM_SAMPLES * 2  # 2 bytes per sample for 16-bit audio

# Ensure output directories exist
mp3_file_folder = "uploaded_files"
mp3_chunk_folder = "chunks"
os.makedirs(mp3_file_folder, exist_ok=True)
os.makedirs(mp3_chunk_folder, exist_ok=True)

API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key = API_KEY)
model = 'whisper-large-v3'

# Global variable to control recording
if "recording" not in st.session_state:
    st.session_state.recording = False
if "docsearch" not in st.session_state:
    st.session_state.docsearch = None

def save_chunk_wav(audio_chunk, chunk_number):
    filename = os.path.join(mp3_chunk_folder, f"chunk_{chunk_number}_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav")
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pyaudio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(audio_chunk.tobytes())
    return filename

def split_text_into_chunks(text, word_limit=4096):
    words = text.split()
    for i in range(0, len(words), word_limit):
        yield " ".join(words[i:i + word_limit])

def start_recording():
    global recording 
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    st.write("Recording...")
    chunk_number = 0

    st.write("Live Transcription: ")
    full_text = ""
    while st.session_state.recording:
        data = stream.read(NUM_BYTES, exception_on_overflow=False)
        audio_chunk = np.frombuffer(data, dtype=np.int16)

        chunk_path = save_chunk_wav(audio_chunk, chunk_number)
        chunk_number+=1

        transcription = audio_to_text(chunk_path)
        full_text += transcription
        if(len(full_text) > 4096):
            text_chunks = split_text_into_chunks(full_text)
            for i, chunk in enumerate(text_chunks):
                transcription_chunks.append(chunk)
            full_text = ""

        st.write(transcription)
        time.sleep(0.01)
        print(transcription_chunks)
    
    stream.stop_stream()
    stream.close()
    p.terminate()

def start_transcription():
    st.session_state.recording = True
    start_recording()

def stop_transcription():
    """Function to stop recording."""
    st.session_state.recording = False 
    st.write("Recording stopped.")
    documents = [Document(page_content=chunk) for chunk in transcription_chunks]
    st.session_state.docsearch = store_embeddings(documents)