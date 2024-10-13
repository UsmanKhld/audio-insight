import pyaudio 
import wave 
import requests
import time
import os 
from moviepy.editor import AudioFileClip
from groq import Groq
from dotenv import load_dotenv
import json
import numpy as np
import streamlit as st
from podcast.speech_to_text import audio_to_text
from datetime import datetime
import threading

# Audio stream parameters
CHUNK = 8192
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
DURATION = 5

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

def save_chunk_wav(audio_chunk, chunk_number):
    filename = os.path.join(mp3_chunk_folder, f"chunk_{chunk_number}_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav")
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pyaudio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(audio_chunk.tobytes())
    return filename

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

    while st.session_state.recording:
        data = stream.read(NUM_BYTES, exception_on_overflow=False)
        audio_chunk = np.frombuffer(data, dtype=np.int16)

        chunk_path = save_chunk_wav(audio_chunk, chunk_number)
        chunk_number+=1

        transcription = audio_to_text(chunk_path)
        st.write(f"Live Transcription: {transcription}")
        time.sleep(0.01)
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    st.write("Recording stopped.")

def start_transcription():
    st.session_state.recording = True
    start_recording()

def stop_transcription():
    """Function to stop recording."""
    st.session_state.recording = False 