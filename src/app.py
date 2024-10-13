import os
import openai
import streamlit as st
from moviepy.editor import AudioFileClip
from dotenv import load_dotenv
from podcast.speech_to_text import audio_to_text
from podcast.embedding import store_embeddings
from podcast.question_answer import query_vector_database
from langchain.docstore.document import Document
from live_audio_transcription.stream_audio import start_transcription, stop_transcription

# custom imports
from uiLayouts import *

# Set page title and icon
uiHeroSection()
# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Ensure the OPENAI_API_KEY is loaded
if OPENAI_API_KEY is None:
    st.error("API key not found. Please set the OPENAI_API_KEY in your .env file.")
    st.stop()

# Initialize OpenAI client
openai.api_key = OPENAI_API_KEY

# Ensure output directories exist
mp3_file_folder = "uploaded_files"
mp3_chunk_folder = "chunks"
os.makedirs(mp3_file_folder, exist_ok=True)
os.makedirs(mp3_chunk_folder, exist_ok=True)

load_lottieurl("https://lottie.host/c328b028-512c-4d26-bfb4-3eadb686289a/zidbWrQeAG.json")
with st.sidebar:
        uiSidebarWorkingInfo()
        uiSidebarInfo()

# Upload audio file
uploaded_file = st.file_uploader("Upload an MP3 file", type="mp3")

# Session state to store the last processed file to avoid reprocessing
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None
if "transcriptions" not in st.session_state:
    st.session_state.transcriptions = []
if "docsearch" not in st.session_state:
    st.session_state.docsearch = None

# Function to split the transcription into smaller chunks
def split_into_chunks(text, chunk_size=4096):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i + chunk_size])

# Function to use OpenAI's GPT-4 model for generating a response
def gpt4_chat_completion(prompt):
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

# When a new file is uploaded, reset the session state for transcriptions and embeddings
if uploaded_file is not None:
    # Check if the new file is different from the last processed file
    if uploaded_file.name != st.session_state.last_uploaded_file:
        st.session_state.transcriptions = []  # Reset transcriptions
        st.session_state.docsearch = None  # Reset embeddings
        st.session_state.last_uploaded_file = uploaded_file.name  # Update the last file

    # Save the uploaded file
    filepath = os.path.join(mp3_file_folder, uploaded_file.name)
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Optional: To save and chunk the audio file
    audio = AudioFileClip(filepath)
    chunk_length = 60  # seconds

    # Process and transcribe each chunk only if not already done
    if not st.session_state.transcriptions:
        for start in range(0, int(audio.duration), chunk_length):
            end = min(start + chunk_length, int(audio.duration))
            audio_chunk = audio.subclip(start, end)
            chunk_filename = os.path.join(mp3_chunk_folder, f"chunk_{start}.mp3")
            audio_chunk.write_audiofile(chunk_filename)

            # Process and transcribe each chunk using the speech-to-text function
            transcription = audio_to_text(chunk_filename)
            st.session_state.transcriptions.append(transcription)  # Collect transcriptions

        # Combine all transcriptions into a single text
        combined_transcription = " ".join(st.session_state.transcriptions)
        
        # Split the combined transcription into smaller chunks
        transcription_chunks = list(split_into_chunks(combined_transcription))

        # Create Document objects for each chunk and store embeddings
        documents = [Document(page_content=chunk) for chunk in transcription_chunks]
        st.session_state.docsearch = store_embeddings(documents)

        st.write(f"Transcription: {combined_transcription[:500]}...")  # Show the first 500 characters

# User query
user_question = st.text_input("Ask a question about the podcast")
if user_question and st.session_state.docsearch:
    # Query all stored vectors for relevant chunks
    relevant_transcripts = query_vector_database(st.session_state.docsearch, user_question)
    
    # Generate a prompt for the GPT-4 model using the relevant transcripts
    prompt = f"You are a helpful person whos job is to give answers from this and only this transcript:\n\n{relevant_transcripts}\n\nDo not answer based on any other knowledge, if the transcript does not contain information on the question, reply that it is not mentioned. Here is the question: Question: {user_question}"
    
    # Get response from GPT-4
    response = gpt4_chat_completion(prompt)
    
    st.write(f"Response: {response}")
    # Clear the input box by resetting the session state

    #Live Transcription
st.write("Start recording and view live transcription. Stop recording when done.")

    #Start/Stop button logic
if st.button("Stop Recording"):
    stop_transcription()

if st.button("Start Recording"):
    start_transcription()
    
