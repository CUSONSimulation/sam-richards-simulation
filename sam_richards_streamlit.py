# sam_richards_streamlit.py

import openai
import sounddevice as sd
import numpy as np
import queue
import tempfile
import os
import time
import wave
from dotenv import load_dotenv
import streamlit as st
from datetime import datetime

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Config
VOICE = "echo"
SAMPLE_RATE = 16000
CHANNELS = 1
DURATION = 5  # seconds per recording
q = queue.Queue()

# Character profile
SAM_PROMPT = """
You are Sam Richards, the Operations Manager of a County Corrections Facility. You’ve been in this role for 14 years. 
You are skeptical, resistant to change, and strongly protective of your staff, routine, and facility operations.

You are being asked to support a flu vaccination program for incarcerated individuals, and you are highly resistant to this.

Respond with a skeptical, often dismissive tone. Interrupt frequently, cite operational barriers, and emphasize staffing, budget, and safety issues. 
Mention past failures like TB testing and mental health screenings. Downplay the importance of flu shots. 
Use phrases like: "We've always done it this way", "This isn't going to work here", "That's a lawsuit waiting to happen", etc.

Stay completely in character. Do not assist or make it easy. You only reluctantly agree to 'review' proposals if heavily pressured.

You are not an AI. You are Sam Richards. Respond to the student nurse naturally, using realistic conversational language.
"""

# Audio recording
def record_audio():
    def callback(indata, frames, time, status):
        if status:
            st.warning(f"⚠️ {status}")
        q.put(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=callback):
        st.info("🎤 Speak now...")
        audio = np.empty((0, CHANNELS), dtype=np.float32)
        start_time = time.time()
        while time.time() - start_time < DURATION:
            audio = np.vstack((audio, q.get()))
        return audio

def save_audio_to_wav(audio_data):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(temp_file.name, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())
    return temp_file.name

# Whisper STT
def transcribe_audio(file_path):
    with open(file_path, "rb") as f:
        transcript = openai.audio.transcriptions.create(model="whisper-1", file=f)
    return transcript.text

# GPT response
def get_sam_response(messages):
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=messages
    )
    return response.choices[0].message.content

# TTS playback
def speak_text(text):
    response = openai.audio.speech.create(
        model="tts-1",
        voice=VOICE,
        input=text
    )
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        f.write(response.content)
        return f.name

# Streamlit UI
st.title("🧍 Sam Richards Corrections Simulation")
st.write("Speak into your microphone and convince Sam Richards to support a flu vaccination program.")

if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SAM_PROMPT}]

if 'transcript' not in st.session_state:
    st.session_state.transcript = []

if st.button("Record Response"):
    audio = record_audio()
    wav_file = save_audio_to_wav(audio)
    user_input = transcribe_audio(wav_file)
    st.markdown(f"**You said:** {user_input}")
    st.session_state.transcript.append(f"Nurse: {user_input}")

    st.session_state.messages.append({"role": "user", "content": user_input})
    sam_reply = get_sam_response(st.session_state.messages)
    st.session_state.messages.append({"role": "assistant", "content": sam_reply})
    st.session_state.transcript.append(f"Sam Richards: {sam_reply}")

    st.markdown(f"**Sam Richards:** {sam_reply}")
    audio_path = speak_text(sam_reply)
    audio_file = open(audio_path, 'rb')
    st.audio(audio_file.read(), format='audio/mp3')

# Downloadable transcript
if st.session_state.transcript:
    full_transcript = "\n".join(st.session_state.transcript)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    transcript_filename = f"transcript_{timestamp}.txt"
    st.download_button(
        label="📄 Download Transcript for Evaluation",
        data=full_transcript,
        file_name=transcript_filename,
        mime="text/plain"
    )

