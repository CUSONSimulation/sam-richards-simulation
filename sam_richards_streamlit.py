# sam_richards_streamlit.py (Streamlit Cloud Compatible)

import openai
import tempfile
import os
from dotenv import load_dotenv
import streamlit as st
from datetime import datetime

# Load API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Config
VOICE = "echo"

# Character Persona
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

# Transcribe uploaded audio
def transcribe_audio(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name
    with open(tmp_path, "rb") as f:
        transcript = openai.audio.transcriptions.create(model="whisper-1", file=f)
    return transcript.text

# Get GPT-4 response
def get_sam_response(messages):
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=messages
    )
    return response.choices[0].message.content

# Generate TTS response
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
st.caption("🔁 Updated: Streamlit Cloud compatible (audio file upload only)")
st.write("Upload a short `.wav` audio clip of your message to Sam Richards.")

if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SAM_PROMPT}]

if 'transcript' not in st.session_state:
    st.session_state.transcript = []

uploaded_file = st.file_uploader("📤 Upload your .wav file", type=["wav"])

if uploaded_file is not None:
    user_input = transcribe_audio(uploaded_file)
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

# Download transcript
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

