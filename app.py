import streamlit as st
import whisper
from textblob import TextBlob
import tempfile
import os

st.set_page_config(page_title="Audio Transcription")

st.title("🎙 Audio Transcription App")

uploaded_file = st.file_uploader(
    "Upload Audio File",
    type=["mp3", "wav", "m4a", "ogg"]
)

if uploaded_file is not None:

    st.audio(uploaded_file)

    st.write("Loading Whisper model...")

    model = whisper.load_model("base")

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_audio_path = tmp_file.name

    st.write("Transcribing audio...")

    result = model.transcribe(temp_audio_path)

    st.success("Transcription Completed ✅")

    st.subheader("Transcript")

    st.write(result["text"])

    # Sentiment Analysis
    blob = TextBlob(result["text"])

    sentiment = blob.sentiment.polarity

    st.subheader("Sentiment Analysis Result")

    st.write(f"Sentiment Score: {sentiment}")

    if sentiment > 0.5:
        st.success("Highly Positive 🙂")

    elif sentiment > 0:
        st.success("Slightly Positive 😊")

    elif sentiment < -0.5:
        st.error("Highly Negative 😠")

    elif sentiment < 0:
        st.warning("Slightly Negative 😕")

    else:
        st.info("Neutral 😐")

    os.remove(temp_audio_path)
