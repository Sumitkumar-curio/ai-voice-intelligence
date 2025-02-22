import streamlit as st
import requests

st.title("AI Voice Intelligence for Customer Support")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_file:
    files = {"audio": uploaded_file.getvalue()}
    response = requests.post("http://127.0.0.1:8000/analyze/", files=files)

    if response.status_code == 200:
        data = response.json()
        st.write("### Transcription:")
        st.write(data["transcription"])
        st.write("### Sentiment Analysis:")
        st.write(data["sentiment"])
