import gradio as gr
import requests

def process_audio(audio):
    files = {"audio": audio}
    response = requests.post("http://127.0.0.1:8000/analyze/", files=files)

    if response.status_code == 200:
        data = response.json()
        return data["transcription"], data["sentiment"]
    else:
        return "Error", "Error"

gr.Interface(
    fn=process_audio,
    inputs="audio",
    outputs=["text", "text"]
).launch(share=True)
