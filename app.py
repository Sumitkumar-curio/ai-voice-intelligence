import gradio as gr
from transformers import pipeline
import torch

# Load Whisper model and tokenizer
whisper_model = "openai/whisper-base"
device = "cuda" if torch.cuda.is_available() else "cpu"
asr = pipeline("automatic-speech-recognition", model=whisper_model, device=device)

# Load Sentiment Analysis model
sentiment_model = "cardiffnlp/twitter-roberta-base-sentiment-latest"
sentiment_analyzer = pipeline("sentiment-analysis", model=sentiment_model, device=device)

def transcribe_and_analyze(audio):
    # Transcribe audio to text
    transcription = asr(audio)["text"]
    
    # Analyze sentiment of the transcription
    sentiment = sentiment_analyzer(transcription)[0]
    
    return transcription, sentiment["label"], sentiment["score"]

# Define Gradio interface
interface = gr.Interface(
    fn=transcribe_and_analyze,
    inputs=gr.Audio(sources=["microphone", "upload"], type="filepath", label="Upload or Record Audio"),  # Added upload option
    outputs=[
        gr.Textbox(label="Transcription"),
        gr.Textbox(label="Sentiment"),
        gr.Number(label="Confidence Score")
    ],
    title="Real-Time Audio Transcription and Sentiment Analysis",
    description="This application transcribes audio input and analyzes the sentiment of the transcribed text.",
    live=True
)

if __name__ == "__main__":
    interface.launch()
