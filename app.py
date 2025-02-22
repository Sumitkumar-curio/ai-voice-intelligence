from fastapi import FastAPI, UploadFile, File
import whisper
from transformers import pipeline
import tempfile

app = FastAPI()

# Load Models
speech_model = whisper.load_model("base")
sentiment_analyzer = pipeline("sentiment-analysis")

@app.post("/analyze/")
async def analyze(audio: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        temp.write(await audio.read())
        temp_path = temp.name

    # Transcribe Speech
    transcription = speech_model.transcribe(temp_path)["text"]

    # Analyze Sentiment
    sentiment = sentiment_analyzer(transcription)

    return {
        "transcription": transcription,
        "sentiment": sentiment[0]
    }

# Run Server: uvicorn app:app --reload
