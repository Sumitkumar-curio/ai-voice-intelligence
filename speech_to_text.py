import whisper

# Load Whisper Model
model = whisper.load_model("base")

def transcribe_audio(audio_file):
    result = model.transcribe(audio_file)
    return result["text"]

# Test
if __name__ == "__main__":
    transcript = transcribe_audio("sample_audio.wav")
    print("Transcription:", transcript)
