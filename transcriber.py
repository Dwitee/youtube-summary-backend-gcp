import whisper
import os

# Load the Whisper model once
model = whisper.load_model("tiny")  # or "base", "small", etc.

def transcribe_with_whisper(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"[DEBUG] Transcribing file: {file_path}")
    result = model.transcribe(file_path)
    transcript = result["text"]
    print(f"[DEBUG] Transcription complete. Word count: {len(transcript.split())}")
    return transcript