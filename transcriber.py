import whisper
import os
import torch

# Determine device and load the Whisper model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[DEBUG] Loading Whisper model on device: {device}")
model = whisper.load_model("tiny", device=device)  # or "base", "small", etc.

def transcribe_with_whisper(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"[DEBUG] Transcribing file: {file_path}")
    result = model.transcribe(file_path)
    print(f"[DEBUG] Whisper raw result object: {result}")
    transcript = result["text"]
    print(f"[DEBUG] Transcription complete. Word count: {len(transcript.split())}")
    return transcript