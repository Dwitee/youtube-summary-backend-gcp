from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, VideoUnavailable
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
import job_processor
from summarize import summarize_text
from transcriber import transcribe_with_whisper
import subprocess
import tempfile
import os
import whisper
import yt_dlp
from mindmap_generator import generate_mindmap_transformer, generate_mindmap_from_gguf


# Rename Whisper model variable to whisper_model
whisper_model = whisper.load_model("tiny", device="cuda")

# Add Zephyr model and tokenizer initialization
from transformers import AutoModelForCausalLM, AutoTokenizer

zephyr_model_name = "HuggingFaceH4/zephyr-7b-beta"
zephyr_tokenizer = AutoTokenizer.from_pretrained(zephyr_model_name)
zephyr_model = AutoModelForCausalLM.from_pretrained(zephyr_model_name, device_map="auto")

app = Flask(__name__)
CORS(app)

@app.route("/summarize-text", methods=["POST"])
def summarize():
    data = request.get_json()
    text = data.get("text", "").strip()

    print("Received text:", text[:100])  # Debug log
    if not text:
        return jsonify({"error": "Empty input"}), 400

    try:
        summary = summarize_text(text)
        print("Generated summary:", summary)
        return jsonify({"summary": summary})
    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 500


# New route: /summarize-url
@app.route("/summarize-url", methods=["POST"])
def summarize_url():
    data = request.get_json()
    url = data.get("url")

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    # Extract video ID from URL
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    if not match:
        return jsonify({"error": "Invalid YouTube URL"}), 400

    video_id = match.group(1)
    print("Extracted Video ID:", video_id)  #  Debug log

    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = " ".join([entry["text"] for entry in transcript_list])
    except VideoUnavailable:
        return jsonify({"error": "Video unavailable"}), 404
    except TranscriptsDisabled:
        return jsonify({"error": "Transcript is disabled for this video"}), 403
    except Exception as e:
        return jsonify({"error": f"Transcript fetch failed: {str(e)}"}), 500

    if len(full_text.strip().split()) == 0:
        return jsonify({"error": "Transcript is empty"}), 400

    # Truncate and summarize using existing summarize_text function
    if len(full_text.split()) > 400:
        full_text = " ".join(full_text.split()[:400])
        print("Transcript truncated to 400 words")  #  Debug log

    try:
        summary = summarize_text(full_text)
        return jsonify({"summary": summary})
    except Exception as e:
        print("Summarization failed:", str(e))  #  Debug log
        return jsonify({"error": str(e)}), 500

@app.route("/summarize-url-whisper", methods=["POST"])
def summarize_url_whisper():
    data = request.get_json()
    url = data.get("url")

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    print("Downloading audio from URL:", url)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_template = os.path.join(tmpdir, "audio.%(ext)s")
            final_audio_path = os.path.join(tmpdir, "audio.mp3")
            cookie_path = "youtube_cookies.txt"
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': output_template,
                'quiet': True,
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
            }

            if os.path.exists(cookie_path):
                print("✅ Using cookiefile for authentication.")
                ydl_opts['cookiefile'] = cookie_path
            else:
                print("⚠️ No cookiefile found. Proceeding without cookies.")

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            print("Audio downloaded, transcribing with Whisper...")
            print("Checking if file exists:", os.path.exists(final_audio_path))
            result = whisper_model.transcribe(final_audio_path)
            full_text = result["text"]
            print("Transcription complete. Word count:", len(full_text.split()))
    except Exception as e:
        print("Whisper transcription failed:", str(e))
        return jsonify({"error": f"Whisper transcription failed: {str(e)}"}), 500

    if len(full_text.split()) > 400:
        full_text = " ".join(full_text.split()[:400])
        print("Transcript truncated to 400 words")  # 🔍 Debug log

    try:
        summary = summarize_text(full_text)
        return jsonify({"summary": summary})
    except Exception as e:
        print("Summarization failed:", str(e))
        return jsonify({"error": str(e)}), 500




# New route: /summarize-upload
@app.route("/summarize-upload", methods=["POST"])
def summarize_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, file.filename)
            file.save(file_path)

            print("Transcribing uploaded file with Whisper...")
            full_text = transcribe_with_whisper(file_path)
            print("Transcription complete. Word count:", len(full_text.split()))

            if len(full_text.split()) > 400:
                full_text = " ".join(full_text.split()[:400])
                print("Transcript truncated to 400 words")

            summary = summarize_text(full_text)
            print("Sending summary response:", summary)
            return jsonify({"summary": summary})
    except Exception as e:
        print("Upload summarization failed:", str(e))
        return jsonify({"error": f"Upload summarization failed: {str(e)}"}), 500


@app.route("/submit-job", methods=["POST"])
def submit_job():
    return job_processor.submit_job_handler()

@app.route("/job-result/<job_id>", methods=["GET"])
def job_result(job_id):
    return job_processor.job_result_handler(job_id)



@app.route("/generate-mindmap", methods=["POST"])
def generate_mindmap():
    data = request.get_json()
    summary = data.get("summary", "").strip()
    model_type = data.get("model_type", "zephyr-gguf")  

    if not summary:
        return jsonify({"error": "Empty summary provided"}), 400

    try:
        print(f"Generating mind map using model: {model_type}")  # Debug log
        model_dispatch = {
            "transformer": generate_mindmap_transformer,
            "zephyr-gguf": generate_mindmap_from_gguf,
        }

        generator_fn = model_dispatch.get(model_type)
        if not generator_fn:
            return jsonify({"error": f"Unsupported model_type: {model_type}"}), 400

        mindmap_json = generator_fn(summary)
        return jsonify({"mindmap": mindmap_json})
    except Exception as e:
        print("Mind map generation failed:", str(e))
        return jsonify({"error": str(e)}), 500
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)