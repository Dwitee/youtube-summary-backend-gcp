from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, VideoUnavailable
import re
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import job_processor
from summarize import summarize_text
from transcriber import transcribe_with_whisper
import subprocess
import tempfile
import os
import whisper
import yt_dlp
from mindmap_generator import generate_mindmap_transformer, generate_mindmap_mistral, generate_mindmap_gemini
import hashlib
import json
import redis
from config import REDIS_URL
import os
from google.cloud import storage
cache_store = {}

# Initialize Redis client for summary persistence
redis_client = redis.from_url(REDIS_URL, decode_responses=True)


# Rename Whisper model variable to whisper_model
whisper_model = whisper.load_model("tiny", device="cuda")

# Add Zephyr model and tokenizer initialization
# from transformers import AutoModelForCausalLM, AutoTokenizer

# zephyr_model_name = "HuggingFaceH4/zephyr-7b-beta"
# zephyr_tokenizer = AutoTokenizer.from_pretrained(zephyr_model_name)
# zephyr_model = AutoModelForCausalLM.from_pretrained(zephyr_model_name, device_map="auto")


app = Flask(__name__)
# Google Cloud Storage setup
GCS_BUCKET = os.environ['GCS_BUCKET_NAME']
storage_client = storage.Client()
bucket = storage_client.bucket(GCS_BUCKET)
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
        file_content = file.read()
        file_hash = hashlib.md5(file_content).hexdigest()

        if file_hash in cache_store:
            print("✅ Returning cached summary.")
            return jsonify({"summary": cache_store[file_hash]})

        file.seek(0)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, file.filename)
            with open(file_path, "wb") as f:
                f.write(file_content)

            print("Transcribing uploaded file with Whisper...")
            full_text = transcribe_with_whisper(file_path)
            print("Transcription complete. Word count:", len(full_text.split()))

            if len(full_text.split()) > 400:
                full_text = " ".join(full_text.split()[:400])
                print("Transcript truncated to 400 words")

            summary = summarize_text(full_text)
            cache_store[file_hash] = summary
            print("Sending summary response:", summary)
            return jsonify({"summary": summary})
    except Exception as e:
        print("Upload summarization failed:", str(e))
        return jsonify({"error": f"Upload summarization failed: {str(e)}"}), 500



@app.route("/submit-job", methods=["POST"])
def submit_job():
    return job_processor.submit_job_handler()

# New route: /submit-video-to-summarize
@app.route("/submit-video-to-summarize", methods=["POST"])
def submit_video_to_summarize():
    """
    Enqueue a video summarization job.
    Expects JSON payload with 'id', 'title', 'thumbnailUrl', and 'videoUrl'.
    """
    data = request.get_json()
    print(f"[DEBUG] submit-video-to-summarize called with payload: {data}")
    # Basic validation
    if not data or not all(k in data for k in ("id", "title", "thumbnailUrl", "videoUrl")):
        return jsonify({"error": "Missing one of id, title, thumbnailUrl, videoUrl"}), 400
    # Delegate to job processor
    return job_processor.submit_video_to_summarize_handler()

@app.route("/job-result/<job_id>", methods=["GET"])
def job_result(job_id):
    return job_processor.job_result_handler(job_id)



@app.route("/generate-mindmap", methods=["POST"])
def generate_mindmap():
    # Use global Redis client
    r = redis_client
    import hashlib

    data = request.get_json()
    summary = data.get("summary", "").strip()
    model_type = data.get("model_type", "zephyr-gguf")  

    if not summary:
        return jsonify({"error": "Empty summary provided"}), 400

    cache_key = f"{model_type}_mindmap_" + hashlib.md5(summary.encode("utf-8")).hexdigest()
    cached = r.get(cache_key)
    if cached:
        print(f"[DEBUG] Returning cached mindmap key {cache_key} for model {model_type} ")
        return jsonify({"mindmap": json.loads(cached)})

    try:
        print(f"Generating mind map using model: {model_type}")  # Debug log
        model_dispatch = {
            "transformer": generate_mindmap_transformer,
            "mistral": generate_mindmap_mistral,
            "gemini": generate_mindmap_gemini
        }

        generator_fn = model_dispatch.get(model_type)
        if not generator_fn:
            return jsonify({"error": f"Unsupported model_type: {model_type}"}), 400

        mindmap_json = generator_fn(summary)
        r.set(cache_key, json.dumps(mindmap_json), ex=172800)
        return jsonify({"mindmap": mindmap_json})
    except Exception as e:
        print("Mind map generation failed:", str(e))
        return jsonify({"error": str(e)}), 500


# Serve a mindmap HTML file from /tmp
@app.route("/mindmap/<filename>")
def serve_mindmap(filename):
    file_path = os.path.join("/home/dwiteekrishnapanda/mindmaps", filename)
    if os.path.exists(file_path):
        return send_file(file_path, mimetype="text/html")
    else:
        return "File not found", 404
    


# New route: /upload-mindmap
from datetime import datetime

@app.route("/upload-mindmap", methods=["POST"])
def upload_mindmap():
    data = request.get_json()
    html_content = data.get("html", "")

    if not html_content:
        return jsonify({"error": "Missing 'html' content"}), 400

    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"mindmap_{timestamp}.html"
        output_dir = "/home/dwiteekrishnapanda/mindmaps"
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, filename)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"[DEBUG] Mindmap HTML saved to: {file_path}")
        return jsonify({"filename": filename})
    except Exception as e:
        print(f"[ERROR] Failed to save mindmap HTML: {e}")
        return jsonify({"error": f"Failed to save mindmap HTML: {str(e)}"}), 500

@app.route('/upload-thumb', methods=['POST'])
def upload_thumbnail():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    filename = file.filename  # expected "<id>.png"
    blob = bucket.blob(f'thumbnails/{filename}')
    blob.upload_from_file(file.stream, content_type=file.mimetype)
    thumb_url = blob.public_url
    print(f"[DEBUG] upload_thumbnail succeeded: {thumb_url}")  # Debug log
    return jsonify({"thumbUrl": thumb_url}), 200

@app.route('/upload-video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    filename = file.filename  # expected "<id>.webm"
    blob = bucket.blob(f'videos/{filename}')
    blob.upload_from_file(file.stream, content_type=file.mimetype)
    return jsonify({"videoUrl": blob.public_url}), 200


@app.route("/save-summary", methods=["POST"])
def save_summary():
    """
    Save summary entry as JSON under key summary:<id> in Redis.
    Expects JSON with at least 'id' field.
    """
    entry = request.get_json()
     # Do not persist thumbnail in Redis
    # entry.pop("thumbnail", None)
    summary_id = entry.get("id")
    if not summary_id:
        return jsonify({"error": "Missing 'id'"}), 400
    key = f"summary:{summary_id}"
    redis_client.set(key, json.dumps(entry))
    return jsonify({"status": "saved", "id": summary_id}), 201

@app.route("/list-summaries", methods=["GET"])
def list_summaries():
    """
    List all saved summary entries from Redis.
    """
    keys = redis_client.keys("summary:*")
    entries = []
    for key in keys:
        data = redis_client.get(key)
        entry = json.loads(data)
        # remove the thumbnail field if you don't want to ship it
        entry.pop("thumbnail", None)
        entries.append(entry)
    return jsonify(entries), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)