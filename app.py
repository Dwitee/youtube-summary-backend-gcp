

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, VideoUnavailable
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from summarize import summarize_text

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
    print("Extracted Video ID:", video_id)  # 🔍 Debug log

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
        print("Transcript truncated to 400 words")  # 🔍 Debug log

    try:
        summary = summarize_text(full_text)
        return jsonify({"summary": summary})
    except Exception as e:
        print("Summarization failed:", str(e))  # 🔍 Debug log
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
