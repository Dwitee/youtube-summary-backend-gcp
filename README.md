

# ReMindmap Backend (GCP)

This repository contains the backend for the **ReMindMap** application, a tool designed to make video and audio media more accessible, especially for users with communication and comprehension difficulties such as aphasia.

The backend is powered by Flask and runs on a Google Cloud Platform (GCP) VM using Gunicorn for production deployment. It integrates Whisper, T5, Zephyr/Gemini, and LLaMA.cpp models for transcription, summarization, and mind map generation.

---

## Features

- Accepts uploaded video/audio files or YouTube URLs (first 7 minutes).
- Uses ASR models (Whisper) to transcribe speech.
- Summarizes video/audio content into timestamped chapters.
- Generates a structured, interactive mind map with narration and emojis.
- Text-to-speech support with controllable speed (0.25x, 0.5x, 1x).
- Stores metadata and generated outputs in Redis.
- Uploads and retrieves files from Google Cloud Storage (GCS).
- Cloud Logging for monitoring API performance and debugging.

---

##  Deployment

### Requirements

- Python 3.10+
- Redis (local or cloud)
- GCP service account with access to Cloud Storage and Vertex AI (optional)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Locally (for development)

```bash
python3 app.py
```

### Run in Production (Gunicorn)

```bash
gunicorn -w 4 -b 0.0.0.0:8080 app:app
```

---

## 🔌 API Endpoints

### `/submit`
Upload video/audio and start the summarization pipeline.

### `/download-youtube-and-submit`
Provide a valid YouTube URL, download the first 7 minutes, upload to GCS, and return metadata.

### `/list-summaries`
Returns a list of all completed video summaries.

### `/generate-mindmap`
Triggers mind map generation for a given summary.

For rest of the endpoints please refer app.py

---

##  Folder Structure

```
.
├── app.py                     # Main Flask app
├── job_processor.py          # Handles background processing logic
├── youtube_cookies.txt       # Optional for authenticating YouTube downloads
├── requirements.txt          # Python dependencies
├── mindmap_generator.py      # generate mindmaps
|── transcriber.py
|── summarizer.py
|── start.sh                  # ./start.sh can start the server in gunicorn
└── mindmaps/                 # Generated mind map HTML files
```

---

##  Notes

- This project uses cookie-based authentication for YouTube video downloads. Place your `youtube_cookies.txt` in the root directory if needed.
- To avoid rate limits or errors from YouTube, update your cookies regularly.

---

##  License

MIT License © Dwitee Krishna Panda