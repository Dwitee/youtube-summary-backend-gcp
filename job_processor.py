import os
import uuid
from threading import Thread
from flask import request, jsonify
from transcriber import transcribe_with_whisper
from summarize import summarize_text

job_results = {}

def process_job(file_path, job_id):
    try:
        print(f"[DEBUG] Processing job {job_id}")
        transcript = transcribe_with_whisper(file_path)
        summary = summarize_text(transcript)
        job_results[job_id] = summary
        os.remove(file_path)
    except Exception as e:
        job_results[job_id] = f"Error: {str(e)}"

def submit_job_handler():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    job_id = str(uuid.uuid4())
    file_path = f"/tmp/{job_id}.mp3"
    file.save(file_path)
    Thread(target=process_job, args=(file_path, job_id)).start()
    return jsonify({"job_id": job_id})

def job_result_handler(job_id):
    if job_id in job_results:
        return jsonify({"summary": job_results[job_id]})
    return jsonify({"status": "processing"})