import os
import uuid
from threading import Thread
from flask import request, jsonify
from transcriber import transcribe_with_whisper
from summarize import summarize_text
import redis
import hashlib

r = redis.Redis(host='localhost', port=6379, db=0)

job_results = {}

def process_job(file_path, job_id, model_name):
    try:
        file_hash = None
        with open(file_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        cached_transcript = r.get(f"{file_hash}_transcript")
        cached_summary = r.get(f"{file_hash}_summary")
        if cached_summary and cached_transcript:
            print(f"[DEBUG] Cache hit for job {job_id}")
            job_results[job_id] = cached_summary.decode()
            os.remove(file_path)
            return
        print(f"[DEBUG] Processing job {job_id} with model: {model_name}")
        transcript = transcribe_with_whisper(file_path)
        summary = summarize_text(transcript, model_name)
        r.set(f"{file_hash}_transcript", transcript, ex=172800)
        r.set(f"{file_hash}_summary", summary, ex=172800)
        job_results[job_id] = summary
        os.remove(file_path)
    except Exception as e:
        job_results[job_id] = f"Error: {str(e)}"

def submit_job_handler():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    model_name = request.form.get("model_name", "t5-small")
    job_id = str(uuid.uuid4())
    file_path = f"/tmp/{job_id}.mp3"
    file.save(file_path)
    Thread(target=process_job, args=(file_path, job_id, model_name)).start()
    return jsonify({"job_id": job_id})

def job_result_handler(job_id):
    if job_id in job_results:
        return jsonify({"summary": job_results[job_id]})
    return jsonify({"status": "processing"})