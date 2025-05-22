
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
