from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from pypdf import PdfReader
import io
import os
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from urllib.parse import urlparse, parse_qs

app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.route("/", methods=["GET"])
def home():
    return "Backend is live. Use POST /analyze"

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data received"}), 400

    # YouTube case
    if data.get("yt_url"):
        yt_url = data["yt_url"]
        try:
            parsed_url = urlparse(yt_url)
            if parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
                video_id = parse_qs(parsed_url.query).get("v")
                if not video_id:
                    return jsonify({"error": "Invalid YouTube URL"}), 400
                video_id = video_id[0]
            elif parsed_url.hostname == "youtu.be":
                video_id = parsed_url.path[1:]
            else:
                return jsonify({"error": "Invalid YouTube URL"}), 400

            try:
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                transcript_text = " ".join([t["text"] for t in transcript_list])
            except (TranscriptsDisabled, NoTranscriptFound):
                return jsonify({"error": "Transcript not available for this video"}), 400

            if not transcript_text.strip():
                return jsonify({"error": "Transcript is empty"}), 400

            prompt = f"""
You are a strict study decision AI.

From the syllabus below (YouTube transcript):
1. Pick max 3 topics to study TODAY
2. Say what to IGNORE today
3. Be short and direct

SYLLABUS:
{transcript_text[:12000]}
"""
            ai_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )

            return jsonify({"result": ai_response.choices[0].message.content})

        except Exception as e:
            return jsonify({"error": f"Failed to process YouTube link: {str(e)}"}), 500

    # PDF case
    if data.get("file_url"):
        try:
            file_url = data["file_url"]
            response = requests.get(file_url, timeout=15)
            content_type = response.headers.get("Content-Type", "")
            if "pdf" not in content_type.lower():
                return jsonify({"error": "Only PDF files are supported."}), 400

            pdf_bytes = response.content
            if len(pdf_bytes) < 1000:
                return jsonify({"error": "PDF file is empty or corrupted."}), 400

            reader = PdfReader(io.BytesIO(pdf_bytes))
            text = "".join([page.extract_text() or "" for page in reader.pages])

            if not text.strip():
                return jsonify({"error": "No readable text found in PDF."}), 400

            prompt = f"""
You are a strict study decision AI.

From the syllabus below:
1. Pick max 3 topics to study TODAY
2. Say what to IGNORE today
3. Be short and direct

SYLLABUS:
{text[:12000]}
"""
            ai_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )

            return jsonify({"result": ai_response.choices[0].message.content})

        except Exception as e:
            return jsonify({"error": f"Failed to process PDF: {str(e)}"}), 500

    return jsonify({"error": "No valid input"}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
