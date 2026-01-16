from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from pypdf import PdfReader
import io
import os
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi
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

    # ================== YOUTUBE ==================
    if "yt_url" in data:
        yt_url = data["yt_url"]

        try:
            parsed_url = urlparse(yt_url)

            if "youtube.com" in parsed_url.hostname:
                video_id = parse_qs(parsed_url.query).get("v", [None])[0]
            elif "youtu.be" in parsed_url.hostname:
                video_id = parsed_url.path.replace("/", "")
            else:
                return jsonify({"error": "Invalid YouTube URL"}), 400

            if not video_id:
                return jsonify({"error": "Video ID not found"}), 400

            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            transcript_text = " ".join([i["text"] for i in transcript])

            prompt = f"""
You are a strict study decision AI.

From the syllabus below:
1. Pick max 3 topics to study TODAY
2. Say what to IGNORE today
3. Be short and direct

SYLLABUS:
{transcript_text[:12000]}
"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )

            return jsonify({
                "result": response.choices[0].message.content
            })

        except Exception as e:
            return jsonify({
                "error": "Transcript not available for this video",
                "details": str(e)
            }), 400

    # ================== PDF ==================
    if "file_url" in data:
        try:
            res = requests.get(data["file_url"], timeout=10)
            reader = PdfReader(io.BytesIO(res.content))
            text = "".join(page.extract_text() or "" for page in reader.pages)

            prompt = f"""
You are a strict study decision AI.

From the syllabus below:
1. Pick max 3 topics to study TODAY
2. Say what to IGNORE today
3. Be short and direct

SYLLABUS:
{text[:12000]}
"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )

            return jsonify({
                "result": response.choices[0].message.content
            })

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "No valid input"}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
