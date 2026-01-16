from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import io
import os
from pypdf import PdfReader
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs

app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.route("/", methods=["GET"])
def home():
    return "Study AI Backend is running"

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()

    if not data:
        return jsonify({"error": "No data received"}), 400

    # =======================
    # YOUTUBE CASE
    # =======================
    if "yt_url" in data:
        try:
            yt_url = data["yt_url"]
            parsed = urlparse(yt_url)

            if "youtube.com" in parsed.hostname:
                video_id = parse_qs(parsed.query).get("v", [None])[0]
            elif "youtu.be" in parsed.hostname:
                video_id = parsed.path.replace("/", "")
            else:
                return jsonify({"error": "Invalid YouTube URL"}), 400

            if not video_id:
                return jsonify({"error": "Video ID not found"}), 400

            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            text = " ".join([t["text"] for t in transcript])

            return analyze_text(text)

        except Exception:
            return jsonify({
                "error": "YouTube transcript not available for this video"
            }), 400

    # =======================
    # PDF CASE
    # =======================
    if "file_url" in data:
        try:
            res = requests.get(data["file_url"], timeout=15)
            reader = PdfReader(io.BytesIO(res.content))

            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""

            if not text.strip():
                return jsonify({"error": "No readable text in PDF"}), 400

            return analyze_text(text)

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "No valid input provided"}), 400


# =======================
# COMMON ANALYSIS FUNCTION
# =======================
def analyze_text(text):
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
