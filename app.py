from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import io
import os

from pypdf import PdfReader
from openai import OpenAI
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound
)
from urllib.parse import urlparse, parse_qs

# --------------------
# App setup
# --------------------
app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --------------------
# Home
# --------------------
@app.route("/", methods=["GET"])
def home():
    return "Backend is live"

# --------------------
# Analyze endpoint
# --------------------
@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()

    if not data:
        return jsonify({"error": "No data received"}), 400

    # ==================================================
    # YOUTUBE CASE
    # ==================================================
    if "yt_url" in data:
        yt_url = data["yt_url"]

        try:
            # ---- extract video id ----
            parsed = urlparse(yt_url)

            if parsed.hostname in ["www.youtube.com", "youtube.com"]:
                video_id = parse_qs(parsed.query).get("v")
                if not video_id:
                    return jsonify({"error": "Invalid YouTube URL"}), 400
                video_id = video_id[0]

            elif parsed.hostname == "youtu.be":
                video_id = parsed.path.replace("/", "")

            else:
                return jsonify({"error": "Invalid YouTube URL"}), 400

            # ---- fetch transcript (NO yt-dlp, NO whisper) ----
            try:
                transcript = YouTubeTranscriptApi.get_transcript(
                    video_id,
                    languages=["en"]
                )
            except (TranscriptsDisabled, NoTranscriptFound):
                return jsonify({"error": "Transcript not available"}), 400

            transcript_text = " ".join([t["text"] for t in transcript])

            if not transcript_text.strip():
                return jsonify({"error": "Empty transcript"}), 400

            # ---- OpenAI prompt ----
            prompt = f"""
You are a strict study decision AI.

From the YouTube transcript below:
1. Pick max 3 topics to study TODAY
2. Clearly say what to IGNORE today
3. Be short and direct

TRANSCRIPT:
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
                "error": f"YouTube processing failed: {str(e)}"
            }), 500

    # ==================================================
    # PDF CASE
    # ==================================================
    if "file_url" in data:
        try:
            file_url = data["file_url"]

            r = requests.get(file_url, timeout=20)
            if r.status_code != 200:
                return jsonify({"error": "Failed to fetch PDF"}), 400

            reader = PdfReader(io.BytesIO(r.content))

            full_text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"

            if not full_text.strip():
                return jsonify({"error": "No readable text in PDF"}), 400

            prompt = f"""
You are a strict study decision AI.

From the PDF syllabus below:
1. Pick max 3 topics to study TODAY
2. Say what to IGNORE today
3. Be short and direct

PDF CONTENT:
{full_text[:12000]}
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
                "error": f"PDF processing failed: {str(e)}"
            }), 500

    return jsonify({"error": "No valid input"}), 400


# --------------------
# Run
# --------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
