from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import subprocess
from openai import OpenAI

app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.route("/", methods=["GET"])
def home():
    return "Backend is live (Whisper enabled)"

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()

    if not data or "yt_url" not in data:
        return jsonify({"error": "yt_url required"}), 400

    yt_url = data["yt_url"]
    audio_file = "audio.mp3"

    try:
        # =======================
        # STEP 1: Download audio
        # =======================
        subprocess.run(
            [
                "yt-dlp",
                "-f", "bestaudio",
                "--extract-audio",
                "--audio-format", "mp3",
                "-o", audio_file,
                yt_url
            ],
            check=True
        )

        # =======================
        # STEP 2: Whisper STT
        # =======================
        with open(audio_file, "rb") as f:
            transcript = client.audio.transcriptions.create(
                file=f,
                model="gpt-4o-transcribe"
            )

        transcript_text = transcript.text

        if not transcript_text.strip():
            return jsonify({"error": "Empty transcript"}), 400

        # =======================
        # STEP 3: GPT analysis
        # =======================
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
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(audio_file):
            os.remove(audio_file)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
