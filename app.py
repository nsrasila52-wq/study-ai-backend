from flask import Flask, request, jsonify
from flask_cors import CORS
import requests, io, os, subprocess
from pypdf import PdfReader
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs

app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.route("/", methods=["GET"])
def home():
    return "Study AI Backend running"

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data"}), 400

    if "yt_url" in data:
        return handle_youtube(data["yt_url"])

    if "file_url" in data:
        return handle_pdf(data["file_url"])

    return jsonify({"error": "Invalid input"}), 400


# ===================== YOUTUBE =====================
def handle_youtube(yt_url):
    try:
        parsed = urlparse(yt_url)

        if "youtube.com" in parsed.hostname:
            video_id = parse_qs(parsed.query).get("v", [None])[0]
        elif "youtu.be" in parsed.hostname:
            video_id = parsed.path.replace("/", "")
        else:
            return jsonify({"error": "Invalid YouTube URL"}), 400

        # 1️⃣ Try captions first
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            text = " ".join(t["text"] for t in transcript)
        except Exception:
            # 2️⃣ Whisper fallback
            text = whisper_transcribe(yt_url)

        return analyze_text(text)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def whisper_transcribe(yt_url):
    audio_file = "audio.mp3"

    subprocess.run(
        "yt-dlp -f bestaudio -x --audio-format mp3 -o audio.mp3 " + yt_url,
        shell=True,
        check=True
    )

    with open(audio_file, "rb") as f:
        transcript = client.audio.transcriptions.create(
            file=f,
            model="gpt-4o-transcribe"
        )

    os.remove(audio_file)
    return transcript.text


# ===================== PDF =====================
def handle_pdf(file_url):
    res = requests.get(file_url, timeout=15)
    reader = PdfReader(io.BytesIO(res.content))
    text = "".join(page.extract_text() or "" for page in reader.pages)
    return analyze_text(text)


# ===================== GPT =====================
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
