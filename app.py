from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs

app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    yt_url = data.get("yt_url")

    if not yt_url:
        return jsonify({"error": "yt_url required"}), 400

    # ✅ Extract video ID (same as video)
    parsed = urlparse(yt_url)

    if "youtube.com" in parsed.netloc:
        video_id = parse_qs(parsed.query).get("v", [None])[0]
    elif "youtu.be" in parsed.netloc:
        video_id = parsed.path[1:]
    else:
        return jsonify({"error": "Invalid YouTube URL"}), 400

    if not video_id:
        return jsonify({"error": "Video ID not found"}), 400

    # ✅ Get transcript (VIDEO METHOD)
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
    except Exception as e:
        return jsonify({"error": f"Transcript error: {str(e)}"}), 400

    transcript_text = " ".join([t["text"] for t in transcript])

    # ✅ OpenAI call (MODERN + VIDEO STYLE PROMPT)
    prompt = f"""
You are a study assistant.

From this YouTube transcript:
1. Pick max 3 important topics to study today
2. Mention what can be ignored today
3. Be direct and short

Transcript:
{transcript_text[:12000]}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return jsonify({
        "result": response.choices[0].message.content
    })


if __name__ == "__main__":
    app.run(port=10000)
