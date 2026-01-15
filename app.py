from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from pypdf import PdfReader
import io
import os
from openai import OpenAI

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

    if "yt_url" in data and data["yt_url"]:
        return jsonify({"result": "YouTube link received successfully."})

    if "file_url" in data and data["file_url"]:
        try:
            pdf_bytes = requests.get(data["file_url"]).content
            reader = PdfReader(io.BytesIO(pdf_bytes))

            text = ""
            for page in reader.pages:
                if page.extract_text():
                    text += page.extract_text() + "\n"

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

            return jsonify({"result": response.choices[0].message.content})

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "No valid input"}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
