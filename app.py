from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from pypdf import PdfReader
import io
import os
from openai import OpenAI

app = Flask(__name__)
CORS(app)

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@app.route("/", methods=["GET"])
def home():
    return "Backend is live. Use POST /analyze"


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()

    if not data:
        return jsonify({"error": "No data received"}), 400

    # -------------------------
    # YouTube case
    # -------------------------
    if data.get("yt_url"):
        return jsonify({
            "result": "YouTube link received successfully."
        })

    # -------------------------
    # PDF file case
    # -------------------------
    if data.get("file_url"):
        try:
            file_url = data["file_url"]

            # Download file safely
            response = requests.get(file_url, stream=True, timeout=15)
            content_type = response.headers.get("Content-Type", "")

            # Allow only PDFs
            if "pdf" not in content_type.lower():
                return jsonify({
                    "error": "Only PDF files are supported right now."
                }), 400

            pdf_bytes = response.content

            if len(pdf_bytes) < 1000:
                return jsonify({
                    "error": "PDF file is corrupted or empty."
                }), 400

            reader = PdfReader(io.BytesIO(pdf_bytes))

            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

            if not text.strip():
                return jsonify({
                    "error": "No readable text found in this PDF."
                }), 400

            # -------------------------
            # OpenAI prompt
            # -------------------------
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
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            return jsonify({
                "result": ai_response.choices[0].message.content
            })

        except Exception as e:
            return jsonify({
                "error": f"Failed to process PDF: {str(e)}"
            }), 500

    return jsonify({"error": "No valid input"}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
