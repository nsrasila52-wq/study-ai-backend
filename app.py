from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import io
import os
import json
from pypdf import PdfReader
from openai import OpenAI

app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.route("/", methods=["GET"])
def home():
    return "Backend is live"

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data received"}), 400

    content_text = ""

    # -------- PDF INPUT --------
    if "file_url" in data:
        try:
            r = requests.get(data["file_url"], timeout=20)
            if r.status_code != 200:
                return jsonify({"error": "Failed to fetch PDF"}), 400

            reader = PdfReader(io.BytesIO(r.content))
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    content_text += text + "\n"

        except Exception as e:
            return jsonify({"error": f"PDF processing failed: {str(e)}"}), 500

    # -------- IMAGE PLACEHOLDER --------
    elif "image_url" in data:
        content_text = "Extracted text from image"

    if not content_text.strip():
        return jsonify({"error": "No readable content"}), 400

    # -------- AI PROMPT (STRICT JSON) --------
    prompt = f"""
You are a strict study decision AI.

ONLY return valid JSON.
No markdown.
No extra text.

JSON FORMAT:
{{
  "important_topics": ["topic1", "topic2", "topic3"],
  "ignore_topics": ["topicA", "topicB"],
  "questions": ["question1", "question2", "question3"]
}}

CONTENT:
{content_text[:12000]}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    ai_text = response.choices[0].message.content.strip()

    # -------- JSON PARSE SAFE --------
    try:
        parsed = json.loads(ai_text)
    except Exception:
        return jsonify({
            "error": "AI did not return valid JSON",
            "raw_output": ai_text
        }), 500

    return jsonify({
        "result": {
            "important_topics": parsed.get("important_topics", []),
            "ignore_topics": parsed.get("ignore_topics", []),
            "questions": parsed.get("questions", [])
        }
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
