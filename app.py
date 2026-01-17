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

    if "file_url" in data:
        try:
            r = requests.get(data["file_url"], timeout=20)
            reader = PdfReader(io.BytesIO(r.content))
            for page in reader.pages:
                if page.extract_text():
                    content_text += page.extract_text() + "\n"
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    if not content_text.strip():
        return jsonify({"error": "No readable content"}), 400

    prompt = f"""
You are a strict study AI.

Return ONLY valid JSON in this exact format.
Do not add extra text.

{{
  "important_topics": [max 3 items],
  "ignore_topics": [2–3 items],
  "questions": [2–5 questions]
}}

Rules:
- important_topics and ignore_topics MUST NOT be empty
- Be specific
- No placeholders

CONTENT:
{content_text[:12000]}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    try:
        ai_json = response.choices[0].message.content
        parsed = json.loads(ai_json)
    except:
        return jsonify({"error": "AI did not return valid JSON"}), 500

    return jsonify({
        "result": {
            "important_topics": parsed["important_topics"],
            "ignore_topics": parsed["ignore_topics"],
            "questions": parsed["questions"]
        }
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
