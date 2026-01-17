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

# --------------------
# Home
# --------------------
@app.route("/", methods=["GET"])
def home():
    return "Backend is live"

# --------------------
# Analyze PDF / Image
# --------------------
@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data received"}), 400

    content_text = ""

    # ---------- PDF ----------
    if "file_url" in data:
        try:
            r = requests.get(data["file_url"], timeout=20)
            reader = PdfReader(io.BytesIO(r.content))
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    content_text += text + "\n"
        except Exception as e:
            return jsonify({"error": f"PDF error: {str(e)}"}), 500

    # ---------- IMAGE (NEW) ----------
    elif "image_url" in data:
        try:
            vision_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract all readable study-related text from this image."},
                            {
                                "type": "image_url",
                                "image_url": {"url": data["image_url"]}
                            }
                        ]
                    }
                ]
            )

            content_text = vision_response.choices[0].message.content

        except Exception as e:
            return jsonify({"error": f"Image error: {str(e)}"}), 500

    # ---------- NO CONTENT ----------
    if not content_text or not content_text.strip():
        return jsonify({"error": "No readable content"}), 400

    # ---------- AI ANALYSIS ----------
    prompt = f"""
You are a strict study AI.

Return ONLY valid JSON in this exact format.
Do not add extra text.

{{
  "important_topics": ["topic1", "topic2", "topic3"],
  "ignore_topics": ["topicA", "topicB"],
  "questions": ["question1", "question2", "question3"]
}}

Rules:
- important_topics and ignore_topics MUST NOT be empty
- Be specific
- No placeholders
- Plain text only

CONTENT:
{content_text[:12000]}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    try:
        ai_json = response.choices[0].message.content.strip()
        parsed = json.loads(ai_json)
    except Exception:
        return jsonify({"error": "AI did not return valid JSON"}), 500

    return jsonify({
        "result": {
            "important_topics": parsed["important_topics"],
            "ignore_topics": parsed["ignore_topics"],
            "questions": parsed["questions"]
        }
    })

# --------------------
# Check Answer
# --------------------
@app.route("/check_answer", methods=["POST"])
def check_answer():
    data = request.get_json()
    if not data or "question" not in data or "answer" not in data:
        return jsonify({"error": "Question and answer required"}), 400

    question = data["question"]
    answer = data["answer"]

    prompt = f"""
You are a strict examiner.

Question:
{question}

Student Answer:
{answer}

Rules:
- If correct → Correct
- If incorrect → Incorrect: short reason
- Be brief
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        feedback = response.choices[0].message.content.strip()
        return jsonify({"feedback": feedback})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --------------------
# Run
# --------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
