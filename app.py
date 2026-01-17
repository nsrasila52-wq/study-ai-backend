from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import io
import os
from pypdf import PdfReader
from openai import OpenAI
import json

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
# Analyze PDF / Photo
# --------------------
@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data received"}), 400

    content_text = ""
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

    elif "image_url" in data:
        # placeholder for image analysis text extraction
        content_text = "Extracted text from image placeholder"

    if not content_text.strip():
        return jsonify({"error": "No readable content"}), 400

    # Prompt for AI to return JSON
    prompt = f"""
You are a strict study AI.

Rules:
1️⃣ Pick max 3 topics to study as "important_topics".
2️⃣ Pick topics to ignore as "ignore_topics".
3️⃣ Create 2-3 clear questions as "questions".
4️⃣ Reply **only in JSON format** exactly like this:

{{
  "important_topics": ["topic1", "topic2", "topic3"],
  "ignore_topics": ["topic4", "topic5"],
  "questions": ["Question1?", "Question2?", "Question3?"]
}}

CONTENT:
{content_text[:12000]}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        ai_text = response.choices[0].message.content

        # Try to parse AI response as JSON
        try:
            ai_json = json.loads(ai_text)
        except:
            return jsonify({"error": "AI did not return valid JSON"}), 500

        return jsonify(ai_json)

    except Exception as e:
        return jsonify({"error": f"AI request failed: {str(e)}"}), 500


# --------------------
# Check Answer endpoint
# --------------------
@app.route("/check_answer", methods=["POST"])
def check_answer():
    data = request.get_json()
    if not data or "question" not in data or "answer" not in data:
        return jsonify({"error": "Question and answer required"}), 400

    question = data["question"]
    answer = data["answer"]

    prompt = f"""
You are an expert teacher.

Question: {question}
Student's Answer: {answer}

Rules:
- Check if the student's answer is correct or incorrect.
- Reply only in this format: "Correct" or "Incorrect: <short explanation>".
- Be very brief.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    result_text = response.choices[0].message.content
    return jsonify({"feedback": result_text})


# --------------------
# Run
# --------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
