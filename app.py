from flask import Flask, request, jsonify
from flask_cors import CORS
import requests, io, os, re
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
        content_text = "Extracted text from image placeholder"

    if not content_text.strip():
        return jsonify({"error": "No readable content"}), 400

    # -------------------- PROMPT --------------------
    prompt = f"""
You are a strict study AI.

Rules:
1️⃣ Pick **max 3 important topics** and clearly return them under 'Important Topics'.
2️⃣ Mention topics to ignore under 'Topics to Ignore'.
3️⃣ Create 2-3 clear questions based on the content.
4️⃣ Return the final answer ONLY as a valid JSON object like this:

{{
  "topics": {{
      "important": ["topic1", "topic2", "topic3"],
      "ignore": ["topicA", "topicB"]
  }},
  "questions": ["question1", "question2", "question3"]
}}

CONTENT:
{content_text[:12000]}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        raw_text = response.choices[0].message.content

        # Extract JSON from AI response
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            json_text = match.group()
            import json
            result_json = json.loads(json_text)
            return jsonify(result_json)
        else:
            return jsonify({"error": "AI did not return valid JSON"}), 500

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

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        result_text = response.choices[0].message.content
        return jsonify({"feedback": result_text})
    except Exception as e:
        return jsonify({"error": f"AI request failed: {str(e)}"}), 500

# --------------------
# Run
# --------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
