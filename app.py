from flask import Flask, request, jsonify
from flask_cors import CORS
import requests, io, os
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
        return jsonify({"error":"No data received"}),400

    content_text = ""
    if "file_url" in data:
        try:
            r = requests.get(data["file_url"], timeout=20)
            if r.status_code != 200:
                return jsonify({"error":"Failed to fetch PDF"}),400
            reader = PdfReader(io.BytesIO(r.content))
            for page in reader.pages:
                text = page.extract_text()
                if text: content_text += text+"\n"
        except Exception as e:
            return jsonify({"error":f"PDF processing failed: {str(e)}"}),500
    elif "image_url" in data:
        content_text = "Extracted text from image placeholder"

    if not content_text.strip():
        return jsonify({"error":"No readable content"}),400

    # AI Prompt - JSON output
    prompt = f"""
You are a strict study AI. Analyze the content and respond ONLY in valid JSON.

Rules:
1️⃣ Pick max 3 topics to study from the content as "important_topics".
2️⃣ Mention topics to ignore as "ignore_topics".
3️⃣ Create 2-3 clear questions as "questions".
4️⃣ Return ONLY JSON, no extra text.

CONTENT:
{content_text[:12000]}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}]
    )

    ai_text = response.choices[0].message.content

    # Try parsing AI JSON
    import json
    try:
        ai_json = json.loads(ai_text)
        # Expecting: {"important_topics":[..], "ignore_topics":[..], "questions":[..]}
        topics_text = "Important Topics:\n" + "\n".join(ai_json.get("important_topics",[]))
        topics_text += "\n\nTopics to Ignore:\n" + "\n".join(ai_json.get("ignore_topics",[]))
        questions_list = ai_json.get("questions",[])
    except Exception as e:
        topics_text = "Error parsing AI output"
        questions_list = []

    return jsonify({"result":{"topics":topics_text,"questions":questions_list}})

@app.route("/check_answer", methods=["POST"])
def check_answer():
    data = request.get_json()
    if not data or "question" not in data or "answer" not in data:
        return jsonify({"error":"Question and answer required"}),400

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
        messages=[{"role":"user","content":prompt}]
    )

    result_text = response.choices[0].message.content
    return jsonify({"feedback":result_text})

if __name__=="__main__":
    port=int(os.environ.get("PORT",10000))
    app.run(host="0.0.0.0", port=port)
