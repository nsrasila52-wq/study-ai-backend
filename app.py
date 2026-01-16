from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import io
import os

from pypdf import PdfReader
from openai import OpenAI

# --------------------
# App setup
# --------------------
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
# Analyze endpoint
# --------------------
@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()

    if not data:
        return jsonify({"error": "No data received"}), 400

    # ==================================================
    # PDF CASE
    # ==================================================
    if "file_url" in data:
        try:
            file_url = data["file_url"]

            r = requests.get(file_url, timeout=20)
            if r.status_code != 200:
                return jsonify({"error": "Failed to fetch PDF"}), 400

            reader = PdfReader(io.BytesIO(r.content))

            full_text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"

            if not full_text.strip():
                return jsonify({"error": "No readable text in PDF"}), 400

            # Prompt without "today"
            prompt = f"""
You are a strict study decision AI.

Rules:
1️⃣ Pick **max 3 topics** to study. Prioritize the most important parts.
2️⃣ Clearly say what to **ignore**. Be explicit.
3️⃣ Be **short, direct, and actionable**. No extra sentences.

PDF CONTENT:
{full_text[:12000]}
"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )

            return jsonify({
                "result": response.choices[0].message.content
            })

        except Exception as e:
            return jsonify({
                "error": f"PDF processing failed: {str(e)}"
            }), 500

    # ==================================================
    # PHOTO CASE (optional)
    # ==================================================
    if "image_url" in data:
        try:
            return jsonify({"result": "Photo analyze logic placeholder (already working)"})
        except Exception as e:
            return jsonify({
                "error": f"Photo processing failed: {str(e)}"
            }), 500

    return jsonify({"error": "No valid input"}), 400


# --------------------
# Run
# --------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
