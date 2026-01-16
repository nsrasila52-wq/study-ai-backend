from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi
import re
import requests
from PyPDF2 import PdfReader
import io

app = FastAPI()

class AnalyzeRequest(BaseModel):
    yt_url: str | None = None
    file_url: str | None = None


# ---------- HELPERS ----------

def extract_video_id(url: str):
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    if not match:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")
    return match.group(1)


def get_youtube_text(url: str):
    video_id = extract_video_id(url)
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    text = " ".join([item["text"] for item in transcript])
    return text


def get_pdf_text(file_url: str):
    response = requests.get(file_url)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="PDF download failed")

    pdf_bytes = io.BytesIO(response.content)
    reader = PdfReader(pdf_bytes)

    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() or ""

    if not full_text.strip():
        raise HTTPException(status_code=400, detail="PDF text extraction failed")

    return full_text


# ---------- MAIN API ----------

@app.post("/analyze")
def analyze(data: AnalyzeRequest):

    if data.yt_url:
        text = get_youtube_text(data.yt_url)
        source = "YouTube"

    elif data.file_url:
        text = get_pdf_text(data.file_url)
        source = "PDF"

    else:
        raise HTTPException(
            status_code=400,
            detail="Please provide yt_url or file_url"
        )

    # Abhi simple response — yahin OpenAI / logic lagega
    return {
        "source": source,
        "text_preview": text[:700],
        "length": len(text)
    }
