import os
import re
import asyncio
import tempfile
import requests
import edge_tts
from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
VOICE        = os.getenv("EDGE_TTS_VOICE", "en-IN-NeerjaExpressiveNeural")
VOICE_RATE   = os.getenv("EDGE_TTS_RATE",  "+0%")
VOICE_PITCH  = os.getenv("EDGE_TTS_PITCH", "+0Hz")
PORT         = int(os.getenv("PORT", 5001))

# ── Text cleaner for TTS ──────────────────────────────────────────────────────
def clean_for_tts(text: str) -> str:
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'`([^`]*)`', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'[#*_~>|]', ' ', text)
    text = re.sub(r'^\s*-\s+', '. ', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ── Input validation ──────────────────────────────────────────────────────────
def validate_prompt(prompt: str) -> str:
    if not prompt or not isinstance(prompt, str):
        raise ValueError("prompt is required")
    prompt = prompt.strip()
    if len(prompt) < 2:
        raise ValueError("prompt is too short")
    if len(prompt) > 1000:
        raise ValueError("prompt too long (max 1000 chars)")
    return prompt

def validate_text_input(text: str) -> str:
    if not text or not isinstance(text, str):
        raise ValueError("text is required")
    text = text.strip()
    if len(text) < 2:
        raise ValueError("text is too short")
    if len(text) > 3000:
        text = text[:3000]  # cap silently for TTS
    return text

# ── Groq AI ───────────────────────────────────────────────────────────────────
def ask_groq(prompt: str) -> str:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not configured")
    response = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}",
                 "Content-Type":  "application/json"},
        json={
            "model":    "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": f"""Answer as a spoken teacher.

Rules:
- Do not use markdown, *, **, #, bullet points, tables or code blocks
- Speak naturally like a classroom teacher
- Keep answer concise (under 150 words)

Question: {prompt}"""}]
        },
        timeout=60
    )
    if response.status_code != 200:
        raise ValueError(f"Groq API Error {response.status_code}: {response.text[:200]}")
    return response.json()["choices"][0]["message"]["content"]

# ── Edge TTS ──────────────────────────────────────────────────────────────────
async def _tts(text: str, path: str):
    communicate = edge_tts.Communicate(
        text=text, voice=VOICE, rate=VOICE_RATE, pitch=VOICE_PITCH
    )
    await communicate.save(path)

def generate_speech(text: str) -> str:
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3", dir="/tmp")
    f.close()
    asyncio.run(_tts(text, f.name))
    return f.name

def get_base_url() -> str:
    host   = request.headers.get("X-Forwarded-Host") or request.host
    scheme = request.headers.get("X-Forwarded-Proto", request.scheme)
    return f"{scheme}://{host}"

# ── ROUTES ────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return jsonify({
        "service": "MaitriLearn Voice Service",
        "status":  "running",
        "voice":   VOICE,
        "version": "2.0",
        "endpoints": ["/generate", "/tts", "/audio/<filename>",
                      "/voices", "/health"]
    })

@app.route("/health")
def health():
    return jsonify({"status": "ok", "voice": VOICE})

@app.route("/voices")
def voices():
    return jsonify({
        "current": VOICE,
        "available": [
            {"id": "en-IN-NeerjaExpressiveNeural", "name": "Neerja Expressive ⭐", "recommended": True},
            {"id": "en-IN-NeerjaNeural",           "name": "Neerja (Female)"},
            {"id": "en-IN-PrabhatNeural",          "name": "Prabhat (Male)"},
        ]
    })

@app.route("/tts", methods=["POST"])
def tts_only():
    """
    TTS only — convert text to speech, no AI.
    Used by whiteboard narration (AI already generated text).
    Body: { "text": "text to speak" }
    """
    data = request.get_json(silent=True) or {}
    try:
        text = validate_text_input(data.get("text", ""))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    try:
        clean  = clean_for_tts(text)
        path   = generate_speech(clean)
        fname  = os.path.basename(path)
        base   = get_base_url()
        return jsonify({
            "success":   True,
            "audio":     f"/audio/{fname}",
            "audio_url": f"{base}/audio/{fname}",
            "voice":     VOICE
        })
    except Exception as e:
        print(f"[tts] Error: {e}")
        return jsonify({"error": str(e)}), 503

@app.route("/generate", methods=["POST"])
def generate():
    """
    Full pipeline — Groq AI answer + TTS.
    Used by voice tutor feature.
    Body: { "prompt": "student question" }
    """
    data = request.get_json(silent=True) or {}
    try:
        prompt = validate_prompt(data.get("prompt", ""))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    try:
        print(f"[generate] Prompt: {prompt[:80]}")
        answer   = ask_groq(prompt)
        clean    = clean_for_tts(answer)
        path     = generate_speech(clean)
        fname    = os.path.basename(path)
        base     = get_base_url()

        return jsonify({
            "success":   True,
            "text":      answer,
            "audio":     f"/audio/{fname}",
            "audio_url": f"{base}/audio/{fname}",
            "voice":     VOICE
        })
    except Exception as e:
        print(f"[generate] Error: {e}")
        return jsonify({"error": str(e)}), 503

@app.route("/audio/<filename>")
def audio(filename):
    filename = os.path.basename(filename)
    filepath = f"/tmp/{filename}"
    if not os.path.exists(filepath):
        return jsonify({"error": "Audio not found"}), 404
    with open(filepath, "rb") as f:
        data = f.read()
    return Response(
        data,
        mimetype="audio/mpeg",
        headers={
            "Content-Length":              str(len(data)),
            "Accept-Ranges":               "bytes",
            "Cache-Control":               "no-cache",
            "Access-Control-Allow-Origin": "*"
        }
    )

if __name__ == "__main__":
    print(f"🎤 MaitriLearn Voice Service — voice: {VOICE} — port: {PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
