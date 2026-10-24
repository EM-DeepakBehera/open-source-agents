"""Scam Shield web server: serves the lightweight PWA and the /api/check endpoint."""

from __future__ import annotations

import base64
import logging
import os
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.middleware.gzip import GZipMiddleware  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from google.adk.runners import Runner  # noqa: E402
from google.adk.sessions import InMemorySessionService  # noqa: E402
from google.genai import types  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from scam_shield.agent import MODEL, root_agent  # noqa: E402
from scam_shield.tools import rules_verdict  # noqa: E402

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
log = logging.getLogger("scam-shield")

STATIC = Path(__file__).parent / "static"
APP_NAME = "scam_shield"
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MINUTE", "20"))
MAX_IMAGE_BYTES = 1_500_000
LANGUAGES = {
    "en": "English", "hi": "Hindi", "kn": "Kannada", "ta": "Tamil",
    "te": "Telugu", "ml": "Malayalam", "mr": "Marathi", "bn": "Bengali",
}

sessions = InMemorySessionService()
runner = Runner(app_name=APP_NAME, agent=root_agent, session_service=sessions)
app = FastAPI(title="Scam Shield", docs_url=None, redoc_url=None)
app.add_middleware(GZipMiddleware, minimum_size=500)
_hits: dict[str, deque] = defaultdict(deque)


class CheckRequest(BaseModel):
    text: str = Field(default="", max_length=5000)
    image_base64: str | None = Field(default=None, max_length=2_100_000)
    image_mime: str = "image/jpeg"
    language: str = "en"
    simple: bool = False


def _rate_limited(ip: str) -> bool:
    now = time.monotonic()
    window = _hits[ip]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= RATE_LIMIT:
        return True
    window.append(now)
    return False


def _ai_enabled() -> bool:
    return bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
                or os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true"))


async def run_agent(text: str, image: bytes | None, mime: str, language: str, simple: bool) -> dict:
    """Runs the multi-agent pipeline once and returns the structured verdict."""
    parts = []
    if text:
        parts.append(types.Part(text=f"Message to check:\n\"\"\"\n{text}\n\"\"\""))
    if image:
        parts.append(types.Part(text="Screenshot of the message to check:"))
        parts.append(types.Part.from_bytes(data=image, mime_type=mime))
    session_id = uuid.uuid4().hex
    await sessions.create_session(
        app_name=APP_NAME, user_id="web", session_id=session_id,
        state={"language": LANGUAGES.get(language, "English"), "simple_mode": "on" if simple else "off"},
    )
    try:
        async for _ in runner.run_async(user_id="web", session_id=session_id,
                                        new_message=types.Content(role="user", parts=parts)):
            pass
        session = await sessions.get_session(app_name=APP_NAME, user_id="web", session_id=session_id)
        verdict = session.state.get("verdict")
        if not isinstance(verdict, dict):
            raise RuntimeError("Agent did not return a verdict")
        return verdict
    finally:
        await sessions.delete_session(app_name=APP_NAME, user_id="web", session_id=session_id)


@app.post("/api/check")
async def check(req: CheckRequest, request: Request):
    if _rate_limited(request.client.host if request.client else "unknown"):
        raise HTTPException(429, "Too many checks. Please wait a minute.")
    text = req.text.strip()
    image = None
    if req.image_base64:
        if req.image_mime not in ("image/jpeg", "image/png", "image/webp"):
            raise HTTPException(400, "Unsupported image type")
        try:
            image = base64.b64decode(req.image_base64.split(",")[-1], validate=True)
        except ValueError:
            raise HTTPException(400, "Invalid image data")
        if len(image) > MAX_IMAGE_BYTES:
            raise HTTPException(413, "Image too large")
    if not text and not image:
        raise HTTPException(400, "Please provide a message or a screenshot")

    started = time.perf_counter()
    if _ai_enabled():
        try:
            result = await run_agent(text, image, req.image_mime, req.language, req.simple)
            result["mode"] = "ai"
            result["model"] = MODEL
            result["seconds"] = round(time.perf_counter() - started, 2)
            return result
        except Exception:  # noqa: BLE001 - any AI failure falls back to rules
            log.exception("AI check failed; falling back to rules")
    if not text:
        raise HTTPException(503, "AI is unavailable, and screenshots need AI. Please paste the text.")
    result = rules_verdict(text)
    result.pop("details", None)
    result["mode"] = "rules"
    result["seconds"] = round(time.perf_counter() - started, 2)
    return result


@app.get("/api/health")
async def health():
    return {"ok": True, "ai": _ai_enabled(), "model": MODEL}


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/sw.js")
async def service_worker():
    return FileResponse(STATIC / "sw.js", media_type="text/javascript",
                        headers={"Cache-Control": "no-cache"})


app.mount("/static", StaticFiles(directory=STATIC), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8080")))
