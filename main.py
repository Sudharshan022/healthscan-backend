"""
HealthScan India – FastAPI Backend
===================================
Main application entry point. Orchestrates all routes, middleware,
CORS, and async lifecycle events.

Run with:  uvicorn main:app --reload --port 8000
"""

import uuid
import time
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse

from database import init_db, save_scan_record, get_scan_record, get_all_scans
from crypto_utils import encrypt_image_bytes, decrypt_image_bytes
from image_pipeline import run_full_pipeline
from nlp_utils import translate_report
from geo_utils import find_nearby_specialists

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Storage paths ─────────────────────────────────────────────────────────────
ENCRYPTED_STORE = Path("encrypted_images")
ENCRYPTED_STORE.mkdir(exist_ok=True)


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("🚀 HealthScan India starting up…")
    init_db()
    log.info("✅ Database initialized")
    yield
    log.info("🛑 HealthScan India shutting down…")


# ── App factory ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="HealthScan India API",
    description="AI-powered visual symptom analysis with end-to-end encryption, "
                "multilingual NLP, and geolocation medical routing.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the compiled frontend from /frontend
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ─────────────────────────────────────────────────────────────────────────────
# ROUTE: Health check
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def root():
    return {"status": "ok", "service": "HealthScan India", "version": "1.0.0"}


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy", "timestamp": time.time()}


# ─────────────────────────────────────────────────────────────────────────────
# ROUTE: /upload  — Core image analysis endpoint
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/upload", tags=["Analysis"])
async def upload_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Image of the affected body area (JPEG/PNG)"),
    body_part: str = Form("skin", description="Body part being analyzed, e.g. 'skin', 'joint'"),
    language: str = Form("en", description="ISO language code: en, hi, ta, te, kn, ml, bn, mr"),
    latitude: float = Form(None, description="User latitude for specialist search"),
    longitude: float = Form(None, description="User longitude for specialist search"),
    user_id: str = Form("anonymous"),
):
    """
    Full pipeline:
    1. Validate & read image bytes
    2. Encrypt and persist to disk (AES-256)
    3. Decrypt in-memory only for OpenCV preprocessing
    4. CNN inference → probabilistic classification
    5. Translate report to requested language
    6. Geolocation specialist routing (if coordinates provided)
    7. Save metadata to SQL, return structured response
    """

    # ── 1. Validate file type ────────────────────────────────────────────────
    allowed_types = {"image/jpeg", "image/png", "image/webp"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. Use JPEG or PNG.",
        )

    raw_bytes = await file.read()
    if len(raw_bytes) > 10 * 1024 * 1024:  # 10 MB guard
        raise HTTPException(status_code=413, detail="Image exceeds 10 MB limit.")

    scan_id = str(uuid.uuid4())
    log.info(f"[{scan_id}] Received upload | body_part={body_part} | lang={language}")

    # ── 2. Encrypt & persist ─────────────────────────────────────────────────
    encrypted_payload, encryption_meta = encrypt_image_bytes(raw_bytes)
    encrypted_path = ENCRYPTED_STORE / f"{scan_id}.enc"
    encrypted_path.write_bytes(encrypted_payload)
    log.info(f"[{scan_id}] Image encrypted (AES-256-GCM) and stored at {encrypted_path}")

    # ── 3 & 4. Decrypt in-memory → OpenCV → CNN ──────────────────────────────
    decrypted_bytes = decrypt_image_bytes(encrypted_payload, encryption_meta)
    pipeline_result = run_full_pipeline(decrypted_bytes, body_part=body_part)
    log.info(f"[{scan_id}] Pipeline complete | top_label={pipeline_result['top_label']}")

    # ── 5. Translate report ───────────────────────────────────────────────────
    translated = await translate_report(pipeline_result["report"], target_lang=language)

    # ── 6. Geolocation specialist routing ────────────────────────────────────
    specialists = []
    if latitude and longitude:
        specialists = await find_nearby_specialists(
            lat=latitude,
            lon=longitude,
            condition_category=pipeline_result["category"],
        )

    # ── 7. Persist metadata ───────────────────────────────────────────────────
    record = {
        "scan_id": scan_id,
        "user_id": user_id,
        "body_part": body_part,
        "language": language,
        "top_label": pipeline_result["top_label"],
        "confidence": pipeline_result["confidence"],
        "category": pipeline_result["category"],
        "encrypted_path": str(encrypted_path),
        "encryption_key_id": encryption_meta["key_id"],
        "timestamp": time.time(),
    }
    background_tasks.add_task(save_scan_record, record)

    # ── Response ──────────────────────────────────────────────────────────────
    return JSONResponse(
        content={
            "scan_id": scan_id,
            "status": "success",
            "ai_visual_assessment": {
                "top_label": pipeline_result["top_label"],
                "confidence_percent": round(pipeline_result["confidence"] * 100, 2),
                "all_probabilities": pipeline_result["all_probabilities"],
                "severity": pipeline_result["severity"],
                "category": pipeline_result["category"],
            },
            "report": {
                "original_en": pipeline_result["report"],
                "translated": translated,
                "language": language,
            },
            "recommendations": pipeline_result["recommendations"],
            "nearby_specialists": specialists,
            "encryption": {
                "status": "encrypted_at_rest",
                "algorithm": "AES-256-GCM",
                "key_id": encryption_meta["key_id"],
            },
            "disclaimer": (
                "⚠️ MEDICAL DISCLAIMER: This is an AI Visual Assessment only and does NOT "
                "constitute medical diagnosis, advice, or treatment. Always consult a qualified, "
                "licensed medical professional for any health concerns. In an emergency, "
                "call 112 (India national emergency number) immediately."
            ),
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# ROUTE: /scan/{scan_id}  — Retrieve past scan metadata
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/scan/{scan_id}", tags=["Records"])
async def get_scan(scan_id: str):
    """Fetch previously stored scan metadata (image is never returned raw)."""
    record = get_scan_record(scan_id)
    if not record:
        raise HTTPException(status_code=404, detail="Scan record not found.")
    # Scrub sensitive paths before returning
    record.pop("encrypted_path", None)
    record.pop("encryption_key_id", None)
    return record


# ─────────────────────────────────────────────────────────────────────────────
# ROUTE: /history — All scans for a user
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/history", tags=["Records"])
async def scan_history(user_id: str = "anonymous", limit: int = 20):
    """Return the last N scan summaries for a given user."""
    records = get_all_scans(user_id=user_id, limit=limit)
    for r in records:
        r.pop("encrypted_path", None)
        r.pop("encryption_key_id", None)
    return {"user_id": user_id, "scans": records}


# ─────────────────────────────────────────────────────────────────────────────
# ROUTE: Serve frontend
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/app", tags=["Frontend"])
async def serve_frontend():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "Frontend not found. See /frontend/index.html"}
