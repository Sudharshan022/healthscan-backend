"""
HealthScan India - FastAPI Backend v2.0
Run with: uvicorn main:app --reload --port 8000
"""

import uuid
import time
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from database import init_db, save_scan_record, get_scan_record, get_all_scans
from crypto_utils import encrypt_image_bytes, decrypt_image_bytes
from image_pipeline import run_full_pipeline
from nlp_utils import translate_report
from geo_utils import find_nearby_specialists
from report_pipeline import parse_report_text, build_report_analysis

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

ENCRYPTED_STORE = Path("encrypted_images")
ENCRYPTED_STORE.mkdir(exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("HealthScan India v2 starting")
    init_db()
    yield


app = FastAPI(title="HealthScan India API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"status": "ok", "service": "HealthScan India", "version": "2.0.0"}


@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": time.time()}


@app.post("/upload")
async def upload_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    body_part: str = Form("skin"),
    language: str = Form("en"),
    latitude: float = Form(None),
    longitude: float = Form(None),
    user_id: str = Form("anonymous"),
):
    allowed_types = {"image/jpeg", "image/png", "image/webp"}
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=415, detail="Use JPEG or PNG image.")

    raw_bytes = await file.read()
    if len(raw_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image exceeds 10 MB limit.")

    scan_id = str(uuid.uuid4())
    encrypted_payload, encryption_meta = encrypt_image_bytes(raw_bytes)
    encrypted_path = ENCRYPTED_STORE / (scan_id + ".enc")
    encrypted_path.write_bytes(encrypted_payload)

    decrypted_bytes = decrypt_image_bytes(encrypted_payload, encryption_meta)
    pipeline_result = run_full_pipeline(decrypted_bytes, body_part=body_part)
    translated = await translate_report(pipeline_result["report"], target_lang=language)

    specialists = []
    if latitude and longitude:
        specialists = await find_nearby_specialists(
            lat=latitude, lon=longitude,
            condition_category=pipeline_result["category"],
        )

    background_tasks.add_task(save_scan_record, {
        "scan_id": scan_id, "user_id": user_id,
        "body_part": body_part, "language": language,
        "top_label": pipeline_result["top_label"],
        "confidence": pipeline_result["confidence"],
        "category": pipeline_result["category"],
        "encrypted_path": str(encrypted_path),
        "encryption_key_id": encryption_meta["key_id"],
        "timestamp": time.time(),
    })

    return JSONResponse(content={
        "scan_id": scan_id, "status": "success", "type": "image_analysis",
        "ai_visual_assessment": {
            "top_label": pipeline_result["top_label"],
            "confidence_percent": round(pipeline_result["confidence"] * 100, 2),
            "all_probabilities": pipeline_result["all_probabilities"],
            "severity": pipeline_result["severity"],
            "category": pipeline_result["category"],
        },
        "report": {"original_en": pipeline_result["report"], "translated": translated, "language": language},
        "recommendations": pipeline_result["recommendations"],
        "nearby_specialists": specialists,
        "encryption": {"status": "encrypted_at_rest", "algorithm": "AES-256-GCM", "key_id": encryption_meta["key_id"]},
        "disclaimer": "MEDICAL DISCLAIMER: AI assessment only. Consult a qualified doctor. Emergency: 112.",
    })


@app.post("/analyze-report")
async def analyze_report(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(None),
    report_text: str = Form(""),
    language: str = Form("en"),
    user_id: str = Form("anonymous"),
    latitude: float = Form(None),
    longitude: float = Form(None),
):
    scan_id = str(uuid.uuid4())
    extracted_text = report_text.strip()

    if file and file.filename:
        raw_bytes = await file.read()
        if len(raw_bytes) > 15 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File exceeds 15 MB.")

        if file.content_type and "pdf" in file.content_type:
            try:
                import io
                import pdfplumber
                with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
                    pages_text = [p.extract_text() or "" for p in pdf.pages]
                    extracted_text = " ".join(pages_text) + " " + extracted_text
            except ImportError:
                extracted_text += " " + (file.filename or "")

        elif file.content_type and "image" in file.content_type:
            try:
                import io
                import pytesseract
                from PIL import Image
                img = Image.open(io.BytesIO(raw_bytes))
                extracted_text = pytesseract.image_to_string(img) + " " + extracted_text
            except ImportError:
                extracted_text += " " + (file.filename or "")

    if not extracted_text:
        raise HTTPException(status_code=400, detail="Please upload a file or paste report text.")

    parsed = parse_report_text(extracted_text)
    analysis_result = build_report_analysis(parsed, user_text_input=extracted_text)

    summary_for_translation = (
        analysis_result["primary_condition_display"] + ". " + analysis_result["description"]
    )
    translated = await translate_report(summary_for_translation, target_lang=language)

    specialists = []
    if latitude and longitude:
        specialists = await find_nearby_specialists(lat=latitude, lon=longitude, condition_category="general")

    background_tasks.add_task(save_scan_record, {
        "scan_id": scan_id, "user_id": user_id,
        "body_part": "report", "language": language,
        "top_label": analysis_result["primary_condition_display"],
        "confidence": 0.85, "category": "report_analysis",
        "encrypted_path": "", "encryption_key_id": "",
        "timestamp": time.time(),
    })

    return JSONResponse(content={
        "scan_id": scan_id, "status": "success", "type": "report_analysis",
        "analysis": analysis_result,
        "translated_summary": translated,
        "nearby_specialists": specialists,
        "disclaimer": analysis_result["disclaimer"],
    })


@app.get("/scan/{scan_id}")
async def get_scan(scan_id: str):
    record = get_scan_record(scan_id)
    if not record:
        raise HTTPException(status_code=404, detail="Scan not found.")
    record.pop("encrypted_path", None)
    record.pop("encryption_key_id", None)
    return record


@app.get("/history")
async def scan_history(user_id: str = "anonymous", limit: int = 20):
    records = get_all_scans(user_id=user_id, limit=limit)
    for r in records:
        r.pop("encrypted_path", None)
        r.pop("encryption_key_id", None)
    return {"user_id": user_id, "scans": records}
