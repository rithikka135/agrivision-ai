import os
import io
import base64
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.vision import classifier
from app.rag import rag_engine
from app.pdf_generator import generate_pdf_report
from app.dosage import calculate_field_dosage
from app.database import save_scan_record, fetch_recent_scans
from app.economics import calculate_economic_risk
from app.dealers import fetch_nearby_dealers

app = FastAPI(
    title="Smart Crop Health Diagnostic Platform",
    version="4.3.0",
    description="Enterprise AI diagnostic engine combining vision inference, Grad-CAM, leaf damage estimation, weather ingestion, and dosage math."
)

# Enable CORS for browser/frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serves the primary web dashboard UI."""
    html_path = os.path.join("app", "templates", "index.html")
    if not os.path.exists(html_path):
        html_path = "index.html"
        
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Smart Crop Platform API Active</h1><p>index.html not found in templates.</p>"

@app.post("/diagnose")
async def diagnose_crop(
    file: UploadFile = File(...),
    latitude: float = Form(11.0168),
    longitude: float = Form(76.9558),
    language: str = Form("en"),
    field_acres: float = Form(1.0)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Upload an image.")
    
    image_bytes = await file.read()
    disease, confidence, gradcam_data, damage_pct = classifier.predict(image_bytes)
    
    # Process Grad-CAM Base64 format
    if isinstance(gradcam_data, str) and gradcam_data.startswith("data:image"):
        gradcam_b64 = gradcam_data
    elif isinstance(gradcam_data, bytes):
        raw_b64 = base64.b64encode(gradcam_data).decode("utf-8")
        gradcam_b64 = f"data:image/jpeg;base64,{raw_b64}"
    else:
        gradcam_b64 = str(gradcam_data)

    # -------------------------------------------------------------
    # LOGIC GUARD: IF CROP IS HEALTHY OR UNCLEAR
    # -------------------------------------------------------------
    is_healthy = "healthy" in disease.lower()
    is_unclear = "unclear" in disease.lower() or confidence < 55.0

    if is_unclear:
        weather_info = rag_engine.fetch_weather(latitude, longitude)
        advisory = {
            "disease_detected": "Unclear / Non-Crop Image Detected",
            "weather_context": weather_info,
            "llm_advisory": "The uploaded photo is either not a leaf or is too blurry for an accurate diagnosis. Please upload a clear photo.",
            "audio_b64": ""
        }
        damage_pct = 0.0
        dosage_info = {"status": "No Chemical Required", "message": "Diagnosis inconclusive."}
        economic_info = {"status": "Calculation Skipped", "crop_type": "Unknown", "projected_yield_kg": "N/A", "estimated_financial_loss": "₹0", "action_note": "Re-upload image."}

    elif is_healthy:
        weather_info = rag_engine.fetch_weather(latitude, longitude)
        # Override weather risk and damage for healthy crops
        weather_info["risk_score"] = "Low Risk (Healthy Crop)"
        
        advisory = {
            "disease_detected": disease,
            "weather_context": weather_info,
            "llm_advisory": "Your crop shows no signs of disease infection. Maintain standard irrigation and nutrient management.",
            "audio_b64": ""
        }
        damage_pct = 0.0  # Force damage to 0% for healthy plants
        dosage_info = {"status": "No Chemical Required", "message": "Crop is healthy! No chemical treatment required."}
        economic_info = {"status": "Safe", "crop_type": disease.split("___")[0], "projected_yield_kg": "Optimal", "estimated_financial_loss": "₹0", "action_note": "No yield loss expected."}

    else:
        # Standard flow for infected crops
        advisory = rag_engine.generate_advisory(disease, latitude, longitude, lang=language)
        dosage_info = calculate_field_dosage(disease, field_acres)
        economic_info = calculate_economic_risk(disease, field_acres)

    nearby_dealers = fetch_nearby_dealers(latitude, longitude)
    
    save_scan_record(
        disease=advisory["disease_detected"],
        confidence=confidence,
        lat=latitude,
        lon=longitude,
        acres=field_acres,
        risk=advisory["weather_context"].get("risk_score", "Low Risk")
    )
    
    return {
        "status": "success",
        "confidence_score": confidence,
        "leaf_damage_percentage": f"{damage_pct}%",
        "diagnosis": advisory,
        "field_dosage": dosage_info,
        "economic_risk": economic_info,
        "nearby_dealers": nearby_dealers,
        "gradcam_image_b64": gradcam_b64
    }
@app.get("/scan-history")
async def get_history():
    """Fetches recent scan records from SQLite audit database."""
    return fetch_recent_scans(limit=5)

@app.post("/calculate-dosage")
async def recalculate_dosage(
    disease_name: str = Form(...),
    field_acres: float = Form(1.0)
):
    """Dynamic dosage recalculation endpoint triggered when user adjusts acreage slider."""
    if "Unclear" in disease_name:
        return {
            "field_dosage": {
                "status": "No Chemical Required",
                "message": "Diagnosis inconclusive. Upload a clear leaf image."
            },
            "economic_risk": {
                "status": "Calculation Skipped",
                "crop_type": "Unknown",
                "projected_yield_kg": "N/A",
                "estimated_financial_loss": "₹0",
                "action_note": "Re-upload a clear leaf image."
            }
        }
        
    dosage_info = calculate_field_dosage(disease_name, field_acres)
    economic_info = calculate_economic_risk(disease_name, field_acres)
    return {
        "field_dosage": dosage_info,
        "economic_risk": economic_info
    }

@app.post("/download-pdf")
async def download_pdf(
    file: UploadFile = File(...),
    latitude: float = Form(11.0168),
    longitude: float = Form(76.9558),
    language: str = Form("en")
):
    """Generates downloadable PDF diagnostic certificates via ReportLab."""
    image_bytes = await file.read()
    
    # Vision Inference
    disease, confidence, gradcam_data, damage_pct = classifier.predict(image_bytes)
    
    # Convert Grad-CAM string back to raw bytes if needed for ReportLab PDF renderer
    if isinstance(gradcam_data, str) and "base64," in gradcam_data:
        raw_b64_str = gradcam_data.split("base64,")[1]
        gradcam_bytes = base64.b64decode(raw_b64_str)
    elif isinstance(gradcam_data, bytes):
        gradcam_bytes = gradcam_data
    else:
        gradcam_bytes = image_bytes
    
    if "Unclear" in disease or confidence < 55.0:
        advisory = {
            "disease_detected": "Unclear / Non-Crop Image Detected",
            "weather_context": rag_engine.fetch_weather(latitude, longitude),
            "llm_advisory": "The uploaded photo is either not a leaf or is too blurry for an accurate diagnostic."
        }
    else:
        advisory = rag_engine.generate_advisory(disease, latitude, longitude, lang=language)
    
    pdf_bytes = generate_pdf_report(
        disease=advisory["disease_detected"],
        confidence=confidence,
        weather=advisory["weather_context"],
        advisory_text=f"Estimated Leaf Damage Area: {damage_pct}%\n\n{advisory['llm_advisory']}",
        gradcam_bytes=gradcam_bytes
    )
    
    return Response(
        content=pdf_bytes, 
        media_type="application/pdf", 
        headers={"Content-Disposition": "attachment; filename=Crop_Health_Diagnostic_Report.pdf"}
    )