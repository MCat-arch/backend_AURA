import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Header, status
from fastapi.middleware.cors import CORSMiddleware

from app.models.schemas import (PanicInput, PanicPredictionResponse, HealthCheckResponse)
from app.services.panic_detector import PanicDetectorService
from app.services.lstm_service import LSTMService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("AuraAPI")
detector_service: PanicDetectorService | None = None


# ─── LIFESPAN ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector_service
    logger.info("Starting AURA API...")

    lstm_service = LSTMService(
        model_path="app/ml_model/best_ft.keras",
        meta_path="app/ml_model/model_meta_panic.json",
    )
    logger.info("LSTMService loaded successfully — dual-engine mode AKTIF")

    # Initialize detector (dengan atau tanpa LSTM)
    detector_service = PanicDetectorService(lstm_service=lstm_service)
    mode = "dual-engine (threshold + LSTM)" if lstm_service else "threshold-only"
    logger.info(f"PanicDetectorService siap — mode: {mode}")

    yield
    detector_service = None
    logger.info("AURA API shutdown")
# ─── APP ──────────────────────────────────────────────
app = FastAPI(
    title="AURA Panic Attack Detection API",
    description="Backend deteksi panic attack dari data BLE armband + phone sensors. "
                "Dual-engine: Personal Threshold (per-minute) + LSTM Inference (per-7-minutes).",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
def _get_detector() -> PanicDetectorService:
    if detector_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Engine belum siap",
        )
    return detector_service
# ─── ENDPOINTS ────────────────────────────────────────
@app.get("/", response_model=HealthCheckResponse)
def health_check():
    svc = detector_service
    return HealthCheckResponse(
        status="healthy",
        version="2.0.0",
        detection_engine="dual-engine (threshold + lstm)",
        active_sessions=svc.active_sessions if svc else 0,
    )
@app.post("/predict", response_model=PanicPredictionResponse)
def predict_panic(
    data: PanicInput,
    x_device_id: str = Header(default="default", alias="X-Device-Id"),
):
    """
    Flutter mengirim HeartRateData.toJson() ke sini setiap 1 menit.
    X-Device-Id header digunakan untuk session management per device.

    Response format:
    {
      "p_panic": 0.9826,       // probabilitas dari LSTM (null jika belum jalan)
      "trigger": true,          // isPanic dari threshold → trigger notifikasi
      "status": "🚨 Panic Tinggi",  // label human-readable
      "bpm": 115,
      "sdnn": 12
    }
    """
    svc = _get_detector()
    try:
        result = svc.predict(session_id=x_device_id, data=data)
        if result.trigger:
            logger.warning(
                f"PANIC TRIGGERED | device={x_device_id} | bpm={result.bpm} | "
                f"p_panic={result.p_panic} | status={result.status}"
            )
        return result
    except Exception as e:
        logger.error(f"Error inferensi: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Kesalahan internal saat memproses deteksi",
        )
@app.post("/reset")
def reset_session(
    x_device_id: str = Header(default="default", alias="X-Device-Id"),
):
    svc = _get_detector()
    svc.reset_session(x_device_id)
    return {"message": f"Session {x_device_id} di-reset", "active_sessions": svc.active_sessions}
@app.post("/cleanup")
def cleanup_sessions():
    svc = _get_detector()
    cleaned = svc.cleanup_stale_sessions()
    return {"cleaned": cleaned, "remaining": svc.active_sessions}