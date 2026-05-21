from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class HRVMetricsInput(BaseModel):
    count: int = Field(..., description="Jumlah RR intervals dalam window")
    meanRR: Optional[float] = Field(None, description="Mean RR interval (ms)")
    sdnn: Optional[float] = Field(None, description="Std dev of NN interval (ms)")
    rmssd: Optional[float] = Field(None, description="Root mean square of successive diffs")
    nn50: Optional[float] = Field(None, description="Pairs of successive Nns differing > 50ms")
    pnn50: Optional[float] = Field(None, description="Percentage of NN50")


class SpatioTemporalInput(BaseModel):
    """Match: lib/models/spatio.model.dart → SpatioTemporal"""
    rawActivityStatus: str = Field(..., description="STILL, WALKING, RUNNING, UNKNOWN")
    time: str = Field(..., description="HH:mm:ss")
    noiseLeveldB: Optional[float] = Field(None, description="Noise level dari mic HP (dB)")
    isWalking: bool = Field(False)
    isRunning: bool = Field(False)
    isStill: bool = Field(True)
    timeOfDayCategory: str = Field("unknown", description="morning/afternoon/evening/night")

class PanicInput(BaseModel):
    # akan dikirim post setiap 1 menit (main lgoic)
    bpm: int = Field(..., description="BPM rata-rata dari RR Intervals", ge=0, le=300)
    timestamp: datetime = Field(..., description="ISO 8601 timestamp")
    rrIntervals: Optional[List[float]] = Field(None, description="Raw RR intervals (ms)")
    HRV10s: Optional[HRVMetricsInput] = None
    HRV30s: Optional[HRVMetricsInput] = None
    HRV60s: Optional[HRVMetricsInput] = None
    rhr: float = Field(0.0, description="Resting Heart Rate", ge=0)
    phoneSensor: SpatioTemporalInput

class PanicPredictionResponse(BaseModel):
    """
    Response format sesuai requirement:
    {
      "p_panic": 0.9826,      # probabilitas dari model LSTM
      "trigger": false,        # isPanic dari threshold (trigger notifikasi)
      "status": "🚨 Panic Tinggi",  # label human-readable
      "bpm": 115,
      "sdnn": 12
    }
    """
    p_panic: Optional[float] = Field(None, ge=0.0, le=1.0, description="Probabilitas panic dari LSTM (null jika belum jalan)")
    trigger: bool = Field(False, description="isPanic dari threshold check — trigger notifikasi")
    status: str = Field("Menunggu data", description="Label human-readable")
    bpm: int = Field(0, description="BPM saat ini")
    sdnn: Optional[float] = Field(None, description="SDNN saat ini (dari HRV60s)")

class HealthCheckResponse(BaseModel):
    status: str
    version: str
    detection_engine: str
    active_sessions: int
