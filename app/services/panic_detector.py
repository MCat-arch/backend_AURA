"""
PanicDetectorService: Dual-engine detection (Threshold + LSTM)

Layer 1 (Threshold): Runs every minute, determines isPanic → triggers notification
Layer 2 (LSTM): Runs every 7 minutes, provides p_panic probability for status labels

Flow:
    1. First 10 minutes → Calibration (collect baseline data)
    2. After calibration → Active detection:
       - Every minute: threshold check → isPanic → trigger
       - Every 7 minutes: LSTM inference → p_panic → status label
"""

import math
import time
import logging
import collections
from dataclasses import dataclass, field
from typing import Optional
from app.models.schemas import PanicInput, PanicPredictionResponse
from app.services.lstm_service import LSTMService

logger = logging.getLogger("PanicDetector")

# Constants
CALIBRATION_SAMPLES = 10       # 10 minutes of data
LSTM_WINDOW_SIZE = 7           # 7 data points for LSTM
LSTM_INTERVAL_SECONDS = 420    # 7 minutes between LSTM runs


@dataclass
class PanicDetectorSession:
    # --- Calibration ---
    calibration_bpm_buffer: list = field(default_factory=list)
    calibration_rmssd_buffer: list = field(default_factory=list)

    # --- Baseline stats (EMA) ---
    moving_mean: Optional[float] = None
    moving_variance: Optional[float] = None
    moving_std_dev: float = 0.0
    alpha: float = 0.2  # EMA smoothing factor

    # --- Personal thresholds (from calibration) ---
    threshold_upper_bpm: Optional[float] = None
    threshold_lower_rmssd: Optional[float] = None
    threshold_max_delta: Optional[float] = None

    # --- LSTM data window (sliding window of 9-feature vectors) ---
    data_window: collections.deque = field(
        default_factory=lambda: collections.deque(maxlen=LSTM_WINDOW_SIZE)
    )

    # --- LSTM timing ---
    last_lstm_time: Optional[float] = None

    # --- Tracking ---
    previous_bpm: Optional[int] = None
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    total_predictions: int = 0

    # --- Last known LSTM result ---
    last_p_panic: Optional[float] = None

    @property
    def is_calibrated(self) -> bool:
        return len(self.calibration_bpm_buffer) >= CALIBRATION_SAMPLES


class PanicDetectorService:
    def __init__(self, lstm_service: Optional[LSTMService] = None):
        self._sessions: dict[str, PanicDetectorSession] = {}
        self._lstm_service = lstm_service  # None = threshold-only mode
        mode = "dual-engine (threshold + LSTM)" if lstm_service else "threshold-only"
        logger.info(f"PanicDetectorService initialized — mode: {mode}")

    @property
    def active_sessions(self) -> int:
        return len(self._sessions)

    def get_or_create_session(self, session_id: str) -> PanicDetectorSession:
        if session_id not in self._sessions:
            self._sessions[session_id] = PanicDetectorSession()
            logger.info(f"New session created: {session_id}")
        return self._sessions[session_id]

    def reset_session(self, session_id: str) -> None:
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Session reset: {session_id}")

    def cleanup_stale_sessions(self, max_age_seconds: int = 86400) -> int:
        """Hapus session idle > 24 jam"""
        now = time.time()
        stale = [
            sid for sid, s in self._sessions.items()
            if (now - s.last_updated) > max_age_seconds
        ]
        for sid in stale:
            del self._sessions[sid]
        if stale:
            logger.info(f"Cleaned up {len(stale)} stale sessions")
        return len(stale)

    # predict

    def predict(self, session_id: str, data: PanicInput) -> PanicPredictionResponse:
        session = self.get_or_create_session(session_id)
        session.total_predictions += 1
        session.last_updated = time.time()

        # Extract SDNN from HRV60s for response
        current_sdnn = None
        if data.HRV60s and data.HRV60s.sdnn is not None:
            current_sdnn = data.HRV60s.sdnn

        # ─── FILTER: Skip if user is moving ───
        if not data.phoneSensor.isStill:
            return PanicPredictionResponse(
                p_panic=None,
                trigger=False,
                status="Data tidak diproses saat bergerak",
                bpm=data.bpm,
                sdnn=current_sdnn,
            )

        # ─── CALIBRATION PHASE (first 10 minutes) ───
        if not session.is_calibrated:
            session.calibration_bpm_buffer.append(float(data.bpm))

            # Also collect RMSSD for HRV threshold
            if data.HRV60s and data.HRV60s.rmssd is not None:
                session.calibration_rmssd_buffer.append(data.HRV60s.rmssd)

            progress = len(session.calibration_bpm_buffer)

            if progress >= CALIBRATION_SAMPLES:
                self._calculate_personal_thresholds(session)
                logger.info(
                    f"Calibration complete for {session_id} | "
                    f"upper_bpm={session.threshold_upper_bpm:.1f} | "
                    f"lower_rmssd={session.threshold_lower_rmssd}"
                )

            return PanicPredictionResponse(
                p_panic=None,
                trigger=False,
                status=f"Mengumpulkan baseline ({progress}/{CALIBRATION_SAMPLES})",
                bpm=data.bpm,
                sdnn=current_sdnn,
            )

        # ─── RHR correction ───
        if data.rhr > 0 and session.moving_mean is not None:
            if session.moving_mean < data.rhr - 5:
                session.moving_mean = data.rhr

        # ─── LAYER 1: Threshold Check (every minute) ───
        is_panic = self._detect_anomaly(session, data)

        # ─── Extract features & append to LSTM window (jika LSTM tersedia) ───
        if self._lstm_service is not None:
            hrv_dict = {}
            if data.HRV60s:
                hrv_dict = {
                    "meanRR": data.HRV60s.meanRR,
                    "sdnn": data.HRV60s.sdnn,
                    "rmssd": data.HRV60s.rmssd,
                    "pnn50": data.HRV60s.pnn50,
                    "nn50": data.HRV60s.nn50,
                }

            features = self._lstm_service.extract_features(data.bpm, hrv_dict)
            session.data_window.append(features)

            # ─── LAYER 2: LSTM Inference (every 7 minutes) ───
            p_panic = session.last_p_panic  # Default: use last known
            now = time.time()

            should_run_lstm = (
                len(session.data_window) >= LSTM_WINDOW_SIZE
                and (
                    session.last_lstm_time is None
                    or (now - session.last_lstm_time) >= LSTM_INTERVAL_SECONDS
                )
            )

            if should_run_lstm:
                try:
                    window_list = list(session.data_window)
                    p_panic = self._lstm_service.predict(window_list)
                    p_panic = round(p_panic, 4)
                    session.last_p_panic = p_panic
                    session.last_lstm_time = now
                    logger.info(
                        f"LSTM inference | session={session_id} | "
                        f"p_panic={p_panic}"
                    )
                except Exception as e:
                    logger.error(f"LSTM inference error: {e}", exc_info=True)
                    p_panic = session.last_p_panic
        else:
            p_panic = session.last_p_panic  # threshold-only mode, p_panic=None

        # ─── Update baseline stats (only when not panic) ───
        if not is_panic:
            self._update_stats(session, float(data.bpm))

        session.previous_bpm = data.bpm

        # ─── Determine status label ───
        status = self._determine_status(p_panic, is_panic)

        return PanicPredictionResponse(
            p_panic=p_panic,
            trigger=is_panic,
            status=status,
            bpm=data.bpm,
            sdnn=current_sdnn,
        )

    # ─── CALIBRATION ──────────────────────────────────────

    def _calculate_personal_thresholds(self, session: PanicDetectorSession) -> None:
        """
        Calculate personal thresholds from calibration data.
        Called once after 10 data points are collected.
        """
        buf = session.calibration_bpm_buffer
        if not buf:
            return

        # BPM statistics
        n = len(buf)
        mean_bpm = sum(buf) / n
        sum_sq_diff = sum((x - mean_bpm) ** 2 for x in buf)
        variance_bpm = sum_sq_diff / n
        std_bpm = math.sqrt(variance_bpm)

        if std_bpm < 1.0:
            std_bpm = 5.0  # Minimum std to avoid oversensitivity

        # Set baseline stats
        session.moving_mean = mean_bpm
        session.moving_variance = variance_bpm
        session.moving_std_dev = std_bpm

        # Personal BPM threshold: mean + 2*std
        session.threshold_upper_bpm = mean_bpm + (2 * std_bpm)

        # Max delta: based on observed range during calibration
        deltas = [abs(buf[i] - buf[i - 1]) for i in range(1, len(buf))]
        if deltas:
            max_observed_delta = max(deltas)
            # Threshold = max observed + buffer (at least 15)
            session.threshold_max_delta = max(max_observed_delta * 1.5, 15.0)
        else:
            session.threshold_max_delta = 20.0

        # RMSSD threshold (if data available)
        rmssd_buf = session.calibration_rmssd_buffer
        if len(rmssd_buf) >= 5:  # Need at least some data points
            mean_rmssd = sum(rmssd_buf) / len(rmssd_buf)
            sum_sq_rmssd = sum((x - mean_rmssd) ** 2 for x in rmssd_buf)
            std_rmssd = math.sqrt(sum_sq_rmssd / len(rmssd_buf))
            if std_rmssd < 1.0:
                std_rmssd = 5.0
            session.threshold_lower_rmssd = mean_rmssd - (1.5 * std_rmssd)
        else:
            session.threshold_lower_rmssd = 25.0  # Fallback default

    # ─── THRESHOLD DETECTION ──────────────────────────────

    def _detect_anomaly(self, session: PanicDetectorSession, data: PanicInput) -> bool:
        """
        Layer 1: Personal threshold-based anomaly detection.
        Runs every minute. Returns isPanic (bool).
        This directly determines `trigger` in the response → Flutter sends notification.

        Rules (using personal thresholds from calibration):
        1. BPM exceeds personal upper threshold → Extreme HR surge
        2. Z-score high AND RMSSD below personal lower threshold → HR + HRV anomaly
        3. Sudden BPM spike exceeds personal delta threshold → Sudden spike
        """
        if session.moving_mean is None or session.moving_std_dev == 0:
            return False

        z_score = (data.bpm - session.moving_mean) / session.moving_std_dev

        # Delta BPM (change from previous)
        delta = 0.0
        if session.previous_bpm is not None:
            delta = float(data.bpm - session.previous_bpm)

        # Current RMSSD
        rmssd = 100.0  # default high (normal)
        if data.HRV60s and data.HRV60s.rmssd is not None:
            rmssd = data.HRV60s.rmssd

        # --- Rule 1: BPM above personal upper threshold ---
        if (session.threshold_upper_bpm is not None
                and data.bpm > session.threshold_upper_bpm
                and z_score > 2.5):
            logger.warning(
                f"PANIC Rule 1 | bpm={data.bpm} > threshold={session.threshold_upper_bpm:.1f} | "
                f"z={z_score:.2f}"
            )
            return True

        # --- Rule 2: High Z-score AND low RMSSD ---
        if (session.threshold_lower_rmssd is not None
                and z_score > 2.0
                and rmssd < session.threshold_lower_rmssd):
            logger.warning(
                f"PANIC Rule 2 | z={z_score:.2f} | rmssd={rmssd:.1f} < "
                f"threshold={session.threshold_lower_rmssd:.1f}"
            )
            return True

        # --- Rule 3: Sudden spike ---
        if (session.threshold_max_delta is not None
                and delta > session.threshold_max_delta
                and session.threshold_upper_bpm is not None
                and data.bpm > session.threshold_upper_bpm * 0.8):
            logger.warning(
                f"PANIC Rule 3 | delta={delta:.0f} > threshold={session.threshold_max_delta:.0f} | "
                f"bpm={data.bpm}"
            )
            return True

        return False

    # ─── STATS UPDATE ─────────────────────────────────────

    def _update_stats(self, session: PanicDetectorSession, new_bpm: float) -> None:
        """Update EMA baseline stats (only during non-panic periods)."""
        if session.moving_mean is None:
            return

        diff = new_bpm - session.moving_mean
        increment = session.alpha * diff
        session.moving_mean += increment
        session.moving_variance = (1 - session.alpha) * (
            session.moving_variance + session.alpha * (diff ** 2)
        )
        session.moving_std_dev = math.sqrt(max(session.moving_variance, 0))

    # ─── STATUS LABEL ─────────────────────────────────────

    @staticmethod
    def _determine_status(p_panic: Optional[float], is_panic: bool) -> str:
        """
        Determine human-readable status label.
        Uses p_panic (from LSTM) if available, otherwise falls back to threshold result.
        """
        if p_panic is not None:
            # LSTM-based status (more accurate)
            if p_panic >= 0.80:
                return "🚨 Panic Tinggi"
            elif p_panic >= 0.55:
                return "⚠️ Stress Tinggi"
            elif p_panic >= 0.40:
                return "😐 Waspada"
            else:
                return "✅ Normal"
        else:
            # Threshold-only fallback
            if is_panic:
                return "⚠️ Anomali Terdeteksi"
            else:
                return "✅ Normal"
