import json
import logging
import os
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import requests
from fastapi import Body, File, FastAPI, Query, Request, Response, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("perception")

app = FastAPI(title="Perception Vision Service")

# ── Tunables ──────────────────────────────────────────────────────────
CAM_INDEX = int(os.getenv("CAM_INDEX", "0"))
WIDTH = int(os.getenv("CAM_WIDTH", "1920"))
HEIGHT = int(os.getenv("CAM_HEIGHT", "1080"))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "90"))
CAPTURE_FPS = int(os.getenv("CAPTURE_FPS", "15"))

CHANGE_THRESHOLD = float(os.getenv("CHANGE_THRESHOLD", "30.0"))
CHANGE_CHECK_INTERVAL = float(os.getenv("CHANGE_CHECK_INTERVAL", "1.0"))
MIN_PUSH_INTERVAL = float(os.getenv("MIN_PUSH_INTERVAL", "3.0"))

N8N_WEBHOOK_URL = os.getenv(
    "N8N_WEBHOOK_URL", "http://localhost:5678/webhook/perception-trigger"
)
PUSH_ENABLED = os.getenv("PUSH_ENABLED", "true").lower() == "true"

# QR code → persisted `location` field (sticky until a new QR replaces it)
QR_SCAN_ENABLED = os.getenv("QR_SCAN_ENABLED", "true").lower() == "true"
QR_SCAN_INTERVAL = float(os.getenv("QR_SCAN_INTERVAL", "0.5"))
# Draw QR outline + label on MJPEG /stream frames
QR_STREAM_OVERLAY = os.getenv("QR_STREAM_OVERLAY", "true").lower() == "true"

# DeepFace stream mode: when true, stream runs face analysis + identification (heavy).
# When false, uses simple capture loop — inference only via n8n Perception workflow.
DEEPFACE_STREAM = os.getenv("DEEPFACE_STREAM", "true").lower() == "true"
STREAM_FRAME_THRESHOLD = int(os.getenv("STREAM_FRAME_THRESHOLD", "1"))
STREAM_TIME_THRESHOLD = int(os.getenv("STREAM_TIME_THRESHOLD", "5"))
# Decoupled DeepFace mode: capture/composer runs at PREVIEW_FPS; analysis runs every ANALYSIS_INTERVAL.
PREVIEW_FPS = int(os.getenv("PREVIEW_FPS", "30"))
ANALYSIS_INTERVAL = float(os.getenv("ANALYSIS_INTERVAL", "0.12"))
# Distant faces produce small boxes (pixels); the old default 130 filtered most of them out.
FACE_MIN_WIDTH = int(os.getenv("FACE_MIN_WIDTH", "48"))
# DeepFace detector backends: yunet (fast DNN), opencv (Haar cascade), retinaface, mtcnn, ssd, mediapipe, ...
DEEPFACE_DETECTOR_BACKEND = os.getenv("DEEPFACE_DETECTOR_BACKEND", "yunet").strip() or "yunet"
# Scale factor applied to frames before face detection (smaller = faster detection, coords scaled back up).
DETECTION_SCALE = float(os.getenv("DETECTION_SCALE", "0.5"))

# State storage: JSON file (default) or Redis
STATE_BACKEND = os.getenv("STATE_BACKEND", "file")
STATE_FILE = os.getenv(
    "STATE_FILE",
    str(Path(__file__).resolve().parent.parent / "perception_state.json"),
)
STREAM_FACE_FILE = os.getenv(
    "STREAM_FACE_FILE",
    str(Path(__file__).resolve().parent.parent / "stream_face_detection.json"),
)
FACES_DB_PATH = os.getenv(
    "FACES_DB_PATH",
    str(Path(__file__).resolve().parent / "faces"),
)

# Embedding-based recognition (stores vectors, assigns user_1, user_2, ...)
EMBEDDINGS_PATH = os.getenv(
    "EMBEDDINGS_PATH",
    str(Path(__file__).resolve().parent.parent / "face_embeddings.json"),
)
PERSONALIZATION_PATH = os.getenv(
    "PERSONALIZATION_PATH",
    str(Path(__file__).resolve().parent.parent / "personalization.json"),
)
PROACTIVE_STATE_FILE = os.getenv(
    "PROACTIVE_STATE_FILE",
    str(Path(__file__).resolve().parent.parent / "proactive_state.json"),
)
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.65"))
CLEAR_PRIMARY_RATIO = float(os.getenv("CLEAR_PRIMARY_RATIO", "1.5"))  # Primary face must be this many times larger than second
MERGE_SIMILARITY_THRESHOLD = float(os.getenv("MERGE_SIMILARITY_THRESHOLD", "0.62"))  # Merge user_N into name if above this
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Facenet512")
MAX_EMBEDDINGS_PER_USER = int(os.getenv("MAX_EMBEDDINGS_PER_USER", "20"))
_embeddings_lock = threading.Lock()
# TensorFlow/Keras (DeepFace) is not safe for concurrent inference across threads; without
# this, /identify + background stream can segfault or exit with no Python traceback.
_deepface_lock = threading.Lock()

# Redis (optional, used when STATE_BACKEND=redis)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_KEY = "perception:memo"

_redis_client = None
if STATE_BACKEND == "redis":
    try:
        import redis
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        _redis_client.ping()
        log.info("Using Redis for state at %s", REDIS_URL)
    except Exception as exc:
        log.warning("Redis unavailable, falling back to file: %s", exc)
        STATE_BACKEND = "file"

# DeepFace for face identification and stream
_deepface = None


def _get_deepface():
    global _deepface
    if _deepface is None:
        try:
            from deepface import DeepFace
            _deepface = DeepFace
            log.info("DeepFace loaded")
        except ImportError as exc:
            log.warning("DeepFace not available: %s", exc)
    return _deepface


# ── Shared frame buffer ──────────────────────────────────────────────
@dataclass
class FrameBuffer:
    frame: Optional[np.ndarray] = None
    timestamp: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)


_buf = FrameBuffer()
_cap: Optional[cv2.VideoCapture] = None
_cap_lock = threading.Lock()
_shutdown_event = threading.Event()

# Latest raw frame for the analysis thread (written only by smooth capture).
_raw_lock = threading.Lock()
_latest_raw: Optional[np.ndarray] = None

# Overlay drawn on every preview frame (updated by analysis thread).
_overlay_lock = threading.Lock()
_overlay_faces: list = []
_overlay_countdown: Optional[str] = None


def _empty_vision_state() -> dict:
    return {"faces": [], "scene_caption": None, "location": None, "updated_at": None}


# ── State (file-backed) ───────────────────────────────────────────────
def _read_state_file() -> dict:
    if STATE_BACKEND == "redis" and _redis_client:
        try:
            raw = _redis_client.get(REDIS_KEY)
            if raw:
                return json.loads(raw)
        except Exception:
            pass
        return _empty_vision_state()
    p = Path(STATE_FILE)
    if not p.exists():
        return _empty_vision_state()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if "location" not in data:
            data["location"] = None
        return data
    except Exception as exc:
        log.warning("Failed to read state file: %s", exc)
        return _empty_vision_state()


def _write_state_file(data: dict):
    data["updated_at"] = time.time()
    if STATE_BACKEND == "redis" and _redis_client:
        try:
            _redis_client.set(REDIS_KEY, json.dumps(data))
            return
        except Exception as exc:
            log.warning("Redis write failed, falling back to file: %s", exc)
    p = Path(STATE_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _write_stream_face_file(data: dict):
    """Write DeepFace stream output to a separate file (no Redis)."""
    data["updated_at"] = time.time()
    p = Path(STREAM_FACE_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _read_stream_face_file() -> dict:
    """Read DeepFace stream face state."""
    p = Path(STREAM_FACE_FILE)
    if not p.exists():
        return {"faces": [], "updated_at": None}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Failed to read stream face file: %s", exc)
        return {"faces": [], "updated_at": None}


def _load_personalization() -> dict:
    """Load personalization data: { name: context_string, ... }."""
    p = Path(PERSONALIZATION_PATH)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        log.warning("Failed to read personalization: %s", exc)
        return {}


def _build_personalization_context(faces: list) -> str:
    """Build personalization context string for recognized faces."""
    if not faces:
        return ""
    names = {f.get("name") for f in faces if f.get("name") and f.get("name") != "Unknown"}
    if not names:
        return ""
    personalization = _load_personalization()
    parts = [f"- {name}: {personalization[name]}" for name in names if name in personalization]
    return "\n".join(parts) if parts else ""


def _read_proactive_state() -> dict:
    """Read proactive greeting state."""
    p = Path(PROACTIVE_STATE_FILE)
    if not p.exists():
        return {"last_interaction": 0, "last_greeting": 0}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return {
            "last_interaction": float(data.get("last_interaction", 0)),
            "last_greeting": float(data.get("last_greeting", 0)),
        }
    except Exception as exc:
        log.warning("Failed to read proactive state: %s", exc)
        return {"last_interaction": 0, "last_greeting": 0}


def _write_proactive_state(updates: dict):
    """Update proactive state (merge with existing)."""
    state = _read_proactive_state()
    state.update(updates)
    p = Path(PROACTIVE_STATE_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2), encoding="utf-8")


# ── Embedding-based face recognition ─────────────────────────────────
def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [0, 1]. Same person typically >= 0.5."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    sim = dot / (norm_a * norm_b)
    return float(np.clip(sim, 0, 1))


def _load_embeddings() -> dict:
    """
    Load embedding db. Format: { user_1: { embeddings: [[...], [...], ...], first_seen: ts }, ... }.
    Migrates legacy single-embedding format to embeddings list.
    """
    p = Path(EMBEDDINGS_PATH)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
        # Migrate legacy format
        for uid, entry in data.items():
            if isinstance(entry, dict) and "embedding" in entry and "embeddings" not in entry:
                entry["embeddings"] = [entry.pop("embedding", [])]
        return data
    except Exception as exc:
        log.warning("Failed to load embeddings: %s", exc)
        return {}


def _save_embeddings(data: dict):
    p = Path(EMBEDDINGS_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _match_embedding(emb: np.ndarray, add_if_unknown: bool = True) -> Tuple[str, float]:
    """
    Match embedding against stored embeddings. Caller must hold _embeddings_lock.
    Returns (user_id, confidence). Optionally adds new embedding for unknown faces.
    """
    db = _load_embeddings()
    best_id = None
    best_sim = 0.0
    emb = np.asarray(emb, dtype=np.float64)

    for user_id, entry in db.items():
        emb_list = entry.get("embeddings", entry.get("embedding", []))
        if not isinstance(emb_list, list):
            emb_list = [emb_list]
        for stored_raw in emb_list:
            stored = np.array(stored_raw, dtype=np.float64)
            if len(stored) != len(emb):
                continue
            sim = _cosine_similarity(emb, stored)
            if sim > best_sim and sim >= SIMILARITY_THRESHOLD:
                best_sim = sim
                best_id = user_id

    if best_id is not None:
        entry = db[best_id]
        emb_list = entry.get("embeddings", [])
        if not emb_list:
            emb_list = [entry.get("embedding", [])] if entry.get("embedding") else []
        if emb.tolist() not in emb_list and len(emb_list) < MAX_EMBEDDINGS_PER_USER:
            emb_list.append(emb.tolist())
            entry["embeddings"] = emb_list
            if "embedding" in entry:
                del entry["embedding"]
            _save_embeddings(db)
        return best_id, best_sim

    if add_if_unknown:
        next_num = max(
            (int(k.replace("user_", "")) for k in db if k.startswith("user_") and k[5:].isdigit()),
            default=0,
        ) + 1
        new_id = f"user_{next_num}"
        db[new_id] = {"embeddings": [emb.tolist()], "first_seen": time.time()}
        _save_embeddings(db)
        log.info("New face registered: %s", new_id)
        return new_id, 1.0

    return "Unknown", 0.0


def _identify_by_embedding(face_img: np.ndarray, add_if_unknown: bool = True, detector_backend: str = "skip") -> Tuple[str, float]:
    """
    Match face embedding against stored embeddings (single face).
    Returns (user_id, confidence). Uses Facenet512 by default.
    """
    df = _get_deepface()
    if df is None:
        return "Unknown", 0.0
    try:
        with _deepface_lock:
            objs = df.represent(
                img_path=face_img,
                model_name=EMBEDDING_MODEL,
                detector_backend=detector_backend,
                enforce_detection=False,
                align=True,
            )
        if not objs or len(objs) == 0:
            return "Unknown", 0.0
        emb = np.array(objs[0]["embedding"], dtype=np.float64)
    except Exception as exc:
        log.debug("Embedding extraction failed: %s", exc)
        return "Unknown", 0.0

    with _embeddings_lock:
        return _match_embedding(emb, add_if_unknown)


def _identify_all_faces(
    img: np.ndarray, add_if_unknown: bool = True, detector_backend: Optional[str] = None
) -> List[dict]:
    """
    Identify all faces in image. Returns list of { name, confidence } in detection order.
    """
    db = detector_backend or DEEPFACE_DETECTOR_BACKEND
    df = _get_deepface()
    if df is None:
        return []
    try:
        with _deepface_lock:
            objs = df.represent(
                img_path=img,
                model_name=EMBEDDING_MODEL,
                detector_backend=db,
                enforce_detection=False,
                align=True,
            )
        if not objs or len(objs) == 0:
            return []
    except Exception as exc:
        log.debug("Multi-face embedding extraction failed: %s", exc)
        return []

    identities = []
    with _embeddings_lock:
        for obj in objs:
            emb = np.array(obj["embedding"], dtype=np.float64)
            name, confidence = _match_embedding(emb, add_if_unknown)
            identities.append({"name": name, "confidence": round(confidence, 2)})

    return identities


def _register_face_with_merge(name: str, emb: np.ndarray) -> dict:
    """
    Add embedding to name and merge any matching user_N into it.
    Caller must hold _embeddings_lock.
    Returns { ok, name, merged_from: [...] }.
    """
    db = _load_embeddings()
    emb = np.asarray(emb, dtype=np.float64)
    emb_list = [emb.tolist()]
    merged_from = []

    # Find user_N entries that match this embedding (to merge)
    for uid in list(db.keys()):
        if uid == name or not (uid.startswith("user_") and uid[5:].replace("_", "").isdigit()):
            continue
        entry = db[uid]
        emb_list_other = entry.get("embeddings", entry.get("embedding", []))
        if not isinstance(emb_list_other, list):
            emb_list_other = [emb_list_other]
        for stored in emb_list_other:
            st = np.array(stored, dtype=np.float64)
            if len(st) == len(emb) and _cosine_similarity(emb, st) >= MERGE_SIMILARITY_THRESHOLD:
                merged_from.append(uid)
                for e in emb_list_other:
                    if e not in emb_list and len(emb_list) < MAX_EMBEDDINGS_PER_USER:
                        emb_list.append(e if isinstance(e, list) else list(e))
                break

    # Merge: remove merged users, add their embeddings to target
    for uid in merged_from:
        entry = db.get(uid)
        if entry:
            other_list = entry.get("embeddings", entry.get("embedding", []))
            if not isinstance(other_list, list):
                other_list = [other_list]
            for e in other_list:
                el = e if isinstance(e, list) else list(e)
                if el not in emb_list and len(emb_list) < MAX_EMBEDDINGS_PER_USER * 2:
                    emb_list.append(el)
            del db[uid]
            log.info("Merged %s into %s", uid, name)

    # Add/update target
    if name in db:
        existing = db[name].get("embeddings", [])
        for e in emb_list:
            if e not in existing and len(existing) < MAX_EMBEDDINGS_PER_USER:
                existing.append(e)
        db[name]["embeddings"] = existing
    else:
        db[name] = {"embeddings": emb_list[:MAX_EMBEDDINGS_PER_USER], "first_seen": time.time()}

    _save_embeddings(db)
    return {"ok": True, "name": name, "merged_from": merged_from}


# ── Camera helpers ────────────────────────────────────────────────────
def _video_capture(index: int) -> cv2.VideoCapture:
    """Windows: DirectShow. Linux/Jetson: V4L2 (USB webcams). Else: OpenCV default."""
    if sys.platform == "win32":
        return cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if sys.platform.startswith("linux"):
        return cv2.VideoCapture(index, cv2.CAP_V4L2)
    return cv2.VideoCapture(index)


def _open_camera() -> cv2.VideoCapture:
    cap = _video_capture(CAM_INDEX)
    if not cap.isOpened():
        log.error("Failed to open camera at index %d — check CAM_INDEX and that no other process holds the device", CAM_INDEX)
        return cap
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    for _ in range(5):
        cap.read()
        time.sleep(0.02)
    log.info("Camera opened: index=%d, %dx%d", CAM_INDEX, WIDTH, HEIGHT)
    return cap


def _encode_jpeg(frame: np.ndarray, quality: int = JPEG_QUALITY) -> Optional[bytes]:
    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return jpg.tobytes() if ok else None


# ── DeepFace stream loop (like DeepFace.stream() + JSON state) ──────────
def _grab_facial_areas(
    img: np.ndarray,
    detector_backend: Optional[str] = None,
    threshold: Optional[int] = None,
):
    """Detect faces, return list of (x, y, w, h)."""
    db = detector_backend if detector_backend is not None else DEEPFACE_DETECTOR_BACKEND
    min_w = FACE_MIN_WIDTH if threshold is None else threshold
    df = _get_deepface()
    if df is None:
        return []
    try:
        # Downscale for faster detection, then scale coordinates back up
        scale = DETECTION_SCALE
        small = cv2.resize(img, (0, 0), fx=scale, fy=scale) if scale != 1.0 else img
        with _deepface_lock:
            face_objs = df.extract_faces(
                img_path=small,
                detector_backend=db,
                expand_percentage=0,
                enforce_detection=False,
            )
        inv = 1.0 / scale
        return [
            (int(f["facial_area"]["x"] * inv), int(f["facial_area"]["y"] * inv),
             int(f["facial_area"]["w"] * inv), int(f["facial_area"]["h"] * inv))
            for f in face_objs
            if f["facial_area"]["w"] * inv > min_w
        ]
    except Exception:
        return []


def _draw_overlay(img: np.ndarray, faces_data: list, countdown: Optional[str] = None):
    """Draw bounding boxes and labels on frame."""
    for (x, y, w, h), data in faces_data:
        color = (67, 67, 67)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        label_parts = []
        if data.get("arming"):
            label_parts.append(str(data["arming"]))
        if data.get("name"):
            label_parts.append(f"{data['name']} ({data.get('confidence', 0):.0%})")
        if data.get("emotion"):
            label_parts.append(data["emotion"])
        if data.get("age"):
            label_parts.append(f"{int(data['age'])}y")
        if data.get("gender"):
            label_parts.append(data["gender"][:1])
        if label_parts:
            label = " | ".join(label_parts)
            cv2.putText(img, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    if countdown:
        h, w = img.shape[:2]
        cv2.rectangle(img, (10, 10), (90, 50), (67, 67, 67), -1)
        cv2.putText(img, countdown, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


def _smooth_capture_loop():
    """High-FPS camera read + compose latest overlays (decoupled from DeepFace)."""
    try:
        _smooth_capture_loop_inner()
    except Exception:
        log.exception("Smooth capture loop crashed — no more frames will be produced")


def _smooth_capture_loop_inner():
    global _latest_raw
    interval = 1.0 / max(PREVIEW_FPS, 1)
    while not _shutdown_event.is_set():
        with _cap_lock:
            cap = _cap
        if cap is None or not cap.isOpened():
            time.sleep(interval)
            continue
        ok, raw = cap.read()
        if not ok or raw is None:
            time.sleep(interval)
            continue
        with _raw_lock:
            _latest_raw = raw.copy()
        composed = raw.copy()
        with _overlay_lock:
            faces_snapshot = [(box, dict(info)) for box, info in _overlay_faces]
            cd = _overlay_countdown
        if faces_snapshot or cd:
            _draw_overlay(composed, faces_snapshot, cd)
        with _buf.lock:
            _buf.frame = composed
            _buf.timestamp = time.time()
        time.sleep(interval)


def _deepface_stream_loop():
    """Run DeepFace analysis on snapshots (paired with _smooth_capture_loop)."""
    try:
        _deepface_analysis_loop_inner()
    except Exception:
        log.exception("DeepFace analysis loop crashed — overlays will stop updating")


def _deepface_analysis_loop_inner():
    global _overlay_faces, _overlay_countdown
    df = _get_deepface()
    if df is None:
        return
    detector_backend = DEEPFACE_DETECTOR_BACKEND
    frame_threshold = max(STREAM_FRAME_THRESHOLD, 1)
    time_threshold = max(STREAM_TIME_THRESHOLD, 1)
    arm_count = 0
    in_hold = False
    hold_tic = 0.0
    last_faces_data: list = []
    analysis_sleep = max(ANALYSIS_INTERVAL, 0.02)

    while not _shutdown_event.is_set():
        time.sleep(analysis_sleep)
        with _raw_lock:
            snap = None if _latest_raw is None else _latest_raw.copy()
        if snap is None:
            continue

        now = time.time()
        if in_hold and (now - hold_tic < time_threshold):
            time_left = int(time_threshold - (now - hold_tic) + 1)
            with _overlay_lock:
                _overlay_faces = [(box, dict(info)) for box, info in last_faces_data]
                _overlay_countdown = str(time_left)
            continue

        if in_hold and (now - hold_tic >= time_threshold):
            in_hold = False
            arm_count = 0
            log.info("DeepFace smooth: hold released")
            with _overlay_lock:
                _overlay_countdown = None

        faces_coords = _grab_facial_areas(snap, detector_backend)
        if faces_coords:
            arm_count += 1
        else:
            arm_count = 0

        trigger = (
            arm_count > 0
            and arm_count % frame_threshold == 0
            and faces_coords
        )
        if trigger:
            faces_data = []
            for (x, y, w, h) in faces_coords:
                detected_face = snap[y : y + h, x : x + w]
                name, confidence = _identify_by_embedding(
                    detected_face, add_if_unknown=True
                )
                emotion, age, gender = "neutral", None, None
                try:
                    with _deepface_lock:
                        dem = df.analyze(
                            detected_face,
                            actions=("age", "gender", "emotion"),
                            detector_backend="skip",
                            enforce_detection=False,
                            silent=True,
                        )
                    if dem and len(dem) > 0:
                        d = dem[0]
                        emotion = d.get("dominant_emotion", "neutral")
                        age = d.get("age")
                        gender = (d.get("dominant_gender") or "unknown")[:1]
                except Exception as exc:
                    log.debug("Demography failed: %s", exc)
                faces_data.append(
                    (
                        (x, y, w, h),
                        {
                            "name": name,
                            "confidence": confidence,
                            "emotion": emotion,
                            "age": age,
                            "gender": gender,
                        },
                    )
                )
            last_faces_data = faces_data
            state_faces = [
                {
                    "name": d["name"],
                    "emotion": d.get("emotion"),
                    "age": d.get("age"),
                    "gender": d.get("gender"),
                    "confidence": d.get("confidence", 0),
                }
                for _, d in faces_data
            ]
            _write_stream_face_file({"faces": state_faces})
            in_hold = True
            hold_tic = now
            arm_count = 0
            log.info("DeepFace smooth: analyzed %d face(s)", len(faces_data))
            with _overlay_lock:
                _overlay_faces = [(box, dict(info)) for box, info in faces_data]
                _overlay_countdown = None
        else:
            overlay = []
            if faces_coords:
                rem = frame_threshold - (arm_count % frame_threshold)
                count_str = str(rem)
                for (x, y, w, h) in faces_coords:
                    overlay.append(
                        (
                            (x, y, w, h),
                            {
                                "name": "",
                                "confidence": 0.0,
                                "emotion": "",
                                "age": None,
                                "gender": "",
                                "arming": count_str,
                            },
                        )
                    )
            with _overlay_lock:
                _overlay_faces = overlay
                if not in_hold:
                    _overlay_countdown = None


def _simple_capture_loop(fps: int = CAPTURE_FPS):
    """Original capture loop without DeepFace stream."""
    try:
        _simple_capture_loop_inner(fps)
    except Exception:
        log.exception("Simple capture loop crashed — no more frames will be produced")


def _simple_capture_loop_inner(fps: int = CAPTURE_FPS):
    interval = 1.0 / fps
    while not _shutdown_event.is_set():
        with _cap_lock:
            cap = _cap
        if cap is None or not cap.isOpened():
            time.sleep(interval)
            continue
        ret, frame = cap.read()
        if ret and frame is not None:
            with _buf.lock:
                _buf.frame = frame
                _buf.timestamp = time.time()
        time.sleep(interval)


# ── Change detection + webhook push thread ────────────────────────────
def _change_detection_loop():
    prev_frame: Optional[np.ndarray] = None
    last_push = 0.0

    while not _shutdown_event.is_set():
        with _buf.lock:
            frame = _buf.frame.copy() if _buf.frame is not None else None

        if frame is None:
            time.sleep(CHANGE_CHECK_INTERVAL)
            continue

        if prev_frame is not None:
            diff = _frame_diff(prev_frame, frame)
            now = time.time()

            if diff > CHANGE_THRESHOLD and (now - last_push) > MIN_PUSH_INTERVAL:
                jpg = _encode_jpeg(frame)
                if jpg is not None:
                    log.info("Scene change detected (diff=%.1f), pushing to n8n", diff)
                    try:
                        requests.post(
                            N8N_WEBHOOK_URL,
                            files={"data": ("frame.jpg", jpg, "image/jpeg")},
                            data={"diff_score": str(round(diff, 2))},
                            timeout=5,
                        )
                        last_push = now
                    except Exception as exc:
                        log.warning("Webhook push failed: %s", exc)

        prev_frame = frame
        time.sleep(CHANGE_CHECK_INTERVAL)


def _frame_diff(a: np.ndarray, b: np.ndarray) -> float:
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    return float(np.mean(cv2.absdiff(ga, gb)))


# ── QR code detector (created once, reused across threads) ───────────
_qr_detector = cv2.QRCodeDetector()


def _scan_qr_all(bgr: np.ndarray) -> List[Tuple[str, Optional[np.ndarray]]]:
    """
    Detect all QR codes in frame.
    Returns list of (decoded_text, corners_4x2) tuples (only non-empty decoded values).
    """
    results: List[Tuple[str, Optional[np.ndarray]]] = []
    ok, decoded_info, points, _ = _qr_detector.detectAndDecodeMulti(bgr)
    if ok and points is not None:
        for i, text in enumerate(decoded_info):
            if text:
                pts = points[i].reshape(4, 2).astype(np.float32) if i < len(points) else None
                results.append((text, pts))
    return results


def _persist_location_from_qr_hits(hits: List[Tuple[str, Optional[np.ndarray]]]) -> None:
    """Write state.location from the first decoded QR value."""
    for text, _ in hits:
        if text:
            state = _read_state_file()
            if state.get("location") != text:
                state["location"] = text
                _write_state_file(state)
                log.info("Location from QR code: %s", text)
            return


def _draw_qr_hits_on_bgr(vis: np.ndarray, hits: List[Tuple[str, Optional[np.ndarray]]]) -> None:
    """Draw QR code outlines and decoded labels on BGR image (mutates vis)."""
    for text, pts in hits:
        label = text[:40]
        if pts is not None and pts.size >= 8:
            pi = pts.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pi], True, (64, 255, 64), 2)
            x, y = int(pi[0, 0, 0]), int(pi[0, 0, 1])
            y = max(y - 8, 20)
        else:
            x, y = 12, 28
        cv2.putText(vis, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 255, 220), 1, cv2.LINE_AA)


def _qr_scan_loop():
    """Background: detect QR codes in the live frame and update state.location."""
    interval = max(QR_SCAN_INTERVAL, 0.2)
    while not _shutdown_event.is_set():
        try:
            with _buf.lock:
                frame = _buf.frame.copy() if _buf.frame is not None else None
            if frame is not None:
                hits = _scan_qr_all(frame)
                _persist_location_from_qr_hits(hits)
        except Exception:
            log.exception("QR scan loop error")
        time.sleep(interval)


# ── Lifecycle ─────────────────────────────────────────────────────────
@app.on_event("startup")
def startup_event():
    global _cap
    _cap = _open_camera()
    if not _cap.isOpened():
        log.error("Camera unavailable at startup — frame buffer will be empty until /reopen succeeds")
    Path(FACES_DB_PATH).mkdir(parents=True, exist_ok=True)
    _get_deepface()

    if DEEPFACE_STREAM and _get_deepface():
        threading.Thread(target=_smooth_capture_loop, daemon=True, name="capture-smooth").start()
        threading.Thread(target=_deepface_stream_loop, daemon=True, name="deepface-analyze").start()
        log.info(
            "Smooth preview %d FPS + DeepFace analysis every %.3fs (arm=%d ticks, hold=%ds)",
            PREVIEW_FPS,
            ANALYSIS_INTERVAL,
            STREAM_FRAME_THRESHOLD,
            STREAM_TIME_THRESHOLD,
        )
    elif DEEPFACE_STREAM:
        log.warning("DEEPFACE_STREAM enabled but DeepFace unavailable — using simple capture")
        threading.Thread(target=_simple_capture_loop, daemon=True, name="capture").start()
        log.info("Simple capture loop started at %d FPS", CAPTURE_FPS)
    else:
        threading.Thread(target=_simple_capture_loop, daemon=True, name="capture").start()
        log.info("Simple capture loop started at %d FPS", CAPTURE_FPS)

    if PUSH_ENABLED and not DEEPFACE_STREAM:
        threading.Thread(
            target=_change_detection_loop, daemon=True, name="change-detect"
        ).start()
        log.info("Change detection enabled (webhook=%s)", N8N_WEBHOOK_URL)
    else:
        if DEEPFACE_STREAM:
            log.info("Change detection disabled (DeepFace stream mode)")
        else:
            log.info("Change-detection push is disabled")

    if QR_SCAN_ENABLED:
        threading.Thread(target=_qr_scan_loop, daemon=True, name="qr-scan").start()
        log.info(
            "QR location scan enabled (interval=%.2fs, stream_overlay=%s)",
            QR_SCAN_INTERVAL,
            QR_STREAM_OVERLAY,
        )
    else:
        log.info("QR location scan disabled")

    log.info(
        "State: %s, file: %s, stream_faces: %s, embeddings: %s, stream_mode: %s, preview_fps: %s",
        STATE_BACKEND,
        STATE_FILE,
        STREAM_FACE_FILE,
        EMBEDDINGS_PATH,
        DEEPFACE_STREAM,
        PREVIEW_FPS if DEEPFACE_STREAM and _get_deepface() else CAPTURE_FPS,
    )


@app.on_event("shutdown")
def shutdown_event():
    global _cap
    _shutdown_event.set()
    with _cap_lock:
        if _cap is not None:
            _cap.release()
            _cap = None


# ── Vision state ──────────────────────────────────────────────────────
@app.post("/state")
def update_state(payload: dict = Body(...)):
    """Merge vision state from n8n (faces, scene_caption, etc.). Preserves keys omitted from payload (e.g. location)."""
    data = _read_state_file()
    data.update(payload)
    _write_state_file(data)
    return {"ok": True}


@app.get("/state")
def get_state():
    """Read the latest vision state (n8n/VLM combined: faces + scene_caption + location)."""
    state = _read_state_file()
    faces = state.get("faces") or []
    ctx = _build_personalization_context(faces)
    if ctx:
        state["personalization_context"] = ctx
    return state


@app.get("/stream-faces")
def get_stream_faces():
    """Read face detection from DeepFace stream loop (separate from /state)."""
    return _read_stream_face_file()


# ── Proactive greeting state ─────────────────────────────────────────
@app.get("/proactive-state")
def get_proactive_state():
    """Return last_interaction and last_greeting timestamps for proactive greeting logic."""
    return _read_proactive_state()


@app.post("/proactive-interaction")
def record_proactive_interaction():
    """Record that the user just interacted (ASR or chat). Call from MainWorkflow."""
    _write_proactive_state({"last_interaction": time.time()})
    return {"ok": True}


@app.post("/proactive-greeting-done")
def record_proactive_greeting_done():
    """Record that a proactive greeting was just played. Call after greeting TTS."""
    _write_proactive_state({"last_greeting": time.time()})
    return {"ok": True}


# ── Face identification (embedding-based: user_1, user_2, ...) ─────────
@app.post("/identify")
async def identify(data: Optional[UploadFile] = File(None), add_if_unknown: bool = True):
    """Identify face via embeddings. Multipart 'data' or latest frame. Returns user_N or Unknown."""
    img_bytes = None
    if data and data.filename:
        img_bytes = await data.read()
    else:
        with _buf.lock:
            frame = _buf.frame
        if frame is not None:
            img_bytes = _encode_jpeg(frame)
    if not img_bytes:
        return {"name": "Unknown", "confidence": 0, "reason": "no image provided"}

    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"name": "Unknown", "confidence": 0, "reason": "invalid image"}

    name, confidence = _identify_by_embedding(
        img, add_if_unknown=add_if_unknown, detector_backend=DEEPFACE_DETECTOR_BACKEND
    )
    return {"name": name, "confidence": round(confidence, 2)}


@app.post("/identify-multi")
async def identify_multi(data: Optional[UploadFile] = File(None), add_if_unknown: bool = True):
    """Identify all faces in image. Returns one identity per face in detection order."""
    img_bytes = None
    if data and data.filename:
        img_bytes = await data.read()
    else:
        with _buf.lock:
            frame = _buf.frame
        if frame is not None:
            img_bytes = _encode_jpeg(frame)
    if not img_bytes:
        return {"identities": [], "reason": "no image provided"}

    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"identities": [], "reason": "invalid image"}

    identities = _identify_all_faces(
        img, add_if_unknown=add_if_unknown, detector_backend=DEEPFACE_DETECTOR_BACKEND
    )
    return {"identities": identities}


@app.post("/register-face")
async def register_face(request: Request):
    """
    Register a face with the given name. Uses provided image or latest frame.
    Accepts JSON body {"name": "..."} or form name=... and optional file.
    - 0 faces: no new faces to register
    - 1 unknown: register it
    - 0 unknowns, 1 face (user_N): merge/rename that user into the name
    - 2+ unknowns: register only if clear primary (largest ≥ 1.5× second)
    When registering, merges any matching user_N into the new name.
    """
    name = None
    img_bytes = None
    ct = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
    if ct == "application/json":
        try:
            body = await request.json()
            name = body.get("name") if isinstance(body, dict) else None
        except Exception:
            pass
    else:
        form = await request.form()
        name = form.get("name")
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        if isinstance(name, str) and name.strip():
            pass
        else:
            name = None
        # File can be under "data", "file", or "image"
        for key in ("data", "file", "image"):
            f = form.get(key)
            if hasattr(f, "read") and hasattr(f, "filename") and f.filename:
                img_bytes = await f.read()
                break
    if not name or not str(name).strip():
        return JSONResponse(
            {"ok": False, "message": "Name is required."},
            status_code=400,
        )
    name = str(name).strip()

    if not img_bytes:
        with _buf.lock:
            frame = _buf.frame
        if frame is not None:
            img_bytes = _encode_jpeg(frame)
    if not img_bytes:
        return JSONResponse(
            {"ok": False, "message": "I don't see any image. Please make sure I can see you, or send a photo."},
            status_code=400,
        )

    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse(
            {"ok": False, "message": "Invalid image."},
            status_code=400,
        )

    df = _get_deepface()
    if df is None:
        return JSONResponse(
            {"ok": False, "message": "Face recognition is not available."},
            status_code=503,
        )

    try:
        with _deepface_lock:
            objs = df.represent(
                img_path=img,
                model_name=EMBEDDING_MODEL,
                detector_backend=DEEPFACE_DETECTOR_BACKEND,
                enforce_detection=False,
                align=True,
            )
    except Exception as e:
        log.warning("register-face represent failed: %s", e)
        return JSONResponse(
            {"ok": False, "message": "Could not analyze the image."},
            status_code=500,
        )

    if not objs:
        return {
            "ok": False,
            "message": "I don't see any new faces to register.",
            "suggest_reply": "I don't see any faces in the image. Please come closer or face the camera.",
        }

    # Get identity for each face without adding new unknowns
    with _embeddings_lock:
        face_info = []
        for obj in objs:
            emb = np.array(obj.get("embedding", []), dtype=np.float64)
            fa = obj.get("facial_area") or {}
            w = int(fa.get("w", 0))
            h = int(fa.get("h", 0))
            area = w * h
            identity, conf = _match_embedding(emb, add_if_unknown=False)
            face_info.append({"embedding": emb, "area": area, "identity": identity, "confidence": conf})

    unknowns = [f for f in face_info if f["identity"] == "Unknown"]
    known = [f for f in face_info if f["identity"] != "Unknown"]

    # 0 unknowns, 1 face (user_N): merge/rename that user into the name
    if len(unknowns) == 0 and len(face_info) == 1:
        with _embeddings_lock:
            out = _register_face_with_merge(name, face_info[0]["embedding"])
        return {
            "ok": True,
            "message": f"Nice to meet you, {name}! I've learned your face.",
            "name": name,
            "merged_from": out.get("merged_from", []),
            "suggest_reply": f"Nice to meet you, {name}! I'll remember you.",
        }

    # 0 unknowns, 2+ faces
    if len(unknowns) == 0:
        return {
            "ok": False,
            "message": "I don't see any new faces to register.",
            "suggest_reply": "I already know everyone I see. If you're new, please come closer so I can register you.",
        }

    # 1 unknown: register it
    if len(unknowns) == 1:
        with _embeddings_lock:
            out = _register_face_with_merge(name, unknowns[0]["embedding"])
        return {
            "ok": True,
            "message": f"Nice to meet you, {name}! I've learned your face.",
            "name": name,
            "merged_from": out.get("merged_from", []),
            "suggest_reply": f"Nice to meet you, {name}! I'll remember you.",
        }

    # 2+ unknowns: require clear primary
    sorted_unknowns = sorted(unknowns, key=lambda f: f["area"], reverse=True)
    area0 = sorted_unknowns[0]["area"]
    area1 = sorted_unknowns[1]["area"] if len(sorted_unknowns) > 1 else 0
    if area1 <= 0 or area0 >= CLEAR_PRIMARY_RATIO * area1:
        with _embeddings_lock:
            out = _register_face_with_merge(name, sorted_unknowns[0]["embedding"])
        return {
            "ok": True,
            "message": f"Nice to meet you, {name}! I've learned your face.",
            "name": name,
            "merged_from": out.get("merged_from", []),
            "suggest_reply": f"Nice to meet you, {name}! I'll remember you.",
        }

    return {
        "ok": False,
        "message": "I see multiple people. Can you come closer so I can recognize you, or introduce yourself when you're the only one in frame?",
        "suggest_reply": "I see multiple people. Can you come closer so I can recognize you, or introduce yourself when you're the only one in frame?",
    }


# ── Camera & stream endpoints ────────────────────────────────────────
@app.get("/health")
def health():
    with _buf.lock:
        has_frame = _buf.frame is not None
        ts = _buf.timestamp
    return {
        "ok": True,
        "streaming": has_frame,
        "last_frame_ts": ts,
        "push_enabled": PUSH_ENABLED,
        "state_backend": STATE_BACKEND,
        "state_file": STATE_FILE,
        "stream_face_file": STREAM_FACE_FILE,
        "deepface_stream": DEEPFACE_STREAM,
        "preview_fps": PREVIEW_FPS if DEEPFACE_STREAM else CAPTURE_FPS,
        "analysis_interval_s": ANALYSIS_INTERVAL if DEEPFACE_STREAM else None,
        "face_min_width": FACE_MIN_WIDTH,
        "deepface_detector_backend": DEEPFACE_DETECTOR_BACKEND,
    }


@app.post("/reopen")
def reopen():
    """Re-initialise the camera."""
    global _cap
    with _cap_lock:
        if _cap is not None:
            _cap.release()
        _cap = _open_camera()
    return {"reopened": True}


@app.get("/latest")
def latest(quality: int = Query(JPEG_QUALITY, ge=30, le=100)):
    """Return the most recent buffered frame (with DeepFace overlays when stream mode)."""
    with _buf.lock:
        frame = _buf.frame
    if frame is None:
        return JSONResponse({"error": "no frame available yet"}, status_code=503)
    jpg = _encode_jpeg(frame, quality)
    if jpg is None:
        return JSONResponse({"error": "jpeg encoding failed"}, status_code=500)
    return Response(content=jpg, media_type="image/jpeg")


@app.get("/snapshot")
def snapshot(quality: int = Query(JPEG_QUALITY, ge=30, le=100)):
    """Backwards-compatible: returns latest frame."""
    return latest(quality)


@app.get("/stream")
def stream(
    fps: int = Query(60, ge=1, le=60),
    quality: int = Query(80, ge=30, le=100),
):
    """MJPEG stream with DeepFace overlays; optional QR outline/label when QR_STREAM_OVERLAY=true."""
    return StreamingResponse(
        _mjpeg_generator(fps, quality),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


def _mjpeg_generator(fps: int, quality: int):
    interval = 1.0 / fps
    while not _shutdown_event.is_set():
        with _buf.lock:
            frame = _buf.frame
        if frame is not None:
            to_encode = frame
            if QR_SCAN_ENABLED:
                hits = _scan_qr_all(frame)
                _persist_location_from_qr_hits(hits)
                if QR_STREAM_OVERLAY and hits:
                    to_encode = frame.copy()
                    _draw_qr_hits_on_bgr(to_encode, hits)
            jpg = _encode_jpeg(to_encode, quality)
            if jpg:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                )
        time.sleep(interval)


@app.get("/snapshot_to_file")
def snapshot_to_file(
    path: str = Query("current_frame.jpg"),
    quality: int = Query(JPEG_QUALITY, ge=30, le=100),
):
    """Save the latest buffered frame to a file."""
    with _buf.lock:
        frame = _buf.frame
    if frame is None:
        return JSONResponse({"error": "no frame available yet"}, status_code=503)
    ok = cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return JSONResponse({"error": f"failed to write: {path}"}, status_code=500)
    return {"saved": True, "path": path}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8089)
