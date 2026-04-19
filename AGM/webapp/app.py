"""
app.py — G1 Mission Control Flask server.

Manages two background threads:
  • inference_loop()        — ROMP→GMR→MuJoCo (right panel)
  • vibe_classifier_loop()  — MediaPipe skeleton + posture (left panel)

Runs cleanly in MOCK MODE on Windows dev machine (no heavy deps needed).
Zero-config deployment on target Linux machine: cd webapp && python app.py
"""
import sys
import os

# ── sys.path: find AGM-level packages BEFORE any local imports ───────────────
_HERE   = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
sys.path.insert(0, _PARENT)
sys.path.insert(0, os.path.join(_PARENT, 'GMR'))

# ── Standard library ─────────────────────────────────────────────────────────
import threading
import time

# ── Third-party (always available) ───────────────────────────────────────────
import cv2
import numpy as np
from collections import deque
from flask import Flask, render_template, Response, jsonify

# ── Local modules ─────────────────────────────────────────────────────────────
from config import (
    CAMERA_SOURCE, USE_PHONE_CAMERA, ROBOT_NAME,
    FLASK_HOST, FLASK_PORT, JPEG_QUALITY,
    MJPEG_SLEEP, EMA_ALPHA, FLOOR_OFFSET,
    RENDER_WIDTH, RENDER_HEIGHT, MOCK_MODE
)
from pipeline import (
    CameraStream, process_frame, smooth_qpos,
    init_romp, init_gmr, init_mujoco_renderer,
    generate_mock_frame,
    ROMP_OK, GMR_OK, MUJOCO_OK
)
from vibe import init_mediapipe_detector, detect_posture, MEDIAPIPE_OK

# Import mujoco only if available (for rendering calls inside the loop)
if MUJOCO_OK:
    import mujoco  # noqa: F811

# ─────────────────────────────────────────────────────────────────────────────
#  T3.1 / T3.2 — Flask app + global state
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)

inference_running: bool        = False
lock                           = threading.Lock()
latest_vibe_frame:  bytes|None = None   # LEFT panel  — VIBE annotated skeleton
latest_robot_frame: bytes|None = None   # RIGHT panel — MuJoCo G1 render
latest_webcam_raw:  "np.ndarray|None" = None  # raw BGR shared to VIBE thread
current_posture:    str        = "Unknown"

metrics: dict = {
    "fps":              0,
    "inference_ms":     0,
    "retarget_ms":      0,
    "render_ms":        0,
    "person_detected":  False,
    "status":           "idle",
    "frame_count":      0,
    "posture":          "Unknown",
    "mock_mode":        MOCK_MODE,
}
fps_buffer = deque(maxlen=30)


# ─────────────────────────────────────────────────────────────────────────────
#  T3.3 — inference_loop: ROMP→GMR→MuJoCo render → latest_robot_frame
# ─────────────────────────────────────────────────────────────────────────────
def inference_loop():
    global latest_robot_frame, latest_webcam_raw, inference_running, metrics

    # ── Init phase ───────────────────────────────────────────────────────────
    metrics["status"] = "loading models..."

    romp_model  = init_romp()
    retargeter  = init_gmr(ROBOT_NAME)
    mj_model, mj_data, mj_renderer = init_mujoco_renderer(
        ROBOT_NAME, RENDER_WIDTH, RENDER_HEIGHT
    )

    # Camera — with mock fallback
    cap = None
    if not MOCK_MODE:
        cap = CameraStream(CAMERA_SOURCE)
        if not cap.isOpened():
            print("[inference] Primary camera failed — trying laptop webcam (0)")
            cap = CameraStream(0)
            if not cap.isOpened():
                print("[inference] No camera — falling back to mock frames")
                cap = None

    # Static mock frames (reused every loop iteration in mock mode)
    mock_webcam = generate_mock_frame(
        RENDER_WIDTH, RENDER_HEIGHT,
        "WEBCAM PLACEHOLDER", (100, 200, 100)
    )
    mock_robot = generate_mock_frame(
        RENDER_WIDTH, RENDER_HEIGHT,
        "MUJOCO G1 PLACEHOLDER", (100, 150, 255)
    )

    metrics["status"] = "ready (mock)" if MOCK_MODE else "ready"

    prev_qpos = None
    prev_time = time.time()

    # ── Main loop ────────────────────────────────────────────────────────────
    while True:
        if not inference_running:
            time.sleep(0.05)
            continue

        loop_start = time.time()

        # ── Capture ──────────────────────────────────────────────────────────
        if cap is not None:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            if USE_PHONE_CAMERA:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        else:
            frame = mock_webcam.copy()

        # Share raw frame for VIBE thread to consume
        latest_webcam_raw = frame.copy()

        # ── ROMP inference ───────────────────────────────────────────────────
        human_motion_dict = None
        if romp_model is not None:
            t0 = time.time()
            try:
                human_motion_dict = process_frame(frame, romp_model)
            except Exception as e:
                if metrics["frame_count"] % 200 == 0:
                    print(f"[WARN] ROMP: {e}")
            metrics["inference_ms"] = round((time.time() - t0) * 1000, 1)

        # ── GMR retarget ─────────────────────────────────────────────────────
        robot_qpos = None
        if human_motion_dict is not None and retargeter is not None:
            t1 = time.time()
            try:
                robot_qpos = retargeter.retarget(
                    human_motion_dict, offset_to_ground=True
                )
            except Exception as e:
                if metrics["frame_count"] % 200 == 0:
                    print(f"[WARN] Retarget: {e}")
            metrics["retarget_ms"] = round((time.time() - t1) * 1000, 1)

        # ── EMA smoothing ────────────────────────────────────────────────────
        if robot_qpos is not None:
            if prev_qpos is None:
                prev_qpos = robot_qpos.copy()
            else:
                robot_qpos, prev_qpos = smooth_qpos(robot_qpos, prev_qpos, EMA_ALPHA)

        # ── MuJoCo render (or mock fallback) ─────────────────────────────────
        if robot_qpos is not None and mj_renderer is not None:
            t2 = time.time()
            try:
                root_pos = robot_qpos[:3].copy()
                root_pos[2] -= FLOOR_OFFSET

                mj_data.qpos[:3]  = root_pos
                mj_data.qpos[3:7] = robot_qpos[3:7]
                mj_data.qpos[7:]  = robot_qpos[7:]

                mujoco.mj_forward(mj_model, mj_data)
                mj_renderer.update_scene(mj_data)   # single call — no double-update bug
                robot_rgb = mj_renderer.render()
                robot_bgr = cv2.cvtColor(robot_rgb, cv2.COLOR_RGB2BGR)

                _, buf = cv2.imencode(
                    ".jpg", robot_bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                )
                latest_robot_frame = buf.tobytes()
                metrics["render_ms"] = round((time.time() - t2) * 1000, 1)
            except Exception as e:
                if metrics["frame_count"] % 200 == 0:
                    print(f"[WARN] MuJoCo render: {e}")
                _encode_mock(mock_robot)
        else:
            _encode_mock(mock_robot)

        # ── FPS ──────────────────────────────────────────────────────────────
        now = time.time()
        dt  = max(now - prev_time, 0.001)
        fps_buffer.append(1.0 / dt)
        prev_time = now
        metrics["fps"]              = round(float(np.mean(fps_buffer)), 1)
        metrics["posture"]          = current_posture
        metrics["frame_count"]     += 1
        metrics["person_detected"]  = (human_motion_dict is not None)
        metrics["status"]           = "running"


def _encode_mock(mock_frame):
    """Helper: JPEG-encode the mock robot frame and store in latest_robot_frame."""
    global latest_robot_frame
    _, buf = cv2.imencode(
        ".jpg", mock_frame,
        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    )
    latest_robot_frame = buf.tobytes()


# ─────────────────────────────────────────────────────────────────────────────
#  T3.4 — vibe_classifier_loop: produces LEFT panel content
# ─────────────────────────────────────────────────────────────────────────────
def vibe_classifier_loop():
    global latest_vibe_frame, current_posture

    model_dir = os.path.join(_PARENT, 'vibe_mocap', 'vibe_mocap')
    detector  = init_mediapipe_detector(model_dir)
    # detector=None in mock mode — detect_posture() handles gracefully

    while True:
        if latest_webcam_raw is None:
            time.sleep(0.1)
            continue

        frame_copy = latest_webcam_raw.copy()

        posture, annotated_frame = detect_posture(detector, frame_copy)
        current_posture = posture

        _, buf = cv2.imencode(
            ".jpg", annotated_frame,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )
        latest_vibe_frame = buf.tobytes()

        time.sleep(0.03)   # ~30 Hz — smooth enough for the left panel


# ─────────────────────────────────────────────────────────────────────────────
#  T3.5 — gen_stream: MJPEG multipart generator
# ─────────────────────────────────────────────────────────────────────────────
def gen_stream(stream_type: str):
    """
    MJPEG multipart generator.
    stream_type: 'vibe'  → left panel (annotated skeleton)
                 'robot' → right panel (MuJoCo sim)
    """
    while True:
        if stream_type == "vibe":
            frame_bytes = latest_vibe_frame
        else:
            frame_bytes = latest_robot_frame

        if frame_bytes is None:
            time.sleep(0.1)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes +
            b"\r\n"
        )
        time.sleep(MJPEG_SLEEP)   # 16ms ≈ 60fps cap


# ─────────────────────────────────────────────────────────────────────────────
#  T3.6 — Flask routes
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/vibe_feed")
def vibe_feed():
    """LEFT panel — annotated VIBE skeleton stream."""
    return Response(
        gen_stream("vibe"),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/robot_feed")
def robot_feed():
    """RIGHT panel — MuJoCo G1 simulation stream."""
    return Response(
        gen_stream("robot"),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/toggle_inference")
def toggle_inference():
    global inference_running
    with lock:
        inference_running = not inference_running
        status = "running" if inference_running else "stopped"
        metrics["status"] = status
    return jsonify({"status": status, "inference_running": inference_running})


@app.route("/metrics")
def get_metrics():
    return jsonify(metrics)


@app.route("/health")
def health():
    return jsonify({
        "romp":      ROMP_OK,
        "gmr":       GMR_OK,
        "mujoco":    MUJOCO_OK,
        "vibe":      MEDIAPIPE_OK,
        "mock_mode": MOCK_MODE,
    })


@app.route("/posture")
def get_posture():
    return jsonify({"posture": current_posture})


# ─────────────────────────────────────────────────────────────────────────────
#  T3.7 — __main__
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _LINE = "=" * 44
    _mode = "MOCK" if MOCK_MODE else "LIVE"
    _romp = "OK" if ROMP_OK else "MISSING"
    _gmr  = "OK" if GMR_OK  else "MISSING"
    _muj  = "OK" if MUJOCO_OK else "MISSING"
    _vib  = "OK" if MEDIAPIPE_OK else "MISSING"

    banner = (
        "\n" + _LINE + "\n"
        "  G1 MISSION CONTROL -- Web Dashboard\n" +
        _LINE + "\n"
        f"  Mode:    {_mode}\n"
        f"  ROMP:    {_romp}\n"
        f"  GMR:     {_gmr}\n"
        f"  MuJoCo: {_muj}\n"
        f"  VIBE:    {_vib}\n"
        f"  Camera:  {CAMERA_SOURCE}\n"
        "\n"
        f"  http://localhost:{FLASK_PORT}\n" +
        _LINE + "\n"
    )
    print(banner)

    threading.Thread(target=inference_loop,       daemon=True).start()
    threading.Thread(target=vibe_classifier_loop, daemon=True).start()

    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False)
