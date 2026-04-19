import sys
import os
import cv2
import threading
import time
from flask import Flask, render_template, Response, jsonify

# Add GMR to path manually just in case the editable install is acting up
sys.path.append(os.path.join(os.getcwd(), 'GMR'))

# Import your existing logic from run.py
from run import ROMPGMRPipeline, CameraStream

app = Flask(__name__)

# Global Control State
inference_active = False
latest_webcam_frame = None
latest_robot_frame = None
lock = threading.Lock()

def bg_inference_loop():
    global latest_webcam_frame, latest_robot_frame, inference_active
    
    # Initialize the Pipeline for Unitree G1
    pipeline = ROMPGMRPipeline("unitree_g1")
    cap = CameraStream(0) # Using laptop webcam

    while True:
        if not inference_active:
            time.sleep(0.1)
            continue

        ret, frame = cap.read()
        if not ret: continue

        # 1. Vision Processing
        human_motion_dict = pipeline.process_frame(frame)

        # 2. Kinematics & Rendering
        if human_motion_dict and pipeline.retargeter:
            # Calculate Robot Joint Angles (qpos)
            robot_qpos = pipeline.retargeter.retarget(human_motion_dict, offset_to_ground=True)
            
            # Update MuJoCo and Capture Pixels
            # Note: We use the renderer instead of the viewer for web streaming
            pipeline.viewer.render() # Assuming your pipeline has a render method
            
            cv2.putText(frame, "TRACKING ACTIVE", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "SCANNING FOR PERSON...", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Encode webcam frame for Flask
        _, buffer = cv2.imencode('.jpg', frame)
        with lock:
            latest_webcam_frame = buffer.tobytes()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with lock:
                if latest_webcam_frame:
                    yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                           latest_webcam_frame + b'\r\n')
            time.sleep(0.03) # 30fps stream
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle')
def toggle():
    global inference_active
    inference_active = not inference_active
    return jsonify({"status": "Online" if inference_active else "Offline"})

if __name__ == '__main__':
    # Start the AI loop in the background
    threading.Thread(target=bg_inference_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=False)