import cv2
import time
import torch
import numpy as np
import threading
import requests
from scipy.spatial.transform import Rotation as R

# Try to import ROMP
try:
    import romp
except ImportError:
    print("WARNING: Please install ROMP:")
    print("pip install romp")
    romp = None

# Try to import GMR
try:
    from general_motion_retargeting.motion_retarget import GeneralMotionRetargeting
    from general_motion_retargeting.params import IK_CONFIG_DICT
    import json
except ImportError:
    print("WARNING: GMR not found. Please clone https://github.com/YanjieZe/GMR and run `pip install -e .` inside it.")
    GeneralMotionRetargeting = None

# Standard SMPL Joint Names
SMPL_JOINT_NAMES = [
    'Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2', 
    'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 
    'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 
    'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand'
]

SMPL_TO_GMR_NAMES = {
    'Pelvis': 'pelvis',
    'L_Hip': 'left_hip',
    'R_Hip': 'right_hip',
    'L_Knee': 'left_knee',
    'R_Knee': 'right_knee',
    'L_Ankle': 'left_foot',
    'R_Ankle': 'right_foot',
    'Spine3': 'spine3',
    'L_Shoulder': 'left_shoulder',
    'R_Shoulder': 'right_shoulder',
    'L_Elbow': 'left_elbow',
    'R_Elbow': 'right_elbow',
    'L_Wrist': 'left_wrist',
    'R_Wrist': 'right_wrist'
}

import threading

class CameraStream:
    """
    A threaded video capture stream that continuously pulls frames in the background.
    This entirely eliminates OpenCV's internal network buffering lag!
    """
    def __init__(self, src):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        
        # Start the thread to read frames from the video stream
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while not self.stopped:
            # Continuously grab frames to clear the buffer
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.grabbed, self.frame

    def release(self):
        self.stopped = True
        self.thread.join()
        self.stream.release()

    def isOpened(self):
        return self.stream.isOpened()

class PostureClient:
    """
    Background thread to independently poll the teammate's MediaPipe API.
    This prevents HTTP requests from blocking or lagging the main pipeline!
    """
    def __init__(self, url="http://localhost:8080/get_pose"):
        self.url = url
        self.posture = "Waiting for VIBE API..."
        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()
        
    def update(self):
        while self.running:
            try:
                resp = requests.get(self.url, timeout=0.1)
                if resp.status_code == 200:
                    self.posture = resp.json().get("posture", "Unknown")
            except Exception:
                pass
            time.sleep(0.05) # Poll 20 times a second

class ROMPGMRPipeline:
    def __init__(self, robot_xml_path):
        self.robot_name = robot_xml_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # 1. Initialize ROMP Model
        if romp is not None:
            settings = romp.main.default_settings
            settings.show_largest_person_only = True # Fast, single-person mode
            self.romp_model = romp.ROMP(settings)
            print("ROMP Model Initialized.")
        else:
            self.romp_model = None

        # 2. Initialize GMR Retargeter
        if GeneralMotionRetargeting is not None:
            print(f"Loading GMR with robot: {robot_xml_path}")
            
            # GMR init expects "smplx" as the key in its config dictionary.
            # We lower damping to 0.02 to prevent the IK solver from "freezing", but keep enough to avoid jitter.
            self.retargeter = GeneralMotionRetargeting("smplx", robot_xml_path, use_velocity_limit=True, damping=0.02)
            print("GMR Retargeter Initialized.")
        else:
            self.retargeter = None

        # Ground Calibration State
        self.fixed_ground_offset = None
        
        # Start independent API poller for teammate's VIBE classifier
        self.posture_client = PostureClient()

    def process_frame(self, frame):
        if self.romp_model is None:
            return None

        # Run ROMP inference
        # ROMP takes a BGR numpy image directly
        outputs = self.romp_model(frame)
        
        # Check if a person was detected
        if outputs is None or len(outputs.get('smpl_thetas', [])) == 0:
            return None

        # Extract parameters for the largest person
        # smpl_thetas contains 72 values (24 joints * 3 axis-angle params)
        smpl_thetas = outputs['smpl_thetas'][0] 
        cam_trans = outputs['cam_trans'][0] # (3,) global translation
        
        # ROMP also provides 3D joints which can be used as global translations for each joint
        joints_3d = outputs.get('joints', [None])[0]

        # Convert axis-angle representations to wxyz quaternions for GMR
        axis_angles = np.array(smpl_thetas).reshape(-1, 3)
        
        # Rigorous Forward Kinematics to calculate GLOBAL joint rotations
        SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
        global_rots = []
        for i in range(24):
            if i == 0:
                rot = R.from_rotvec(axis_angles[0])
            else:
                rot = global_rots[SMPL_PARENTS[i]] * R.from_rotvec(axis_angles[i])
            global_rots.append(rot)
            
        # Correct OpenCV (Y-down, Z-forward) to MuJoCo (Z-up, Y-forward)
        rotation_matrix = np.array([
            [1,  0,  0],
            [0,  0,  1],
            [0, -1,  0]
        ])
        gmr_rot_correction = R.from_matrix(rotation_matrix)

        human_motion_dict = {}

        # Map the 24 standard SMPL joints
        for i, joint_name in enumerate(SMPL_JOINT_NAMES):
            if i >= 24:
                break
                
            # Apply correction to get perfect global rotation
            final_rot = (gmr_rot_correction * global_rots[i]).as_quat(scalar_first=True)

            # Use specific joint translation if available
            if joints_3d is not None and i < len(joints_3d):
                translation = joints_3d[i]
            else:
                translation = cam_trans if i == 0 else np.zeros(3)

            # Apply correction to global translation
            final_pos = translation @ rotation_matrix.T

            # Map to GMR specific joint names (e.g. L_Hip -> left_hip)
            gmr_name = SMPL_TO_GMR_NAMES.get(joint_name, joint_name.lower())
            
            # Store as tuple (pos_3d, rot_4d)
            human_motion_dict[gmr_name] = (final_pos, final_rot)

        return human_motion_dict

    def run_webcam(self, camera_source=0, headless=False):
        # Use our custom threaded stream to bypass OpenCV's network buffer lag
        cap = CameraStream(camera_source)
        
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        print("Starting Real-Time Pipeline. Press 'q' to quit (or ^C if headless).")
        
        # Metrics
        prev_time = time.time()
        
        # Initialize 3D Viewer
        viewer = None
        if self.retargeter is not None and not headless:
            try:
                from general_motion_retargeting.robot_motion_viewer import RobotMotionViewer
                viewer = RobotMotionViewer(self.robot_name)
            except Exception as e:
                print("Could not initialize 3D viewer:", e)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Rotate the frame 90 degrees clockwise for phone cameras
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            # --- BROADCAST FRAME TO VIBE CLASSIFIER VIA RAMDISK ---
            # Linux /dev/shm is stored completely in RAM, so this is ultra-fast (sub-1ms)
            # We use an atomic os.rename to prevent the VIBE script from reading a half-written JPEG
            import os
            cv2.imwrite('/dev/shm/droidcam_frame_tmp.jpg', frame)
            os.rename('/dev/shm/droidcam_frame_tmp.jpg', '/dev/shm/droidcam_frame.jpg')

            # 1. Pose Estimation (Webcam -> ROMP)
            human_motion_dict = self.process_frame(frame)

            # 2. Motion Retargeting (ROMP -> GMR)
            if human_motion_dict is not None and self.retargeter is not None:
                
                # Retarget to robot kinematics
                # offset_to_ground=True mathematically forces the lowest foot to the floor.
                # This makes squats work perfectly, but makes jumping impossible.
                robot_qpos = self.retargeter.retarget(human_motion_dict, offset_to_ground=True)
                
                # --- ROBOT JITTER FILTERING (EMA) ---
                # Smoothing the final robot joint positions instead of raw human poses 
                # avoids mathematical bugs with axis-angles.
                alpha = 0.4 # Higher = more responsive/less lag, Lower = smoother (0.4 is a sweet spot)
                if not hasattr(self, 'prev_qpos'):
                    self.prev_qpos = robot_qpos.copy()
                else:
                    # Fix quaternion sign flipping before interpolation
                    q1 = self.prev_qpos[3:7]
                    q2 = robot_qpos[3:7]
                    if np.dot(q1, q2) < 0:
                        robot_qpos[3:7] = -q2
                    
                    self.prev_qpos = alpha * robot_qpos + (1 - alpha) * self.prev_qpos
                    # Re-normalize the root quaternion after interpolation
                    self.prev_qpos[3:7] /= np.linalg.norm(self.prev_qpos[3:7])
                    robot_qpos = self.prev_qpos.copy()
                # ------------------------------------
                
                # Update 3D Viewer
                if viewer is not None and viewer.viewer.is_running():
                    root_pos = robot_qpos[:3].copy()
                    
                    # --- VISUAL FLOOR FIX ---
                    # GMR mathematically forces the feet to Z=0.1, which leaves the 
                    # robot visually floating an inch or two above the MuJoCo grid. 
                    # We subtract a small amount here to make the rubber feet visually flush.
                    root_pos[2] -= 0.06  # Lower by 6cm
                    
                    root_rot = robot_qpos[3:7]
                    dof_pos = robot_qpos[7:]
                    viewer.step(root_pos, root_rot, dof_pos, rate_limit=False)
                
                cv2.putText(frame, "Tracking & Retargeting Active", (20, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No Person Detected", (20, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Display Teammate's VIBE Posture Classification
            cv2.putText(frame, f"VIBE Posture: {self.posture_client.posture}", (20, 110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)

            # Compute and show FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            if not headless:
                # Check if 3D viewer was closed
                if viewer is not None and not viewer.viewer.is_running():
                    break
                    
            # Check for reset flag sent by VIBE classifier
            if os.path.exists('/dev/shm/reset_robot'):
                os.remove('/dev/shm/reset_robot')
                if self.retargeter is not None:
                    # Resets the internal mink IK solver state to default T-pose
                    self.retargeter.setup_retarget_configuration()
                if hasattr(self, 'prev_qpos'):
                    del self.prev_qpos
                print("\nReceived Reset Signal from VIBE! Robot state reset to default neutral pose.\n")

        if viewer is not None:
            try:
                viewer.close()
            except:
                pass

        cap.release()
        if not headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    # GMR uses a string name mapping like "unitree_h1", "unitree_g1", etc.
    # Check GMR/general_motion_retargeting/params.py for actual names.
    ROBOT_NAME = "unitree_g1"
    
    # DroidCam over ADB port forwarding
    CAMERA_SOURCE = "http://localhost:4747/video"
    
    # Set to True to run headlessly (no windows, processes in the background)
    HEADLESS = False
    
    pipeline = ROMPGMRPipeline(ROBOT_NAME)
    
    print("\n" + "="*50)
    print("AI Models and Kinematics loaded successfully!")
    input("Press ENTER to open the camera and 3D viewer...")
    print("="*50 + "\n")
    
    pipeline.run_webcam(camera_source=CAMERA_SOURCE, headless=HEADLESS)
