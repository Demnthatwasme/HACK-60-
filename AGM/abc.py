import cv2
import time
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    import romp
except ImportError:
    print("WARNING: Please install ROMP:")
    print("pip install romp")
    romp = None

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
    'Pelvis': 'pelvis', 'L_Hip': 'left_hip', 'R_Hip': 'right_hip',
    'L_Knee': 'left_knee', 'R_Knee': 'right_knee', 'L_Ankle': 'left_foot',
    'R_Ankle': 'right_foot', 'Spine3': 'spine3', 'L_Shoulder': 'left_shoulder',
    'R_Shoulder': 'right_shoulder', 'L_Elbow': 'left_elbow',
    'R_Elbow': 'right_elbow', 'L_Wrist': 'left_wrist', 'R_Wrist': 'right_wrist'
}

class ROMPGMRPipeline:
    def __init__(self, robot_xml_path):
        self.robot_name = robot_xml_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
       
        if romp is not None:
            settings = romp.main.default_settings
            settings.show_largest_person_only = True
            self.romp_model = romp.ROMP(settings)
            print("ROMP Model Initialized.")
        else:
            self.romp_model = None

        if GeneralMotionRetargeting is not None:
            print(f"Loading GMR with robot: {robot_xml_path}")
            self.retargeter = GeneralMotionRetargeting("smplx", robot_xml_path, use_velocity_limit=True)
            print("GMR Retargeter Initialized.")
        else:
            self.retargeter = None

    def process_frame(self, frame):
        if self.romp_model is None: return None
        outputs = self.romp_model(frame)
        if outputs is None or len(outputs.get('smpl_thetas', [])) == 0: return None

        smpl_thetas = outputs['smpl_thetas'][0]
        cam_trans = outputs['cam_trans'][0]
        joints_3d = outputs.get('joints', [None])[0]

        axis_angles = np.array(smpl_thetas).reshape(-1, 3)
       
        SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
        global_rots = []
        for i in range(24):
            if i == 0:
                rot = R.from_rotvec(axis_angles[0])
            else:
                rot = global_rots[SMPL_PARENTS[i]] * R.from_rotvec(axis_angles[i])
            global_rots.append(rot)
           
        rotation_matrix = np.array([
            [1,  0,  0],
            [0,  0,  1],
            [0, -1,  0]
        ])
        gmr_rot_correction = R.from_matrix(rotation_matrix)

        human_motion_dict = {}
        for i, joint_name in enumerate(SMPL_JOINT_NAMES):
            if i >= 24: break
            final_rot = (gmr_rot_correction * global_rots[i]).as_quat(scalar_first=True)

            if joints_3d is not None and i < len(joints_3d):
                translation = joints_3d[i]
            else:
                translation = cam_trans if i == 0 else np.zeros(3)

            final_pos = translation @ rotation_matrix.T
            gmr_name = SMPL_TO_GMR_NAMES.get(joint_name, joint_name.lower())
            human_motion_dict[gmr_name] = (final_pos, final_rot)

        return human_motion_dict

    def run_webcam(self, camera_source=0, headless=False):
        cap = cv2.VideoCapture(camera_source)
       
        # Standard OpenCV buffer limit for IP cameras
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
       
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        print("Starting Real-Time Pipeline. Press 'q' to quit (or ^C if headless).")
        prev_time = time.time()
       
        viewer = None
        if self.retargeter is not None and not headless:
            try:
                from general_motion_retargeting.robot_motion_viewer import RobotMotionViewer
                viewer = RobotMotionViewer(self.robot_name)
            except Exception as e:
                print("Could not initialize 3D viewer:", e)

        # --- JITTER STABILIZER VARIABLES ---
        smoothed_qpos = None
        # ALPHA controls the smoothing amount.
        # 1.0 = No smoothing (Maximum Jitter)
        # 0.1 = Max smoothing (Looks like moving through honey)
        # 0.5 = Balanced for robotics
        ALPHA = 0.5

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Rotate the frame 90 degrees clockwise for phone cameras
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            human_motion_dict = self.process_frame(frame)

            if human_motion_dict is not None and self.retargeter is not None:
                raw_robot_qpos = self.retargeter.retarget(human_motion_dict, offset_to_ground=True)
               
                # --- APPLY SMOOTHING TO STOP JITTER ---
                if smoothed_qpos is None:
                    smoothed_qpos = np.array(raw_robot_qpos)
                else:
                    # Blend the new frame with the previous frame
                    smoothed_qpos = (ALPHA * np.array(raw_robot_qpos)) + ((1.0 - ALPHA) * smoothed_qpos)
                   
                    # Normalize the quaternion (indices 3:7) so the robot joints don't distort
                    quat = smoothed_qpos[3:7]
                    smoothed_qpos[3:7] = quat / np.linalg.norm(quat)
               
                if viewer is not None and viewer.viewer.is_running():
                    root_pos = smoothed_qpos[:3]
                    root_rot = smoothed_qpos[3:7]
                    dof_pos = smoothed_qpos[7:]
                    viewer.step(root_pos, root_rot, dof_pos, rate_limit=False)
               
                cv2.putText(frame, "Tracking & Retargeting Active", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No Person Detected", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            if not headless:
                cv2.imshow("Webcam to GMR (ROMP)", frame)
                if viewer is not None and not viewer.viewer.is_running():
                    break
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        if viewer is not None:
            try:
                viewer.close()
            except:
                pass

        cap.release()
        if not headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    ROBOT_NAME = "unitree_g1"
    CAMERA_SOURCE = "http://172.18.33.137:8080/video"
    HEADLESS = False
   
    pipeline = ROMPGMRPipeline(ROBOT_NAME)
   
    print("\n" + "="*50)
    print("AI Models and Kinematics loaded successfully!")
    input("Press ENTER to open the camera and 3D viewer...")
    print("="*50 + "\n")
   
    pipeline.run_webcam(camera_source=CAMERA_SOURCE, headless=HEADLESS)