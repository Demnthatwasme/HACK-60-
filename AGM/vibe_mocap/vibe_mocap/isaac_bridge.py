import requests
import numpy as np
# Assuming you have Isaac Gym imported as 'gym' in your existing sim
# from isaacgym import gymapi 

def get_mocap_data():
    try:
        response = requests.get("http://localhost:8080/get_pose", timeout=0.01)
        return np.array(response.json()["landmarks"])
    except:
        return None

# --- Inside your Isaac Gym Main Loop ---
# while simulation_running:
#     data = get_mocap_data()
#     if data is not None:
#         # Map MediaPipe Index 24 (Right Hip) to Isaac Humanoid Root
#         root_pos = data[24] 
#         print(f"Syncing Humanoid to position: {root_pos}")