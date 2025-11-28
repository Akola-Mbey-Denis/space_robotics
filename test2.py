import mujoco
import mujoco.viewer
import numpy as np

# Load the model and create data
model = mujoco.MjModel.from_xml_path("rockie_bogie.xml")
data = mujoco.MjData(model)

# Get body ID using the new API
rover_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rover_body")

with mujoco.viewer.launch_passive(model, data) as viewer:
    t = 0.0
    while viewer.is_running():
        # Apply simple oscillating control to all actuators
        data.ctrl[:] = 0.50 * np.sin(2 * np.pi * 0.5 * t)
        
        # Step simulation
        mujoco.mj_step(model, data)
        viewer.sync()
        
        # Get rover position
        pos = data.xpos[rover_id]
        print(f"t={t:.2f}s | Pos={pos}")
        
        # Increment time
        t += model.opt.timestep
