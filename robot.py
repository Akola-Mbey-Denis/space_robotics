import mujoco
from mujoco import viewer
import numpy as np

# Load MuJoCo model
model = mujoco.MjModel.from_xml_path("rockie_bogie.xml")
data = mujoco.MjData(model)

# Create renderer for stereo cameras
renderer = mujoco.Renderer(model, height=480, width=640)

# Get camera IDs
cam_left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "stereo_left")
cam_right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "stereo_right")

# List of wheel motor names (these are actuator names, not joint names)
wheel_motors = [
    "motor_fl", "motor_ml", "motor_rl",  # Left side
    "motor_fr", "motor_mr", "motor_rr"   # Right side
]

# Get motor IDs
motor_ids = []
for motor_name in wheel_motors:
    motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, motor_name)
    motor_ids.append(motor_id)

# Control parameters
target_velocity = 1.0  # Forward velocity
turn_rate = 0.0  # Differential steering (-1 to 1)

# Simulation parameters
step_count = 0
capture_interval = 500  # Capture stereo images every N steps

# Launch viewer
print("Starting simulation...")
print(f"Number of actuators: {model.nu}")
print(f"Motor IDs: {motor_ids}")


# Define controller function
def controller(model, data):
    global step_count, target_velocity, turn_rate
    
    # Calculate differential drive velocities
    left_velocity = target_velocity - turn_rate
    right_velocity = target_velocity + turn_rate
    
    # Apply motor controls
    # Left side motors (indices 0, 1, 2)
    data.ctrl[motor_ids[0]] = left_velocity
    data.ctrl[motor_ids[1]] = left_velocity
    data.ctrl[motor_ids[2]] = left_velocity
    
    # Right side motors (indices 3, 4, 5)
    data.ctrl[motor_ids[3]] = right_velocity
    data.ctrl[motor_ids[4]] = right_velocity
    data.ctrl[motor_ids[5]] = right_velocity
    
    # Capture stereo images periodically
    if step_count % capture_interval == 0:
        # Render left camera
        renderer.update_scene(data, camera="stereo_left")
        left_image = renderer.render()
        
        # Render right camera
        renderer.update_scene(data, camera="stereo_right")
        right_image = renderer.render()
        
        print(f"Step {step_count}: Captured stereo pair")
        print(f"  Image shapes: Left {left_image.shape}, Right {right_image.shape}")
        print(f"  Chassis position: ({data.qpos[0]:.2f}, {data.qpos[1]:.2f}, {data.qpos[2]:.2f})")
        
        # Optional: Save images as numpy arrays
        if step_count == capture_interval:
            np.save('stereo_left.npy', left_image)
            np.save('stereo_right.npy', right_image)
            print("  Saved stereo images as .npy files")
    
    # Print sensor data periodically
    if step_count % 1000 == 0 and step_count > 0:
        print(f"\n--- Step {step_count} ---")
        print(f"Rocker angles: L={np.rad2deg(data.sensordata[0]):.1f}°, R={np.rad2deg(data.sensordata[1]):.1f}°")
        print(f"Bogie angles: L={np.rad2deg(data.sensordata[2]):.1f}°, R={np.rad2deg(data.sensordata[3]):.1f}°")
    
    # Example: Change direction after certain steps
    if step_count == 3000:
        print("\nTurning right...")
        turn_rate = 0.3
    elif step_count == 6000:
        print("\nGoing straight again...")
        turn_rate = 0.0
    elif step_count == 9000:
        print("\nTurning left...")
        turn_rate = -0.3
    elif step_count == 12000:
        print("\nStopping...")
        target_velocity = 0.0
        turn_rate = 0.0
    
    step_count += 1


# Launch viewer with controller
print("Use mouse to rotate view, scroll to zoom")
print("Press Ctrl+C in terminal to stop simulation\n")

# For macOS: Use handle to control the viewer
v = viewer.launch(model, data)

# Manual simulation loop
try:
    while v.is_running():
        controller(model, data)
        mujoco.mj_step(model, data)
        v.sync()
except KeyboardInterrupt:
    print("\nSimulation stopped by user")
finally:
    v.close()

print("\nSimulation ended")