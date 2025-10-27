
from dm_control import mjcf

# Load your URDF
model = mjcf.from_path("/Users/denismbeyakola/Desktop/space_robotics/rover.urdf")

# Export to MuJoCo XML format
mjcf.export_with_assets(model, "rocker_bogie_mujoco.xml")
