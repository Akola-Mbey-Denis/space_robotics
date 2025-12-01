# """
# Enhanced Rocker-Bogie Rover Navigation Controller
# Features:
# - Adaptive stability control
# - Terrain-aware navigation
# - Systematic image capture
# - Path planning and waypoint following
# - Emergency response system
# """

# import numpy as np
# import mujoco
# import time
# import os
# from dataclasses import dataclass
# from typing import Tuple, Optional, List
# import matplotlib.pyplot as plt
# import cv2
# import imageio
# from collections import deque
# import json

# os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


# @dataclass
# class RoverState:
#     """Current state of the rover"""
#     chassis_pos: np.ndarray
#     chassis_quat: np.ndarray
#     chassis_vel: np.ndarray
#     chassis_gyro: np.ndarray
#     wheel_velocities: np.ndarray
#     wheel_positions: np.ndarray
#     timestamp: float


# @dataclass
# class Waypoint:
#     """Navigation waypoint"""
#     position: np.ndarray  # [x, y]
#     tolerance: float = 0.5
#     capture_image: bool = True
#     max_speed: float = 0.5


# class StabilityMonitor:
#     """Enhanced stability monitoring with predictive warnings"""
    
#     def __init__(self, moon_gravity=True):
#         if moon_gravity:
#             self.max_pitch = np.deg2rad(75)
#             self.max_roll = np.deg2rad(75)
#             self.critical_pitch = np.deg2rad(60)
#             self.critical_roll = np.deg2rad(60)
#         else:
#             self.max_pitch = np.deg2rad(45)
#             self.max_roll = np.deg2rad(45)
#             self.critical_pitch = np.deg2rad(35)
#             self.critical_roll = np.deg2rad(35)
        
#         self.stability_history = deque(maxlen=50)
#         self.min_height = 0.1
#         self.safe_height = 0.2
        
#     def get_roll_pitch_yaw(self, quat: np.ndarray) -> Tuple[float, float, float]:
#         """Convert quaternion to Euler angles"""
#         w, x, y, z = quat
        
#         # Roll (x-axis)
#         sinr_cosp = 2 * (w * x + y * z)
#         cosr_cosp = 1 - 2 * (x * x + y * y)
#         roll = np.arctan2(sinr_cosp, cosr_cosp)
        
#         # Pitch (y-axis)
#         sinp = 2 * (w * y - z * x)
#         pitch = np.arcsin(np.clip(sinp, -1, 1))
        
#         # Yaw (z-axis)
#         siny_cosp = 2 * (w * z + x * y)
#         cosy_cosp = 1 - 2 * (y * y + z * z)
#         yaw = np.arctan2(siny_cosp, cosy_cosp)
        
#         return roll, pitch, yaw
    
#     def check_stability(self, state: RoverState) -> Tuple[bool, float, str]:
#         """Comprehensive stability check with graded warnings"""
#         roll, pitch, yaw = self.get_roll_pitch_yaw(state.chassis_quat)
        
#         # Critical failures
#         if abs(pitch) > self.max_pitch:
#             return False, 0.0, f"CRITICAL: Pitch {np.rad2deg(pitch):.1f}°"
        
#         if abs(roll) > self.max_roll:
#             return False, 0.0, f"CRITICAL: Roll {np.rad2deg(roll):.1f}°"
        
#         if state.chassis_pos[2] < self.min_height:
#             return False, 0.0, "CRITICAL: Chassis collision"
        
#         angular_vel_mag = np.linalg.norm(state.chassis_gyro)
#         if angular_vel_mag > 8.0:  # Increased from 5.0
#             return False, 0.0, f"CRITICAL: Angular velocity {angular_vel_mag:.1f} rad/s"
        
#         # Calculate stability margins
#         pitch_margin = 1.0 - abs(pitch) / self.max_pitch
#         roll_margin = 1.0 - abs(roll) / self.max_roll
#         height_margin = min(1.0, (state.chassis_pos[2] - self.min_height) / self.safe_height)
#         angular_margin = max(0.0, 1.0 - angular_vel_mag / 8.0)  # Increased from 5.0
        
#         stability_margin = min(pitch_margin, roll_margin, height_margin, angular_margin)
#         self.stability_history.append(stability_margin)
        
#         # Trend analysis
#         if len(self.stability_history) > 10:
#             recent_trend = np.mean(list(self.stability_history)[-10:])
#             if recent_trend < 0.3:
#                 return True, stability_margin, f"WARNING: Degrading stability (trend: {recent_trend:.2f})"
        
#         # Warning levels
#         if stability_margin < 0.3:
#             return True, stability_margin, f"CAUTION: Low margin {stability_margin:.2f}"
#         elif stability_margin < 0.5:
#             return True, stability_margin, f"Alert: Moderate stability {stability_margin:.2f}"
        
#         return True, stability_margin, "Stable"
    
#     def get_recommended_speed(self, stability_margin: float) -> float:
#         """Get speed recommendation based on stability"""
#         if stability_margin > 0.8:
#             return 1.0  # Full speed
#         elif stability_margin > 0.5:
#             return 0.7  # Moderate speed
#         elif stability_margin > 0.3:
#             return 0.4  # Reduced speed
#         else:
#             return 0.1  # Crawl speed


# class ImageCaptureSystem:
#     """Systematic image capture with metadata logging"""
    
#     def __init__(self, model, data, output_dir="mission_data"):
#         self.model = model
#         self.data = data
#         self.output_dir = output_dir
#         self.capture_count = 0
        
#         # Create directories
#         self.rgb_dir = os.path.join(output_dir, "rgb")
#         self.depth_dir = os.path.join(output_dir, "depth")
#         self.metadata_dir = os.path.join(output_dir, "metadata")
        
#         for dir_path in [self.rgb_dir, self.depth_dir, self.metadata_dir]:
#             os.makedirs(dir_path, exist_ok=True)
        
#         # Setup renderer
#         self.width = model.vis.global_.offwidth
#         self.height = model.vis.global_.offheight
#         self.renderer = mujoco.Renderer(model, height=self.height, width=self.width)
        
#         # Find cameras
#         self.cameras = {}
#         for cam_name in ['camera_left', 'camera_right']:
#             try:
#                 cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
#                 self.cameras[cam_name] = cam_id
#             except:
#                 pass
        
#         if not self.cameras:
#             print("⚠️ No stereo cameras found, using default camera")
#             self.cameras = {'default': -1}
        
#         print(f"📷 Image capture system initialized with cameras: {list(self.cameras.keys())}")
    
#     def capture(self, state: RoverState, waypoint_id: int = -1, reason: str = "waypoint") -> dict:
#         """Capture images from all cameras with metadata"""
#         capture_data = {
#             'capture_id': self.capture_count,
#             'timestamp': state.timestamp,
#             'waypoint_id': waypoint_id,
#             'reason': reason,
#             'position': state.chassis_pos.tolist(),
#             'orientation': state.chassis_quat.tolist(),
#             'cameras': {}
#         }
        
#         for cam_name, cam_id in self.cameras.items():
#             # RGB capture
#             self.renderer.update_scene(self.data, camera=cam_id)
#             rgb = self.renderer.render()
            
#             rgb_filename = f"{self.capture_count:06d}_{cam_name}_rgb.png"
#             rgb_path = os.path.join(self.rgb_dir, rgb_filename)
#             cv2.imwrite(rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            
#             # Depth capture
#             depth = self._capture_depth(cam_id)
#             depth_filename = f"{self.capture_count:06d}_{cam_name}_depth.exr"
#             depth_path = os.path.join(self.depth_dir, depth_filename)
#             cv2.imwrite(depth_path, depth)
            
#             # Camera pose
#             cam_pos = self.data.cam_xpos[cam_id] if cam_id >= 0 else state.chassis_pos
#             cam_mat = self.data.cam_xmat[cam_id].reshape(3, 3) if cam_id >= 0 else np.eye(3)
            
#             capture_data['cameras'][cam_name] = {
#                 'rgb_file': rgb_filename,
#                 'depth_file': depth_filename,
#                 'position': cam_pos.tolist(),
#                 'orientation': cam_mat.tolist()
#             }
        
#         # Save metadata
#         metadata_file = os.path.join(self.metadata_dir, f"{self.capture_count:06d}_metadata.json")
#         with open(metadata_file, 'w') as f:
#             json.dump(capture_data, f, indent=2)
        
#         self.capture_count += 1
#         print(f"📸 Captured image set {self.capture_count} at position ({state.chassis_pos[0]:.2f}, {state.chassis_pos[1]:.2f})")
        
#         return capture_data
    
#     def _capture_depth(self, cam_id: int) -> np.ndarray:
#         """Capture depth image"""
#         scene = mujoco.MjvScene(self.model, maxgeom=1000)
#         ctx = mujoco.MjrContext(self.model, mujoco.mjtFramebuffer.mjFB_OFFSCREEN)
        
#         cam = mujoco.MjvCamera()
#         if cam_id >= 0:
#             cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
#             cam.fixedcamid = cam_id
#         else:
#             cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
#             cam.trackbodyid = 0
        
#         opt = mujoco.MjvOption()
#         viewport = mujoco.MjrRect(0, 0, self.width, self.height)
        
#         depth = np.zeros((self.height, self.width), dtype=np.float32)
#         rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
#         mujoco.mjv_updateScene(self.model, self.data, opt, None, cam, 
#                               mujoco.mjtCatBit.mjCAT_ALL, scene)
#         mujoco.mjr_render(viewport, scene, ctx)
#         mujoco.mjr_readPixels(rgb, depth, viewport, ctx)
        
#         return depth


# class PathPlanner:
#     """Waypoint-based path planning"""
    
#     def __init__(self):
#         self.waypoints: List[Waypoint] = []
#         self.current_waypoint_idx = 0
#         self.waypoint_reached_callback = None
        
#     def add_waypoint(self, x: float, y: float, tolerance: float = 0.5, 
#                     capture_image: bool = True, max_speed: float = 0.5):
#         """Add waypoint to path"""
#         wp = Waypoint(
#             position=np.array([x, y]),
#             tolerance=tolerance,
#             capture_image=capture_image,
#             max_speed=max_speed
#         )
#         self.waypoints.append(wp)
    
#     def get_current_waypoint(self) -> Optional[Waypoint]:
#         """Get current target waypoint"""
#         if self.current_waypoint_idx < len(self.waypoints):
#             return self.waypoints[self.current_waypoint_idx]
#         return None
    
#     def check_waypoint_reached(self, position: np.ndarray) -> bool:
#         """Check if current waypoint is reached"""
#         waypoint = self.get_current_waypoint()
#         if waypoint is None:
#             return False
        
#         distance = np.linalg.norm(position[:2] - waypoint.position)
#         if distance <= waypoint.tolerance:
#             print(f"✓ Waypoint {self.current_waypoint_idx + 1}/{len(self.waypoints)} reached")
#             self.current_waypoint_idx += 1
#             return True
        
#         return False
    
#     def get_navigation_command(self, current_pos: np.ndarray, current_yaw: float) -> Tuple[float, float]:
#         """Calculate linear and angular velocity commands"""
#         waypoint = self.get_current_waypoint()
#         if waypoint is None:
#             return 0.0, 0.0
        
#         # Vector to waypoint
#         to_waypoint = waypoint.position - current_pos[:2]
#         distance = np.linalg.norm(to_waypoint)
        
#         if distance < 0.1:
#             return 0.0, 0.0
        
#         # Desired heading
#         desired_yaw = np.arctan2(to_waypoint[1], to_waypoint[0])
#         yaw_error = desired_yaw - current_yaw
        
#         # Normalize angle to [-pi, pi]
#         yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))
        
#         # Calculate velocities
#         linear_vel = min(waypoint.max_speed, distance * 0.3)  # Reduced from 0.5
#         angular_vel = np.clip(yaw_error * 1.0, -0.3, 0.3)  # Reduced from 1.5 and 0.5
        
#         # Reduce linear velocity when turning
#         if abs(yaw_error) > np.deg2rad(15):  # Reduced from 30
#             linear_vel *= 0.3  # More aggressive reduction
        
#         return linear_vel, angular_vel
    
#     def is_mission_complete(self) -> bool:
#         """Check if all waypoints reached"""
#         return self.current_waypoint_idx >= len(self.waypoints)


# class PIDController:
#     """PID controller for motor control"""
    
#     def __init__(self, kp: float, ki: float, kd: float):
#         self.kp = kp
#         self.ki = ki
#         self.kd = kd
#         self.integral = 0.0
#         self.prev_error = 0.0
#         self.integral_limit = 5.0  # Reduced from 10.0
    
#     def update(self, error: float, dt: float) -> float:
#         self.integral += error * dt
#         self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        
#         derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
#         self.prev_error = error
        
#         return self.kp * error + self.ki * self.integral + self.kd * derivative
    
#     def reset(self):
#         """Reset PID state"""
#         self.integral = 0.0
#         self.prev_error = 0.0


# class VelocityController:
#     """Differential drive velocity controller"""
    
#     def __init__(self):
#         self.wheel_radius = 0.05
#         self.track_width = 0.4
#         self.max_wheel_velocity = 2.0  # Reduced from 3.3
        
#     def differential_drive(self, linear_vel: float, angular_vel: float) -> np.ndarray:
#         """Compute wheel velocities for differential drive"""
#         v_left = linear_vel - (angular_vel * self.track_width / 2.0)
#         v_right = linear_vel + (angular_vel * self.track_width / 2.0)
        
#         omega_left = v_left / self.wheel_radius
#         omega_right = v_right / self.wheel_radius
        
#         omega_left = np.clip(omega_left, -self.max_wheel_velocity, self.max_wheel_velocity)
#         omega_right = np.clip(omega_right, -self.max_wheel_velocity, self.max_wheel_velocity)
        
#         return np.array([
#             omega_left, omega_left, omega_left,  # Left wheels
#             omega_right, omega_right, omega_right  # Right wheels
#         ])


# class EnhancedRoverController:
#     """Main enhanced rover controller with navigation and imaging"""
    
#     def __init__(self, model, data, mission_name="rover_mission"):
#         self.model = model
#         self.data = data
#         self.mission_name = mission_name
        
#         # Core systems
#         self.stability_monitor = StabilityMonitor(moon_gravity=True)
#         self.velocity_controller = VelocityController()
#         self.path_planner = PathPlanner()
#         self.image_capture = ImageCaptureSystem(model, data, 
#                                                output_dir=f"mission_data/{mission_name}")
        
#         # Motor control
#         self._get_indices()
#         self.wheel_pid = [PIDController(kp=10.0, ki=0.5, kd=0.3) for _ in range(6)]  # Further reduced
        
#         # State
#         self.emergency_stop = False
#         self.last_capture_time = 0.0
#         self.capture_interval = 2.0  # Capture every 2 seconds
#         self.previous_motor_commands = np.zeros(6)  # For smooth acceleration
#         self.max_motor_change = 0.5  # Maximum change per step
        
#         # Telemetry
#         self.telemetry = {
#             'time': [],
#             'position': [],
#             'stability': [],
#             'velocity': [],
#             'captures': [],
#             'roll': [],
#             'pitch': [],
#             'yaw': [],
#             'angular_velocity': []
#         }
        
#         print(f"🤖 Enhanced Rover Controller initialized for mission: {mission_name}")
    
#     def _get_indices(self):
#         """Get motor and sensor indices"""
#         self.motor_names = [
#             'left_bogie_front_axle', 'left_bogie_rear_axle', 'left_rocker_rear_axle',
#             'right_bogie_front_axle', 'right_bogie_rear_axle', 'right_rocker_rear_axle'
#         ]
        
#         self.motor_ids = []
#         for name in self.motor_names:
#             try:
#                 motor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
#                 self.motor_ids.append(motor_id)
#             except:
#                 print(f"⚠️ Motor {name} not found")
        
#         self.sensor_vel_names = [name + '_v' for name in self.motor_names]
#         self.sensor_pos_names = [name + '_p' for name in self.motor_names]
    
#     def get_sensor_data(self, sensor_name: str) -> np.ndarray:
#         """Read sensor data"""
#         try:
#             sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
#             sensor_adr = self.model.sensor_adr[sensor_id]
#             sensor_dim = self.model.sensor_dim[sensor_id]
#             return self.data.sensordata[sensor_adr:sensor_adr + sensor_dim].copy()
#         except:
#             return np.zeros(1)
    
#     def get_state(self, sim_time: float) -> RoverState:
#         """Get current rover state"""
#         wheel_vels = np.array([self.get_sensor_data(name)[0] for name in self.sensor_vel_names])
#         wheel_pos = np.array([self.get_sensor_data(name)[0] for name in self.sensor_pos_names])
        
#         orientation = self.get_sensor_data('orientation')
#         gyro = self.get_sensor_data('angular-velocity')
        
#         chassis_pos = self.data.qpos[0:3].copy()
#         chassis_vel = self.data.qvel[0:3].copy()
        
#         return RoverState(
#             chassis_pos=chassis_pos,
#             chassis_quat=orientation if len(orientation) == 4 else np.array([1, 0, 0, 0]),
#             chassis_vel=chassis_vel,
#             chassis_gyro=gyro if len(gyro) == 3 else np.zeros(3),
#             wheel_velocities=wheel_vels,
#             wheel_positions=wheel_pos,
#             timestamp=sim_time
#         )
    
#     def step(self, sim_time: float, dt: float):
#         """Main control loop step"""
#         state = self.get_state(sim_time)
        
#         # Startup delay - let rover settle
#         if sim_time < 0.5:
#             self.data.ctrl[:] = 0
#             return
        
#         # Stability check
#         is_stable, stability_margin, status_msg = self.stability_monitor.check_stability(state)
        
#         if not is_stable:
#             print(f"🛑 {status_msg}")
#             self.emergency_stop = True
#             self.data.ctrl[:] = 0
#             return
        
#         # Check waypoint reached and capture
#         if self.path_planner.check_waypoint_reached(state.chassis_pos):
#             waypoint = self.path_planner.waypoints[self.path_planner.current_waypoint_idx - 1]
#             if waypoint.capture_image:
#                 capture_data = self.image_capture.capture(state, 
#                                                          self.path_planner.current_waypoint_idx - 1,
#                                                          "waypoint_reached")
#                 self.telemetry['captures'].append(capture_data)
#                 self.last_capture_time = sim_time
        
#         # Periodic capture
#         if sim_time - self.last_capture_time >= self.capture_interval:
#             capture_data = self.image_capture.capture(state, -1, "periodic")
#             self.telemetry['captures'].append(capture_data)
#             self.last_capture_time = sim_time
        
#         # Navigation
#         _, _, yaw = self.stability_monitor.get_roll_pitch_yaw(state.chassis_quat)
#         linear_vel, angular_vel = self.path_planner.get_navigation_command(state.chassis_pos, yaw)
        
#         # Adaptive speed based on stability
#         speed_factor = self.stability_monitor.get_recommended_speed(stability_margin)
#         linear_vel *= speed_factor
#         angular_vel *= speed_factor
        
#         # Compute motor commands
#         desired_wheel_vels = self.velocity_controller.differential_drive(linear_vel, angular_vel)
        
#         motor_commands = np.zeros(6)
#         for i in range(6):
#             error = desired_wheel_vels[i] - state.wheel_velocities[i]
#             motor_commands[i] = self.wheel_pid[i].update(error, dt)
        
#         motor_commands = np.clip(motor_commands, -3.0, 3.0)  # Match XML ctrlrange
        
#         # Apply smooth acceleration limiting
#         motor_delta = motor_commands - self.previous_motor_commands
#         motor_delta = np.clip(motor_delta, -self.max_motor_change, self.max_motor_change)
#         motor_commands = self.previous_motor_commands + motor_delta
#         self.previous_motor_commands = motor_commands.copy()
        
#         for i, motor_id in enumerate(self.motor_ids):
#             self.data.ctrl[motor_id] = motor_commands[i]
        
#         # Log telemetry with orientation data
#         roll, pitch, yaw = self.stability_monitor.get_roll_pitch_yaw(state.chassis_quat)
#         angular_vel_mag = np.linalg.norm(state.chassis_gyro)
        
#         self.telemetry['time'].append(sim_time)
#         self.telemetry['position'].append(state.chassis_pos.copy())
#         self.telemetry['stability'].append(stability_margin)
#         self.telemetry['velocity'].append(np.linalg.norm(state.chassis_vel[:2]))
#         self.telemetry['roll'].append(np.rad2deg(roll))
#         self.telemetry['pitch'].append(np.rad2deg(pitch))
#         self.telemetry['yaw'].append(np.rad2deg(yaw))
#         self.telemetry['angular_velocity'].append(angular_vel_mag)
    
#     def define_exploration_mission(self, area_size: float = 10.0, grid_spacing: float = 3.0):
#         """Define a grid exploration pattern"""
#         print(f"📍 Planning grid exploration: {area_size}m x {area_size}m, spacing: {grid_spacing}m")
        
#         # Create grid waypoints
#         x_points = np.arange(0, area_size, grid_spacing)
#         y_points = np.arange(0, area_size, grid_spacing)
        
#         direction = 1
#         for i, x in enumerate(x_points):
#             if direction == 1:
#                 for y in y_points:
#                     self.path_planner.add_waypoint(x, y, tolerance=0.8, max_speed=0.3)  # Reduced from 0.5
#             else:
#                 for y in reversed(y_points):
#                     self.path_planner.add_waypoint(x, y, tolerance=0.8, max_speed=0.3)  # Reduced from 0.5
#             direction *= -1
        
#         print(f"✓ Planned {len(self.path_planner.waypoints)} waypoints")
    
#     def save_mission_report(self):
#         """Save mission telemetry and report and generate plots"""
#         report_dir = f"mission_data/{self.mission_name}"
#         os.makedirs(report_dir, exist_ok=True)
        
#         # Save telemetry
#         telemetry_file = os.path.join(report_dir, "telemetry.json")
#         telemetry_data = {
#             'time': self.telemetry['time'],
#             'position': [p.tolist() for p in self.telemetry['position']],
#             'stability': self.telemetry['stability'],
#             'velocity': self.telemetry['velocity'],
#             'roll': self.telemetry['roll'],
#             'pitch': self.telemetry['pitch'],
#             'yaw': self.telemetry['yaw'],
#             'angular_velocity': self.telemetry['angular_velocity'],
#             'total_captures': len(self.telemetry['captures'])
#         }
        
#         with open(telemetry_file, 'w') as f:
#             json.dump(telemetry_data, f, indent=2)
        
#         # Generate plots
#         try:
#             self.plot_mission_summary(save_dir=report_dir)
#             self.plot_orientation_analysis(save_dir=report_dir)
#             self.plot_trajectory_3d(save_dir=report_dir)
#             print(f"📊 Mission report and plots saved to {report_dir}")
#         except Exception as e:
#             print(f"⚠️ Plotting failed: {e}")
#             import traceback
#             traceback.print_exc()
    
#     def plot_mission_summary(self, save_dir: str):
#         """Combined figure: trajectory and stability timeline"""
#         times = np.array(self.telemetry['time'])
#         stability = np.array(self.telemetry['stability'])
#         velocity = np.array(self.telemetry['velocity'])
#         positions = np.array(self.telemetry['position']) if len(self.telemetry['position']) > 0 else np.zeros((0, 3))
        
#         if times.size == 0 and positions.size == 0:
#             print("⚠️ No telemetry to generate mission summary.")
#             return
        
#         fig = plt.figure(figsize=(14, 10))
#         gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
#         # Top-left: 2D trajectory
#         ax0 = fig.add_subplot(gs[0, 0])
#         if positions.size:
#             # Color trajectory by stability
#             scatter = ax0.scatter(positions[:, 0], positions[:, 1], c=stability, 
#                                  cmap='RdYlGn', s=20, vmin=0, vmax=1, alpha=0.6)
#             ax0.plot(positions[:, 0], positions[:, 1], 'k-', linewidth=0.5, alpha=0.3)
#             ax0.scatter(positions[0, 0], positions[0, 1], marker='o', s=100, 
#                        c='blue', edgecolors='black', linewidths=2, label='Start', zorder=10)
#             ax0.scatter(positions[-1, 0], positions[-1, 1], marker='*', s=200, 
#                        c='red', edgecolors='black', linewidths=2, label='End', zorder=10)
#             plt.colorbar(scatter, ax=ax0, label='Stability Margin')
        
#         # Waypoints
#         waypoint_list = [wp.position for wp in self.path_planner.waypoints]
#         if len(waypoint_list) > 0:
#             wps = np.array(waypoint_list)
#             ax0.scatter(wps[:, 0], wps[:, 1], marker='x', s=80, linewidths=2, 
#                        c='purple', label='Waypoints', zorder=5)
        
#         # Capture locations
#         capture_positions = []
#         for cap in self.telemetry['captures']:
#             pos = cap.get('position', None)
#             if pos is not None:
#                 capture_positions.append(pos[:2])
#         if capture_positions:
#             cp = np.array(capture_positions)
#             ax0.scatter(cp[:, 0], cp[:, 1], marker='D', s=60, c='orange', 
#                        edgecolors='black', label='Image Captures', zorder=6)
        
#         ax0.set_title('Mission Trajectory (colored by stability)', fontsize=12, fontweight='bold')
#         ax0.set_xlabel('X (m)')
#         ax0.set_ylabel('Y (m)')
#         ax0.set_aspect('equal', adjustable='box')
#         ax0.grid(True, linestyle=':', alpha=0.4)
#         ax0.legend(loc='best')
        
#         # Top-right: Stability and velocity timeline
#         ax1 = fig.add_subplot(gs[0, 1])
#         if times.size:
#             ax1.plot(times, stability, 'g-', linewidth=2, label='Stability Margin')
#             ax1.axhline(y=0.5, color='orange', linestyle='--', linewidth=1, label='Warning Threshold')
#             ax1.axhline(y=0.3, color='red', linestyle='--', linewidth=1, label='Critical Threshold')
#             ax1.set_ylabel('Stability Margin', color='g')
#             ax1.tick_params(axis='y', labelcolor='g')
#             ax1.set_ylim(-0.05, 1.05)
#             ax1.grid(True, linestyle=':', alpha=0.4)
            
#             ax1_twin = ax1.twinx()
#             ax1_twin.plot(times, velocity, 'b--', linewidth=1.5, label='Velocity')
#             ax1_twin.set_ylabel('Velocity (m/s)', color='b')
#             ax1_twin.tick_params(axis='y', labelcolor='b')
            
#             # Mark captures
#             capture_times = [c.get('timestamp', None) for c in self.telemetry['captures'] 
#                            if c.get('timestamp', None) is not None]
#             for ct in capture_times:
#                 ax1.axvline(ct, color='orange', alpha=0.2, linestyle=':')
            
#             lines1, labels1 = ax1.get_legend_handles_labels()
#             lines2, labels2 = ax1_twin.get_legend_handles_labels()
#             ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
#         ax1.set_title('Stability & Velocity vs Time', fontsize=12, fontweight='bold')
#         ax1.set_xlabel('Time (s)')
        
#         # Bottom-left: Height profile
#         ax2 = fig.add_subplot(gs[1, 0])
#         if positions.size:
#             ax2.plot(times, positions[:, 2], 'b-', linewidth=2)
#             ax2.axhline(y=0.25, color='g', linestyle='--', linewidth=1, label='Nominal Height')
#             ax2.set_xlabel('Time (s)')
#             ax2.set_ylabel('Height (m)')
#             ax2.set_title('Chassis Height Over Time', fontsize=12, fontweight='bold')
#             ax2.grid(True, linestyle=':', alpha=0.4)
#             ax2.legend()
        
#         # Bottom-right: Distance traveled
#         ax3 = fig.add_subplot(gs[1, 1])
#         if positions.size:
#             distances = [0]
#             for i in range(1, len(positions)):
#                 dist = np.linalg.norm(positions[i, :2] - positions[i-1, :2])
#                 distances.append(distances[-1] + dist)
            
#             ax3.plot(times, distances, 'purple', linewidth=2)
#             ax3.set_xlabel('Time (s)')
#             ax3.set_ylabel('Distance (m)')
#             ax3.set_title(f'Cumulative Distance: {distances[-1]:.2f}m', fontsize=12, fontweight='bold')
#             ax3.grid(True, linestyle=':', alpha=0.4)
#             ax3.fill_between(times, 0, distances, alpha=0.3, color='purple')
        
#         plt.suptitle(f'Mission Summary: {self.mission_name}', fontsize=14, fontweight='bold', y=0.995)
        
#         fname = os.path.join(save_dir, "mission_summary.png")
#         fig.savefig(fname, bbox_inches='tight', dpi=150)
#         plt.close(fig)
#         print(f"📊 Saved mission summary to {fname}")
    
#     def plot_orientation_analysis(self, save_dir: str):
#         """Plot roll, pitch, yaw and angular velocity"""
#         times = np.array(self.telemetry['time'])
#         roll = np.array(self.telemetry['roll'])
#         pitch = np.array(self.telemetry['pitch'])
#         yaw = np.array(self.telemetry['yaw'])
#         angular_vel = np.array(self.telemetry['angular_velocity'])
        
#         if times.size == 0:
#             print("⚠️ No orientation data to plot.")
#             return
        
#         fig, axes = plt.subplots(2, 2, figsize=(14, 10))
#         fig.suptitle('Rover Orientation Analysis', fontsize=14, fontweight='bold')
        
#         # Roll
#         ax0 = axes[0, 0]
#         ax0.plot(times, roll, 'b-', linewidth=2)
#         ax0.axhline(y=75, color='r', linestyle='--', linewidth=1, label='Critical Limit (±75°)')
#         ax0.axhline(y=-75, color='r', linestyle='--', linewidth=1)
#         ax0.axhline(y=0, color='k', linestyle=':', linewidth=0.5)
#         ax0.fill_between(times, -75, 75, alpha=0.1, color='green')
#         ax0.set_xlabel('Time (s)')
#         ax0.set_ylabel('Roll (degrees)')
#         ax0.set_title('Roll Angle', fontweight='bold')
#         ax0.grid(True, linestyle=':', alpha=0.4)
#         ax0.legend()
        
#         # Pitch
#         ax1 = axes[0, 1]
#         ax1.plot(times, pitch, 'g-', linewidth=2)
#         ax1.axhline(y=75, color='r', linestyle='--', linewidth=1, label='Critical Limit (±75°)')
#         ax1.axhline(y=-75, color='r', linestyle='--', linewidth=1)
#         ax1.axhline(y=0, color='k', linestyle=':', linewidth=0.5)
#         ax1.fill_between(times, -75, 75, alpha=0.1, color='green')
#         ax1.set_xlabel('Time (s)')
#         ax1.set_ylabel('Pitch (degrees)')
#         ax1.set_title('Pitch Angle', fontweight='bold')
#         ax1.grid(True, linestyle=':', alpha=0.4)
#         ax1.legend()
        
#         # Yaw
#         ax2 = axes[1, 0]
#         ax2.plot(times, yaw, 'orange', linewidth=2)
#         ax2.axhline(y=0, color='k', linestyle=':', linewidth=0.5)
#         ax2.set_xlabel('Time (s)')
#         ax2.set_ylabel('Yaw (degrees)')
#         ax2.set_title('Yaw Angle (Heading)', fontweight='bold')
#         ax2.grid(True, linestyle=':', alpha=0.4)
        
#         # Angular velocity magnitude
#         ax3 = axes[1, 1]
#         ax3.plot(times, angular_vel, 'r-', linewidth=2)
#         ax3.axhline(y=8.0, color='r', linestyle='--', linewidth=1, label='Critical Limit (8.0 rad/s)')
#         ax3.fill_between(times, 0, angular_vel, alpha=0.3, color='red')
#         ax3.set_xlabel('Time (s)')
#         ax3.set_ylabel('Angular Velocity (rad/s)')
#         ax3.set_title('Angular Velocity Magnitude', fontweight='bold')
#         ax3.grid(True, linestyle=':', alpha=0.4)
#         ax3.legend()
        
#         plt.tight_layout()
        
#         fname = os.path.join(save_dir, "orientation_analysis.png")
#         fig.savefig(fname, bbox_inches='tight', dpi=150)
#         plt.close(fig)
#         print(f"📊 Saved orientation analysis to {fname}")
    
#     def plot_trajectory_3d(self, save_dir: str):
#         """Plot 3D trajectory with orientation indicators"""
#         positions = np.array(self.telemetry['position']) if len(self.telemetry['position']) > 0 else np.zeros((0, 3))
#         stability = np.array(self.telemetry['stability'])
        
#         if positions.size == 0:
#             print("⚠️ No position data for 3D trajectory.")
#             return
        
#         fig = plt.figure(figsize=(14, 10))
        
#         # 3D trajectory
#         ax1 = fig.add_subplot(121, projection='3d')
        
#         # Color by stability
#         scatter = ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
#                              c=stability, cmap='RdYlGn', s=30, vmin=0, vmax=1, alpha=0.6)
#         ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
#                 'k-', linewidth=1, alpha=0.3)
        
#         # Start and end markers
#         ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
#                    marker='o', s=150, c='blue', edgecolors='black', linewidths=2, label='Start')
#         ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
#                    marker='*', s=250, c='red', edgecolors='black', linewidths=2, label='End')
        
#         # Waypoints (projected to ground)
#         waypoint_list = [wp.position for wp in self.path_planner.waypoints]
#         if len(waypoint_list) > 0:
#             wps = np.array(waypoint_list)
#             ax1.scatter(wps[:, 0], wps[:, 1], np.zeros(len(wps)), 
#                        marker='x', s=100, linewidths=2, c='purple', label='Waypoints', zorder=1)
        
#         ax1.set_xlabel('X (m)')
#         ax1.set_ylabel('Y (m)')
#         ax1.set_zlabel('Z (m)')
#         ax1.set_title('3D Trajectory', fontweight='bold')
#         ax1.legend()
#         plt.colorbar(scatter, ax=ax1, label='Stability Margin', shrink=0.6)
        
#         # Set equal aspect ratio
#         max_range = np.array([positions[:, 0].max()-positions[:, 0].min(),
#                              positions[:, 1].max()-positions[:, 1].min(),
#                              positions[:, 2].max()-positions[:, 2].min()]).max() / 2.0
#         mid_x = (positions[:, 0].max()+positions[:, 0].min()) * 0.5
#         mid_y = (positions[:, 1].max()+positions[:, 1].min()) * 0.5
#         mid_z = (positions[:, 2].max()+positions[:, 2].min()) * 0.5
#         ax1.set_xlim(mid_x - max_range, mid_x + max_range)
#         ax1.set_ylim(mid_y - max_range, mid_y + max_range)
#         ax1.set_zlim(mid_z - max_range, mid_z + max_range)
        
#         # Top-down view with heading arrows
#         ax2 = fig.add_subplot(122)
        
#         # Plot trajectory
#         scatter2 = ax2.scatter(positions[:, 0], positions[:, 1], 
#                               c=stability, cmap='RdYlGn', s=40, vmin=0, vmax=1, alpha=0.6)
#         ax2.plot(positions[:, 0], positions[:, 1], 'k-', linewidth=1, alpha=0.3)
        
#         # Draw heading arrows at intervals
#         yaw = np.array(self.telemetry['yaw'])
#         step = max(1, len(positions) // 20)  # Show ~20 arrows
#         for i in range(0, len(positions), step):
#             yaw_rad = np.deg2rad(yaw[i])
#             dx = 0.3 * np.cos(yaw_rad)
#             dy = 0.3 * np.sin(yaw_rad)
#             ax2.arrow(positions[i, 0], positions[i, 1], dx, dy,
#                      head_width=0.15, head_length=0.1, fc='blue', ec='blue', alpha=0.5)
        
#         # Start and end
#         ax2.scatter(positions[0, 0], positions[0, 1], marker='o', s=150, 
#                    c='blue', edgecolors='black', linewidths=2, label='Start', zorder=10)
#         ax2.scatter(positions[-1, 0], positions[-1, 1], marker='*', s=250, 
#                    c='red', edgecolors='black', linewidths=2, label='End', zorder=10)
        
#         # Waypoints
#         if len(waypoint_list) > 0:
#             wps = np.array(waypoint_list)
#             ax2.scatter(wps[:, 0], wps[:, 1], marker='x', s=100, 
#                        linewidths=2, c='purple', label='Waypoints', zorder=5)
        
#         # Captures
#         capture_positions = []
#         for cap in self.telemetry['captures']:
#             pos = cap.get('position', None)
#             if pos is not None:
#                 capture_positions.append(pos[:2])
#         if capture_positions:
#             cp = np.array(capture_positions)
#             ax2.scatter(cp[:, 0], cp[:, 1], marker='D', s=60, 
#                        c='orange', edgecolors='black', label='Captures', zorder=6)
        
#         ax2.set_xlabel('X (m)')
#         ax2.set_ylabel('Y (m)')
#         ax2.set_title('Top-Down View with Heading Arrows', fontweight='bold')
#         ax2.set_aspect('equal', adjustable='box')
#         ax2.grid(True, linestyle=':', alpha=0.4)
#         ax2.legend(loc='best')
#         plt.colorbar(scatter2, ax=ax2, label='Stability Margin')
        
#         plt.suptitle(f'3D Trajectory Analysis: {self.mission_name}', fontsize=14, fontweight='bold')
#         plt.tight_layout()
        
#         fname = os.path.join(save_dir, "trajectory_3d.png")
#         fig.savefig(fname, bbox_inches='tight', dpi=150)
#         plt.close(fig)
#         print(f"📊 Saved 3D trajectory to {fname}")


# def run_navigation_mission(xml_path: str, mission_name: str = "exploration",
#                           duration: float = 100.0):
#     """Run autonomous navigation mission with MuJoCo viewer."""

#     # Load model
#     model = mujoco.MjModel.from_xml_path(xml_path)
#     data = mujoco.MjData(model)

#     # ---------- CREATE VIEWER ----------
#     # This automatically opens the interactive MJ viewer window
#     viewer = mujoco.viewer.launch_passive(model, data)
#     # -----------------------------------

#     # Create controller
#     controller = EnhancedRoverController(model, data, mission_name=mission_name)

#     # Define mission
#     controller.define_exploration_mission(area_size=12.0, grid_spacing=4.0)

#     # Simulation parameters
#     dt = model.opt.timestep
#     steps = int(duration / dt)

#     print("\n" + "="*70)
#     print("🚀 AUTONOMOUS NAVIGATION MISSION")
#     print("="*70)
#     print(f"Mission: {mission_name}")
#     print(f"Duration: {duration}s")
#     print(f"Waypoints: {len(controller.path_planner.waypoints)}")
#     print(f"Timestep: {dt}s")
#     print("="*70 + "\n")

#     start_time = time.time()

#     # Main simulation loop
#     for step in range(steps):
#         # If the viewer is closed, stop simulation.
#         if viewer.is_running() is False:
#             print("\n🛑 Viewer closed — stopping simulation.")
#             break

#         sim_time = step * dt

#         # Controller step
#         controller.step(sim_time, dt)

#         # MuJoCo step
#         mujoco.mj_step(model, data)

#         # ---------- UPDATE VIEWER ----------
#         viewer.sync()           # redraw frame
#         time.sleep(dt)          # run in realtime
#         # -----------------------------------

#         # Status update
#         if step % 1000 == 0:
#             state = controller.get_state(sim_time)
#             _, stability, _ = controller.stability_monitor.check_stability(state)

#             waypoint = controller.path_planner.get_current_waypoint()
#             if waypoint is not None:
#                 dist_to_wp = np.linalg.norm(state.chassis_pos[:2] - waypoint.position)
#             else:
#                 dist_to_wp = 0.0

#             print(f"t={sim_time:6.1f}s | "
#                   f"Pos: ({state.chassis_pos[0]:5.1f}, {state.chassis_pos[1]:5.1f}) | "
#                   f"WP: {controller.path_planner.current_waypoint_idx + 1}/"
#                   f"{len(controller.path_planner.waypoints)} | "
#                   f"Dist: {dist_to_wp:.1f}m | "
#                   f"Stab: {stability:.2f} | "
#                   f"Imgs: {controller.image_capture.capture_count}")

#         # Check mission completion
#         if controller.path_planner.is_mission_complete():
#             print("\n🎉 Mission complete! All waypoints reached.")
#             break

#         if controller.emergency_stop:
#             print("\n⚠️ Mission aborted due to stability issues")
#             break

#     real_time = time.time() - start_time

#     # Save mission data
#     controller.save_mission_report()

#     print(f"\n✅ Mission completed in {real_time:.1f}s")
#     print(f"📸 Total images captured: {controller.image_capture.capture_count}")
#     print(f"📊 Check mission_data/{mission_name}/ for all data")

#     return controller


# # def run_navigation_mission(xml_path: str, mission_name: str = "exploration", 
# #                           duration: float = 100.0):
# #     """Run autonomous navigation mission"""
    
# #     # Load model
# #     model = mujoco.MjModel.from_xml_path(xml_path)
# #     data = mujoco.MjData(model)
    
# #     # Create controller
# #     controller = EnhancedRoverController(model, data, mission_name=mission_name)
    
# #     # Define mission
# #     controller.define_exploration_mission(area_size=12.0, grid_spacing=4.0)
    
# #     # Simulation parameters
# #     dt = model.opt.timestep
# #     steps = int(duration / dt)
    
# #     print("\n" + "="*70)
# #     print("🚀 AUTONOMOUS NAVIGATION MISSION")
# #     print("="*70)
# #     print(f"Mission: {mission_name}")
# #     print(f"Duration: {duration}s")
# #     print(f"Waypoints: {len(controller.path_planner.waypoints)}")
# #     print(f"Timestep: {dt}s")
# #     print("="*70 + "\n")
    
# #     start_time = time.time()
    
# #     # Main simulation loop
# #     for step in range(steps):
# #         sim_time = step * dt
        
# #         controller.step(sim_time, dt)
# #         mujoco.mj_step(model, data)
        
# #         # Status update
# #         if step % 1000 == 0:
# #             state = controller.get_state(sim_time)
# #             _, stability, _ = controller.stability_monitor.check_stability(state)
            
# #             waypoint = controller.path_planner.get_current_waypoint()
# #             if waypoint is not None:
# #                 dist_to_wp = np.linalg.norm(state.chassis_pos[:2] - waypoint.position)
# #             else:
# #                 dist_to_wp = 0.0
            
# #             print(f"t={sim_time:6.1f}s | "
# #                   f"Pos: ({state.chassis_pos[0]:5.1f}, {state.chassis_pos[1]:5.1f}) | "
# #                   f"WP: {controller.path_planner.current_waypoint_idx + 1}/"
# #                   f"{len(controller.path_planner.waypoints)} | "
# #                   f"Dist: {dist_to_wp:.1f}m | "
# #                   f"Stab: {stability:.2f} | "
# #                   f"Imgs: {controller.image_capture.capture_count}")
        
# #         # Check mission completion
# #         if controller.path_planner.is_mission_complete():
# #             print("\n🎉 Mission complete! All waypoints reached.")
# #             break
        
# #         if controller.emergency_stop:
# #             print("\n⚠️ Mission aborted due to stability issues")
# #             break
    
# #     real_time = time.time() - start_time
    
# #     # Save mission data
# #     controller.save_mission_report()
    
# #     print(f"\n✅ Mission completed in {real_time:.1f}s")
# #     print(f"📸 Total images captured: {controller.image_capture.capture_count}")
# #     print(f"📊 Check mission_data/{mission_name}/ for all data")
    
# #     return controller


# if __name__ == "__main__":
#     import sys
    
#     xml_file = "rockie_bogie.xml"
#     mission_name = "lunar_exploration"
#     use_viewer = True  # Default to using viewer
    
#     if len(sys.argv) > 1:
#         xml_file = sys.argv[1]
#     if len(sys.argv) > 2:
#         mission_name = sys.argv[2]
#     if len(sys.argv) > 3:
#         # Allow disabling viewer with --headless or --no-viewer
#         use_viewer = sys.argv[3].lower() not in ['--headless', '--no-viewer', 'false', '0']
    
#     print("\n🤖 Enhanced Rover Navigation System")
#     print("Features: Stability control, Path planning, Image capture")
#     print(f"Mode: {'VIEWER ENABLED' if use_viewer else 'HEADLESS'}\n")
    
#     try:
#         controller = run_navigation_mission(
#             xml_file,
#             mission_name=mission_name,
#             duration=200.0,
#             # use_viewer=use_viewer
#         )
        
#         print("\n✅ All systems nominal!")
        
#     except FileNotFoundError:
#         print(f"❌ Error: XML file '{xml_file}' not found")
#         sys.exit(1)
#     except Exception as e:
#         print(f"❌ Error: {e}")
#         import traceback
#         traceback.print_exc()
#         sys.exit(1)


"""
Enhanced Rocker-Bogie Rover Navigation Controller
Features:
- Adaptive stability control
- Terrain-aware navigation
- Systematic image capture
- Path planning and waypoint following
- Emergency response system
"""

import numpy as np
import mujoco
import time
import os
from dataclasses import dataclass
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt
import cv2
import imageio
from collections import deque
import json

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


@dataclass
class RoverState:
    """Current state of the rover"""
    chassis_pos: np.ndarray
    chassis_quat: np.ndarray
    chassis_vel: np.ndarray
    chassis_gyro: np.ndarray
    wheel_velocities: np.ndarray
    wheel_positions: np.ndarray
    timestamp: float


@dataclass
class Waypoint:
    """Navigation waypoint"""
    position: np.ndarray  # [x, y]
    tolerance: float = 0.5
    capture_image: bool = True
    max_speed: float = 0.5


class StabilityMonitor:
    """Enhanced stability monitoring with predictive warnings"""
    
    def __init__(self, moon_gravity=True):
        if moon_gravity:
            self.max_pitch = np.deg2rad(75)
            self.max_roll = np.deg2rad(75)
            self.critical_pitch = np.deg2rad(60)
            self.critical_roll = np.deg2rad(60)
        else:
            self.max_pitch = np.deg2rad(45)
            self.max_roll = np.deg2rad(45)
            self.critical_pitch = np.deg2rad(35)
            self.critical_roll = np.deg2rad(35)
        
        self.stability_history = deque(maxlen=50)
        self.min_height = 0.1
        self.safe_height = 0.2
        
    def get_roll_pitch_yaw(self, quat: np.ndarray) -> Tuple[float, float, float]:
        """Convert quaternion to Euler angles"""
        w, x, y, z = quat
        
        # Roll (x-axis)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis)
        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(np.clip(sinp, -1, 1))
        
        # Yaw (z-axis)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw
    
    def check_stability(self, state: RoverState) -> Tuple[bool, float, str]:
        """Comprehensive stability check with graded warnings"""
        roll, pitch, yaw = self.get_roll_pitch_yaw(state.chassis_quat)
        
        # Critical failures
        if abs(pitch) > self.max_pitch:
            return False, 0.0, f"CRITICAL: Pitch {np.rad2deg(pitch):.1f}°"
        
        if abs(roll) > self.max_roll:
            return False, 0.0, f"CRITICAL: Roll {np.rad2deg(roll):.1f}°"
        
        if state.chassis_pos[2] < self.min_height:
            return False, 0.0, "CRITICAL: Chassis collision"
        
        angular_vel_mag = np.linalg.norm(state.chassis_gyro)
        if angular_vel_mag > 8.0:  # Increased from 5.0
            return False, 0.0, f"CRITICAL: Angular velocity {angular_vel_mag:.1f} rad/s"
        
        # Calculate stability margins
        pitch_margin = 1.0 - abs(pitch) / self.max_pitch
        roll_margin = 1.0 - abs(roll) / self.max_roll
        height_margin = min(1.0, (state.chassis_pos[2] - self.min_height) / self.safe_height)
        angular_margin = max(0.0, 1.0 - angular_vel_mag / 8.0)  # Increased from 5.0
        
        stability_margin = min(pitch_margin, roll_margin, height_margin, angular_margin)
        self.stability_history.append(stability_margin)
        
        # Trend analysis
        if len(self.stability_history) > 10:
            recent_trend = np.mean(list(self.stability_history)[-10:])
            if recent_trend < 0.3:
                return True, stability_margin, f"WARNING: Degrading stability (trend: {recent_trend:.2f})"
        
        # Warning levels
        if stability_margin < 0.3:
            return True, stability_margin, f"CAUTION: Low margin {stability_margin:.2f}"
        elif stability_margin < 0.5:
            return True, stability_margin, f"Alert: Moderate stability {stability_margin:.2f}"
        
        return True, stability_margin, "Stable"
    
    def get_recommended_speed(self, stability_margin: float) -> float:
        """Get speed recommendation based on stability"""
        if stability_margin > 0.8:
            return 1.0  # Full speed
        elif stability_margin > 0.5:
            return 0.7  # Moderate speed
        elif stability_margin > 0.3:
            return 0.4  # Reduced speed
        else:
            return 0.1  # Crawl speed


class ImageCaptureSystem:
    """Systematic image capture with metadata logging"""
    
    def __init__(self, model, data, output_dir="mission_data"):
        self.model = model
        self.data = data
        self.output_dir = output_dir
        self.capture_count = 0
        
        # Create directories
        self.rgb_dir = os.path.join(output_dir, "rgb")
        self.depth_dir = os.path.join(output_dir, "depth")
        self.metadata_dir = os.path.join(output_dir, "metadata")
        
        for dir_path in [self.rgb_dir, self.depth_dir, self.metadata_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Setup renderer
        self.width = model.vis.global_.offwidth
        self.height = model.vis.global_.offheight
        self.renderer = mujoco.Renderer(model, height=self.height, width=self.width)
        
        # Find cameras
        self.cameras = {}
        for cam_name in ['camera_left', 'camera_right']:
            try:
                cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
                self.cameras[cam_name] = cam_id
            except:
                pass
        
        if not self.cameras:
            print("⚠️ No stereo cameras found, using default camera")
            self.cameras = {'default': -1}
        
        print(f"📷 Image capture system initialized with cameras: {list(self.cameras.keys())}")
    
    def capture(self, state: RoverState, waypoint_id: int = -1, reason: str = "waypoint") -> dict:
        """Capture images from all cameras with metadata"""
        capture_data = {
            'capture_id': self.capture_count,
            'timestamp': state.timestamp,
            'waypoint_id': waypoint_id,
            'reason': reason,
            'position': state.chassis_pos.tolist(),
            'orientation': state.chassis_quat.tolist(),
            'cameras': {}
        }
        
        for cam_name, cam_id in self.cameras.items():
            # RGB capture
            self.renderer.update_scene(self.data, camera=cam_id)
            rgb = self.renderer.render()
            
            rgb_filename = f"{self.capture_count:06d}_{cam_name}_rgb.png"
            rgb_path = os.path.join(self.rgb_dir, rgb_filename)
            cv2.imwrite(rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            
            # Depth capture
            depth = self._capture_depth(cam_id)
            depth_filename = f"{self.capture_count:06d}_{cam_name}_depth.exr"
            depth_path = os.path.join(self.depth_dir, depth_filename)
            cv2.imwrite(depth_path, depth)
            
            # Camera pose
            cam_pos = self.data.cam_xpos[cam_id] if cam_id >= 0 else state.chassis_pos
            cam_mat = self.data.cam_xmat[cam_id].reshape(3, 3) if cam_id >= 0 else np.eye(3)
            
            capture_data['cameras'][cam_name] = {
                'rgb_file': rgb_filename,
                'depth_file': depth_filename,
                'position': cam_pos.tolist(),
                'orientation': cam_mat.tolist()
            }
        
        # Save metadata
        metadata_file = os.path.join(self.metadata_dir, f"{self.capture_count:06d}_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(capture_data, f, indent=2)
        
        self.capture_count += 1
        print(f"📸 Captured image set {self.capture_count} at position ({state.chassis_pos[0]:.2f}, {state.chassis_pos[1]:.2f})")
        
        return capture_data
    
    def _capture_depth(self, cam_id: int) -> np.ndarray:
        """Capture depth image"""
        scene = mujoco.MjvScene(self.model, maxgeom=1000)
        ctx = mujoco.MjrContext(self.model, mujoco.mjtFramebuffer.mjFB_OFFSCREEN)
        
        cam = mujoco.MjvCamera()
        if cam_id >= 0:
            cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            cam.fixedcamid = cam_id
        else:
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = 0
        
        opt = mujoco.MjvOption()
        viewport = mujoco.MjrRect(0, 0, self.width, self.height)
        
        depth = np.zeros((self.height, self.width), dtype=np.float32)
        rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        mujoco.mjv_updateScene(self.model, self.data, opt, None, cam, 
                              mujoco.mjtCatBit.mjCAT_ALL, scene)
        mujoco.mjr_render(viewport, scene, ctx)
        mujoco.mjr_readPixels(rgb, depth, viewport, ctx)
        
        return depth


class PathPlanner:
    """Waypoint-based path planning"""
    
    def __init__(self):
        self.waypoints: List[Waypoint] = []
        self.current_waypoint_idx = 0
        self.waypoint_reached_callback = None
        
    def add_waypoint(self, x: float, y: float, tolerance: float = 0.5, 
                    capture_image: bool = True, max_speed: float = 0.5):
        """Add waypoint to path"""
        wp = Waypoint(
            position=np.array([x, y]),
            tolerance=tolerance,
            capture_image=capture_image,
            max_speed=max_speed
        )
        self.waypoints.append(wp)
    
    def get_current_waypoint(self) -> Optional[Waypoint]:
        """Get current target waypoint"""
        if self.current_waypoint_idx < len(self.waypoints):
            return self.waypoints[self.current_waypoint_idx]
        return None
    
    def check_waypoint_reached(self, position: np.ndarray) -> bool:
        """Check if current waypoint is reached"""
        waypoint = self.get_current_waypoint()
        if waypoint is None:
            return False
        
        distance = np.linalg.norm(position[:2] - waypoint.position)
        if distance <= waypoint.tolerance:
            print(f"✓ Waypoint {self.current_waypoint_idx + 1}/{len(self.waypoints)} reached")
            self.current_waypoint_idx += 1
            return True
        
        return False
    
    def get_navigation_command(self, current_pos: np.ndarray, current_yaw: float) -> Tuple[float, float]:
        """Calculate linear and angular velocity commands"""
        waypoint = self.get_current_waypoint()
        if waypoint is None:
            return 0.0, 0.0
        
        # Vector to waypoint
        to_waypoint = waypoint.position - current_pos[:2]
        distance = np.linalg.norm(to_waypoint)
        
        if distance < 0.1:
            return 0.0, 0.0
        
        # Desired heading
        desired_yaw = np.arctan2(to_waypoint[1], to_waypoint[0])
        yaw_error = desired_yaw - current_yaw
        
        # Normalize angle to [-pi, pi]
        yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))
        
        # Calculate velocities
        linear_vel = min(waypoint.max_speed, distance * 0.6)  # Increased from 0.3
        angular_vel = np.clip(yaw_error * 1.5, -0.6, 0.6)  # Increased from 1.0 and 0.3
        
        # Reduce linear velocity when turning
        if abs(yaw_error) > np.deg2rad(30):  # Increased threshold from 15
            linear_vel *= 0.5  # Less aggressive reduction
        
        return linear_vel, angular_vel
    
    def is_mission_complete(self) -> bool:
        """Check if all waypoints reached"""
        return self.current_waypoint_idx >= len(self.waypoints)


class PIDController:
    """PID controller for motor control"""
    
    def __init__(self, kp: float, ki: float, kd: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0
        self.integral_limit = 5.0  # Reduced from 10.0
    
    def update(self, error: float, dt: float) -> float:
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error
        
        return self.kp * error + self.ki * self.integral + self.kd * derivative
    
    def reset(self):
        """Reset PID state"""
        self.integral = 0.0
        self.prev_error = 0.0


class VelocityController:
    """Differential drive velocity controller"""
    
    def __init__(self):
        self.wheel_radius = 0.05
        self.track_width = 0.4
        self.max_wheel_velocity = 5.0  # Increased from 2.0 for faster movement
        
    def differential_drive(self, linear_vel: float, angular_vel: float) -> np.ndarray:
        """Compute wheel velocities for differential drive"""
        v_left = linear_vel - (angular_vel * self.track_width / 2.0)
        v_right = linear_vel + (angular_vel * self.track_width / 2.0)
        
        omega_left = v_left / self.wheel_radius
        omega_right = v_right / self.wheel_radius
        
        omega_left = np.clip(omega_left, -self.max_wheel_velocity, self.max_wheel_velocity)
        omega_right = np.clip(omega_right, -self.max_wheel_velocity, self.max_wheel_velocity)
        
        return np.array([
            omega_left, omega_left, omega_left,  # Left wheels
            omega_right, omega_right, omega_right  # Right wheels
        ])


class EnhancedRoverController:
    """Main enhanced rover controller with navigation and imaging"""
    
    def __init__(self, model, data, mission_name="rover_mission"):
        self.model = model
        self.data = data
        self.mission_name = mission_name
        
        # Core systems
        self.stability_monitor = StabilityMonitor(moon_gravity=True)
        self.velocity_controller = VelocityController()
        self.path_planner = PathPlanner()
        self.image_capture = ImageCaptureSystem(model, data, 
                                               output_dir=f"mission_data/{mission_name}")
        
        # Motor control
        self._get_indices()
        self.wheel_pid = [PIDController(kp=20.0, ki=2.0, kd=1.0) for _ in range(6)]  # Increased for faster response
        
        # State
        self.emergency_stop = False
        self.last_capture_time = 0.0
        self.capture_interval = 2.0  # Capture every 2 seconds
        self.previous_motor_commands = np.zeros(6)  # For smooth acceleration
        self.max_motor_change = 1.0  # Increased from 0.5 for faster acceleration
        
        # Telemetry
        self.telemetry = {
            'time': [],
            'position': [],
            'stability': [],
            'velocity': [],
            'captures': [],
            'roll': [],
            'pitch': [],
            'yaw': [],
            'angular_velocity': []
        }
        
        print(f"🤖 Enhanced Rover Controller initialized for mission: {mission_name}")
    
    def _get_indices(self):
        """Get motor and sensor indices"""
        self.motor_names = [
            'left_bogie_front_axle', 'left_bogie_rear_axle', 'left_rocker_rear_axle',
            'right_bogie_front_axle', 'right_bogie_rear_axle', 'right_rocker_rear_axle'
        ]
        
        self.motor_ids = []
        for name in self.motor_names:
            try:
                motor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                self.motor_ids.append(motor_id)
            except:
                print(f"⚠️ Motor {name} not found")
        
        self.sensor_vel_names = [name + '_v' for name in self.motor_names]
        self.sensor_pos_names = [name + '_p' for name in self.motor_names]
    
    def get_sensor_data(self, sensor_name: str) -> np.ndarray:
        """Read sensor data"""
        try:
            sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
            sensor_adr = self.model.sensor_adr[sensor_id]
            sensor_dim = self.model.sensor_dim[sensor_id]
            return self.data.sensordata[sensor_adr:sensor_adr + sensor_dim].copy()
        except:
            return np.zeros(1)
    
    def get_state(self, sim_time: float) -> RoverState:
        """Get current rover state"""
        wheel_vels = np.array([self.get_sensor_data(name)[0] for name in self.sensor_vel_names])
        wheel_pos = np.array([self.get_sensor_data(name)[0] for name in self.sensor_pos_names])
        
        orientation = self.get_sensor_data('orientation')
        gyro = self.get_sensor_data('angular-velocity')
        
        chassis_pos = self.data.qpos[0:3].copy()
        chassis_vel = self.data.qvel[0:3].copy()
        
        return RoverState(
            chassis_pos=chassis_pos,
            chassis_quat=orientation if len(orientation) == 4 else np.array([1, 0, 0, 0]),
            chassis_vel=chassis_vel,
            chassis_gyro=gyro if len(gyro) == 3 else np.zeros(3),
            wheel_velocities=wheel_vels,
            wheel_positions=wheel_pos,
            timestamp=sim_time
        )
    
    def step(self, sim_time: float, dt: float):
        """Main control loop step"""
        state = self.get_state(sim_time)
        
        # Startup delay - let rover settle
        if sim_time < 0.5:
            self.data.ctrl[:] = 0
            return
        
        # Stability check
        is_stable, stability_margin, status_msg = self.stability_monitor.check_stability(state)
        
        if not is_stable:
            print(f"🛑 {status_msg}")
            self.emergency_stop = True
            self.data.ctrl[:] = 0
            return
        
        # Check waypoint reached and capture
        if self.path_planner.check_waypoint_reached(state.chassis_pos):
            waypoint = self.path_planner.waypoints[self.path_planner.current_waypoint_idx - 1]
            if waypoint.capture_image:
                capture_data = self.image_capture.capture(state, 
                                                         self.path_planner.current_waypoint_idx - 1,
                                                         "waypoint_reached")
                self.telemetry['captures'].append(capture_data)
                self.last_capture_time = sim_time
        
        # Periodic capture
        if sim_time - self.last_capture_time >= self.capture_interval:
            capture_data = self.image_capture.capture(state, -1, "periodic")
            self.telemetry['captures'].append(capture_data)
            self.last_capture_time = sim_time
        
        # Navigation
        _, _, yaw = self.stability_monitor.get_roll_pitch_yaw(state.chassis_quat)
        linear_vel, angular_vel = self.path_planner.get_navigation_command(state.chassis_pos, yaw)
        
        # Adaptive speed based on stability
        speed_factor = self.stability_monitor.get_recommended_speed(stability_margin)
        linear_vel *= speed_factor
        angular_vel *= speed_factor
        
        # Compute motor commands
        desired_wheel_vels = self.velocity_controller.differential_drive(linear_vel, angular_vel)
        
        motor_commands = np.zeros(6)
        for i in range(6):
            error = desired_wheel_vels[i] - state.wheel_velocities[i]
            motor_commands[i] = self.wheel_pid[i].update(error, dt)
        
        motor_commands = np.clip(motor_commands, -3.0, 3.0)  # Match XML ctrlrange
        
        # Apply smooth acceleration limiting
        motor_delta = motor_commands - self.previous_motor_commands
        motor_delta = np.clip(motor_delta, -self.max_motor_change, self.max_motor_change)
        motor_commands = self.previous_motor_commands + motor_delta
        self.previous_motor_commands = motor_commands.copy()
        
        for i, motor_id in enumerate(self.motor_ids):
            self.data.ctrl[motor_id] = motor_commands[i]
        
        # Log telemetry with orientation data
        roll, pitch, yaw = self.stability_monitor.get_roll_pitch_yaw(state.chassis_quat)
        angular_vel_mag = np.linalg.norm(state.chassis_gyro)
        
        self.telemetry['time'].append(sim_time)
        self.telemetry['position'].append(state.chassis_pos.copy())
        self.telemetry['stability'].append(stability_margin)
        self.telemetry['velocity'].append(np.linalg.norm(state.chassis_vel[:2]))
        self.telemetry['roll'].append(np.rad2deg(roll))
        self.telemetry['pitch'].append(np.rad2deg(pitch))
        self.telemetry['yaw'].append(np.rad2deg(yaw))
        self.telemetry['angular_velocity'].append(angular_vel_mag)
    
    def define_exploration_mission(self, area_size: float = 10.0, grid_spacing: float = 3.0):
        """Define a grid exploration pattern"""
        print(f"📍 Planning grid exploration: {area_size}m x {area_size}m, spacing: {grid_spacing}m")
        
        # Create grid waypoints with higher speed
        x_points = np.arange(0, area_size, grid_spacing)
        y_points = np.arange(0, area_size, grid_spacing)
        
        direction = 1
        for i, x in enumerate(x_points):
            if direction == 1:
                for y in y_points:
                    self.path_planner.add_waypoint(x, y, tolerance=0.8, max_speed=0.8)  # Increased from 0.3
            else:
                for y in reversed(y_points):
                    self.path_planner.add_waypoint(x, y, tolerance=0.8, max_speed=0.8)  # Increased from 0.3
            direction *= -1
        
        print(f"✓ Planned {len(self.path_planner.waypoints)} waypoints")
    
    def define_complex_mission(self):
        """Define a complex mission with various maneuvers"""
        print(f"📍 Planning complex mission with multiple maneuvers...")
        
        # Mission segments with different characteristics
        mission_segments = [
            # Segment 1: Initial straight run
            ("Straight Sprint", [(i, 0) for i in range(0, 15, 2)], 1.0, 0.6),
            
            # Segment 2: S-curve maneuver
            ("S-Curve Navigation", [
                (15, 0), (16, 2), (17, 4), (18, 6),
                (19, 8), (20, 8), (21, 6), (22, 4), (23, 2), (24, 0)
            ], 0.8, 0.5),
            
            # Segment 3: Circular pattern
            ("Circular Survey", [
                (24 + 3*np.cos(theta), 3*np.sin(theta)) 
                for theta in np.linspace(0, 2*np.pi, 12)
            ], 0.7, 0.5),
            
            # Segment 4: Zigzag pattern
            ("Zigzag Traverse", [
                (27, 0), (29, 3), (31, 0), (33, 3), (35, 0), (37, 3)
            ], 0.9, 0.7),
            
            # Segment 5: Tight spiral
            ("Spiral Maneuver", [
                (40 + r*np.cos(theta), r*np.sin(theta))
                for theta, r in zip(np.linspace(0, 4*np.pi, 16), np.linspace(0.5, 4, 16))
            ], 0.6, 0.5),
            
            # Segment 6: Figure-8 pattern
            ("Figure-8 Pattern", [
                (45 + 3*np.sin(2*theta)*np.cos(theta), 3*np.sin(2*theta)*np.sin(theta))
                for theta in np.linspace(0, 2*np.pi, 16)
            ], 0.7, 0.6),
            
            # Segment 7: Square with diagonal
            ("Square with Diagonal", [
                (50, 0), (55, 0), (55, 5), (50, 5), (50, 0), (55, 5)
            ], 0.8, 0.7),
            
            # Segment 8: Return home with stops
            ("Return Journey", [
                (50, 5), (40, 4), (30, 3), (20, 2), (10, 1), (0, 0)
            ], 0.9, 0.6),
        ]
        
        waypoint_count = 0
        for segment_name, waypoints, max_speed, tolerance in mission_segments:
            print(f"  - {segment_name}: {len(waypoints)} waypoints")
            for x, y in waypoints:
                # Extra captures at key points
                capture = (waypoint_count % 3 == 0)  # Capture every 3rd waypoint
                self.path_planner.add_waypoint(float(x), float(y), 
                                              tolerance=tolerance, 
                                              capture_image=capture,
                                              max_speed=max_speed)
                waypoint_count += 1
        
        print(f"✓ Planned complex mission with {waypoint_count} waypoints across {len(mission_segments)} segments")
    
    def save_mission_report(self):
        """Save mission telemetry and report and generate plots"""
        report_dir = f"mission_data/{self.mission_name}"
        os.makedirs(report_dir, exist_ok=True)
        
        # Save telemetry
        telemetry_file = os.path.join(report_dir, "telemetry.json")
        telemetry_data = {
            'time': self.telemetry['time'],
            'position': [p.tolist() for p in self.telemetry['position']],
            'stability': self.telemetry['stability'],
            'velocity': self.telemetry['velocity'],
            'roll': self.telemetry['roll'],
            'pitch': self.telemetry['pitch'],
            'yaw': self.telemetry['yaw'],
            'angular_velocity': self.telemetry['angular_velocity'],
            'total_captures': len(self.telemetry['captures'])
        }
        
        with open(telemetry_file, 'w') as f:
            json.dump(telemetry_data, f, indent=2)
        
        # Generate plots
        try:
            self.plot_mission_summary(save_dir=report_dir)
            self.plot_orientation_analysis(save_dir=report_dir)
            self.plot_trajectory_3d(save_dir=report_dir)
            print(f"📊 Mission report and plots saved to {report_dir}")
        except Exception as e:
            print(f"⚠️ Plotting failed: {e}")
            import traceback
            traceback.print_exc()
    
    def plot_mission_summary(self, save_dir: str):
        """Combined figure: trajectory and stability timeline"""
        times = np.array(self.telemetry['time'])
        stability = np.array(self.telemetry['stability'])
        velocity = np.array(self.telemetry['velocity'])
        positions = np.array(self.telemetry['position']) if len(self.telemetry['position']) > 0 else np.zeros((0, 3))
        
        if times.size == 0 and positions.size == 0:
            print("⚠️ No telemetry to generate mission summary.")
            return
        
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Top-left: 2D trajectory
        ax0 = fig.add_subplot(gs[0, 0])
        if positions.size:
            # Color trajectory by stability
            scatter = ax0.scatter(positions[:, 0], positions[:, 1], c=stability, 
                                 cmap='RdYlGn', s=20, vmin=0, vmax=1, alpha=0.6)
            ax0.plot(positions[:, 0], positions[:, 1], 'k-', linewidth=0.5, alpha=0.3)
            ax0.scatter(positions[0, 0], positions[0, 1], marker='o', s=100, 
                       c='blue', edgecolors='black', linewidths=2, label='Start', zorder=10)
            ax0.scatter(positions[-1, 0], positions[-1, 1], marker='*', s=200, 
                       c='red', edgecolors='black', linewidths=2, label='End', zorder=10)
            plt.colorbar(scatter, ax=ax0, label='Stability Margin')
        
        # Waypoints
        waypoint_list = [wp.position for wp in self.path_planner.waypoints]
        if len(waypoint_list) > 0:
            wps = np.array(waypoint_list)
            ax0.scatter(wps[:, 0], wps[:, 1], marker='x', s=80, linewidths=2, 
                       c='purple', label='Waypoints', zorder=5)
        
        # Capture locations
        capture_positions = []
        for cap in self.telemetry['captures']:
            pos = cap.get('position', None)
            if pos is not None:
                capture_positions.append(pos[:2])
        if capture_positions:
            cp = np.array(capture_positions)
            ax0.scatter(cp[:, 0], cp[:, 1], marker='D', s=60, c='orange', 
                       edgecolors='black', label='Image Captures', zorder=6)
        
        ax0.set_title('Mission Trajectory (colored by stability)', fontsize=12, fontweight='bold')
        ax0.set_xlabel('X (m)')
        ax0.set_ylabel('Y (m)')
        ax0.set_aspect('equal', adjustable='box')
        ax0.grid(True, linestyle=':', alpha=0.4)
        ax0.legend(loc='best')
        
        # Top-right: Stability and velocity timeline
        ax1 = fig.add_subplot(gs[0, 1])
        if times.size:
            ax1.plot(times, stability, 'g-', linewidth=2, label='Stability Margin')
            ax1.axhline(y=0.5, color='orange', linestyle='--', linewidth=1, label='Warning Threshold')
            ax1.axhline(y=0.3, color='red', linestyle='--', linewidth=1, label='Critical Threshold')
            ax1.set_ylabel('Stability Margin', color='g')
            ax1.tick_params(axis='y', labelcolor='g')
            ax1.set_ylim(-0.05, 1.05)
            ax1.grid(True, linestyle=':', alpha=0.4)
            
            ax1_twin = ax1.twinx()
            ax1_twin.plot(times, velocity, 'b--', linewidth=1.5, label='Velocity')
            ax1_twin.set_ylabel('Velocity (m/s)', color='b')
            ax1_twin.tick_params(axis='y', labelcolor='b')
            
            # Mark captures
            capture_times = [c.get('timestamp', None) for c in self.telemetry['captures'] 
                           if c.get('timestamp', None) is not None]
            for ct in capture_times:
                ax1.axvline(ct, color='orange', alpha=0.2, linestyle=':')
            
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax1_twin.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        ax1.set_title('Stability & Velocity vs Time', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Time (s)')
        
        # Bottom-left: Height profile
        ax2 = fig.add_subplot(gs[1, 0])
        if positions.size:
            ax2.plot(times, positions[:, 2], 'b-', linewidth=2)
            ax2.axhline(y=0.25, color='g', linestyle='--', linewidth=1, label='Nominal Height')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Height (m)')
            ax2.set_title('Chassis Height Over Time', fontsize=12, fontweight='bold')
            ax2.grid(True, linestyle=':', alpha=0.4)
            ax2.legend()
        
        # Bottom-right: Distance traveled
        ax3 = fig.add_subplot(gs[1, 1])
        if positions.size:
            distances = [0]
            for i in range(1, len(positions)):
                dist = np.linalg.norm(positions[i, :2] - positions[i-1, :2])
                distances.append(distances[-1] + dist)
            
            ax3.plot(times, distances, 'purple', linewidth=2)
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Distance (m)')
            ax3.set_title(f'Cumulative Distance: {distances[-1]:.2f}m', fontsize=12, fontweight='bold')
            ax3.grid(True, linestyle=':', alpha=0.4)
            ax3.fill_between(times, 0, distances, alpha=0.3, color='purple')
        
        plt.suptitle(f'Mission Summary: {self.mission_name}', fontsize=14, fontweight='bold', y=0.995)
        
        fname = os.path.join(save_dir, "mission_summary.png")
        fig.savefig(fname, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"📊 Saved mission summary to {fname}")
    
    def plot_orientation_analysis(self, save_dir: str):
        """Plot roll, pitch, yaw and angular velocity"""
        times = np.array(self.telemetry['time'])
        roll = np.array(self.telemetry['roll'])
        pitch = np.array(self.telemetry['pitch'])
        yaw = np.array(self.telemetry['yaw'])
        angular_vel = np.array(self.telemetry['angular_velocity'])
        
        if times.size == 0:
            print("⚠️ No orientation data to plot.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Rover Orientation Analysis', fontsize=14, fontweight='bold')
        
        # Roll
        ax0 = axes[0, 0]
        ax0.plot(times, roll, 'b-', linewidth=2)
        ax0.axhline(y=75, color='r', linestyle='--', linewidth=1, label='Critical Limit (±75°)')
        ax0.axhline(y=-75, color='r', linestyle='--', linewidth=1)
        ax0.axhline(y=0, color='k', linestyle=':', linewidth=0.5)
        ax0.fill_between(times, -75, 75, alpha=0.1, color='green')
        ax0.set_xlabel('Time (s)')
        ax0.set_ylabel('Roll (degrees)')
        ax0.set_title('Roll Angle', fontweight='bold')
        ax0.grid(True, linestyle=':', alpha=0.4)
        ax0.legend()
        
        # Pitch
        ax1 = axes[0, 1]
        ax1.plot(times, pitch, 'g-', linewidth=2)
        ax1.axhline(y=75, color='r', linestyle='--', linewidth=1, label='Critical Limit (±75°)')
        ax1.axhline(y=-75, color='r', linestyle='--', linewidth=1)
        ax1.axhline(y=0, color='k', linestyle=':', linewidth=0.5)
        ax1.fill_between(times, -75, 75, alpha=0.1, color='green')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Pitch (degrees)')
        ax1.set_title('Pitch Angle', fontweight='bold')
        ax1.grid(True, linestyle=':', alpha=0.4)
        ax1.legend()
        
        # Yaw
        ax2 = axes[1, 0]
        ax2.plot(times, yaw, 'orange', linewidth=2)
        ax2.axhline(y=0, color='k', linestyle=':', linewidth=0.5)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Yaw (degrees)')
        ax2.set_title('Yaw Angle (Heading)', fontweight='bold')
        ax2.grid(True, linestyle=':', alpha=0.4)
        
        # Angular velocity magnitude
        ax3 = axes[1, 1]
        ax3.plot(times, angular_vel, 'r-', linewidth=2)
        ax3.axhline(y=8.0, color='r', linestyle='--', linewidth=1, label='Critical Limit (8.0 rad/s)')
        ax3.fill_between(times, 0, angular_vel, alpha=0.3, color='red')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Angular Velocity (rad/s)')
        ax3.set_title('Angular Velocity Magnitude', fontweight='bold')
        ax3.grid(True, linestyle=':', alpha=0.4)
        ax3.legend()
        
        plt.tight_layout()
        
        fname = os.path.join(save_dir, "orientation_analysis.png")
        fig.savefig(fname, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"📊 Saved orientation analysis to {fname}")
    
    def plot_trajectory_3d(self, save_dir: str):
        """Plot 3D trajectory with orientation indicators"""
        positions = np.array(self.telemetry['position']) if len(self.telemetry['position']) > 0 else np.zeros((0, 3))
        stability = np.array(self.telemetry['stability'])
        
        if positions.size == 0:
            print("⚠️ No position data for 3D trajectory.")
            return
        
        fig = plt.figure(figsize=(14, 10))
        
        # 3D trajectory
        ax1 = fig.add_subplot(121, projection='3d')
        
        # Color by stability
        scatter = ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                             c=stability, cmap='RdYlGn', s=30, vmin=0, vmax=1, alpha=0.6)
        ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                'k-', linewidth=1, alpha=0.3)
        
        # Start and end markers
        ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                   marker='o', s=150, c='blue', edgecolors='black', linewidths=2, label='Start')
        ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                   marker='*', s=250, c='red', edgecolors='black', linewidths=2, label='End')
        
        # Waypoints (projected to ground)
        waypoint_list = [wp.position for wp in self.path_planner.waypoints]
        if len(waypoint_list) > 0:
            wps = np.array(waypoint_list)
            ax1.scatter(wps[:, 0], wps[:, 1], np.zeros(len(wps)), 
                       marker='x', s=100, linewidths=2, c='purple', label='Waypoints', zorder=1)
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3D Trajectory', fontweight='bold')
        ax1.legend()
        plt.colorbar(scatter, ax=ax1, label='Stability Margin', shrink=0.6)
        
        # Set equal aspect ratio
        max_range = np.array([positions[:, 0].max()-positions[:, 0].min(),
                             positions[:, 1].max()-positions[:, 1].min(),
                             positions[:, 2].max()-positions[:, 2].min()]).max() / 2.0
        mid_x = (positions[:, 0].max()+positions[:, 0].min()) * 0.5
        mid_y = (positions[:, 1].max()+positions[:, 1].min()) * 0.5
        mid_z = (positions[:, 2].max()+positions[:, 2].min()) * 0.5
        ax1.set_xlim(mid_x - max_range, mid_x + max_range)
        ax1.set_ylim(mid_y - max_range, mid_y + max_range)
        ax1.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Top-down view with heading arrows
        ax2 = fig.add_subplot(122)
        
        # Plot trajectory
        scatter2 = ax2.scatter(positions[:, 0], positions[:, 1], 
                              c=stability, cmap='RdYlGn', s=40, vmin=0, vmax=1, alpha=0.6)
        ax2.plot(positions[:, 0], positions[:, 1], 'k-', linewidth=1, alpha=0.3)
        
        # Draw heading arrows at intervals
        yaw = np.array(self.telemetry['yaw'])
        step = max(1, len(positions) // 20)  # Show ~20 arrows
        for i in range(0, len(positions), step):
            yaw_rad = np.deg2rad(yaw[i])
            dx = 0.3 * np.cos(yaw_rad)
            dy = 0.3 * np.sin(yaw_rad)
            ax2.arrow(positions[i, 0], positions[i, 1], dx, dy,
                     head_width=0.15, head_length=0.1, fc='blue', ec='blue', alpha=0.5)
        
        # Start and end
        ax2.scatter(positions[0, 0], positions[0, 1], marker='o', s=150, 
                   c='blue', edgecolors='black', linewidths=2, label='Start', zorder=10)
        ax2.scatter(positions[-1, 0], positions[-1, 1], marker='*', s=250, 
                   c='red', edgecolors='black', linewidths=2, label='End', zorder=10)
        
        # Waypoints
        if len(waypoint_list) > 0:
            wps = np.array(waypoint_list)
            ax2.scatter(wps[:, 0], wps[:, 1], marker='x', s=100, 
                       linewidths=2, c='purple', label='Waypoints', zorder=5)
        
        # Captures
        capture_positions = []
        for cap in self.telemetry['captures']:
            pos = cap.get('position', None)
            if pos is not None:
                capture_positions.append(pos[:2])
        if capture_positions:
            cp = np.array(capture_positions)
            ax2.scatter(cp[:, 0], cp[:, 1], marker='D', s=60, 
                       c='orange', edgecolors='black', label='Captures', zorder=6)
        
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('Top-Down View with Heading Arrows', fontweight='bold')
        ax2.set_aspect('equal', adjustable='box')
        ax2.grid(True, linestyle=':', alpha=0.4)
        ax2.legend(loc='best')
        plt.colorbar(scatter2, ax=ax2, label='Stability Margin')
        
        plt.suptitle(f'3D Trajectory Analysis: {self.mission_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        fname = os.path.join(save_dir, "trajectory_3d.png")
        fig.savefig(fname, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"📊 Saved 3D trajectory to {fname}")


# def run_navigation_mission(xml_path: str, mission_name: str = "exploration", 
#                           duration: float = 100.0):
#     """Run autonomous navigation mission"""
    
#     # Load model
#     model = mujoco.MjModel.from_xml_path(xml_path)
#     data = mujoco.MjData(model)
    
#     # Create controller
#     controller = EnhancedRoverController(model, data, mission_name=mission_name)
    
#     # Define mission
#     controller.define_exploration_mission(area_size=12.0, grid_spacing=4.0)
    
#     # Simulation parameters
#     dt = model.opt.timestep
#     steps = int(duration / dt)
    
#     print("\n" + "="*70)
#     print("🚀 AUTONOMOUS NAVIGATION MISSION")
#     print("="*70)
#     print(f"Mission: {mission_name}")
#     print(f"Duration: {duration}s")
#     print(f"Waypoints: {len(controller.path_planner.waypoints)}")
#     print(f"Timestep: {dt}s")
#     print("="*70 + "\n")
    
#     start_time = time.time()
    
#     # Main simulation loop
#     for step in range(steps):
#         sim_time = step * dt
        
#         controller.step(sim_time, dt)
#         mujoco.mj_step(model, data)
        
#         # Status update
#         if step % 1000 == 0:
#             state = controller.get_state(sim_time)
#             _, stability, _ = controller.stability_monitor.check_stability(state)
            
#             waypoint = controller.path_planner.get_current_waypoint()
#             if waypoint is not None:
#                 dist_to_wp = np.linalg.norm(state.chassis_pos[:2] - waypoint.position)
#             else:
#                 dist_to_wp = 0.0
            
#             print(f"t={sim_time:6.1f}s | "
#                   f"Pos: ({state.chassis_pos[0]:5.1f}, {state.chassis_pos[1]:5.1f}) | "
#                   f"WP: {controller.path_planner.current_waypoint_idx + 1}/"
#                   f"{len(controller.path_planner.waypoints)} | "
#                   f"Dist: {dist_to_wp:.1f}m | "
#                   f"Stab: {stability:.2f} | "
#                   f"Imgs: {controller.image_capture.capture_count}")
        
#         # Check mission completion
#         if controller.path_planner.is_mission_complete():
#             print("\n🎉 Mission complete! All waypoints reached.")
#             break
        
#         if controller.emergency_stop:
#             print("\n⚠️ Mission aborted due to stability issues")
#             break
    
#     real_time = time.time() - start_time
    
#     # Save mission data
#     controller.save_mission_report()
    
#     print(f"\n✅ Mission completed in {real_time:.1f}s")
#     print(f"📸 Total images captured: {controller.image_capture.capture_count}")
#     print(f"📊 Check mission_data/{mission_name}/ for all data")
    
#     return controller


def run_navigation_mission(xml_path: str, mission_name: str = "exploration",
                          duration: float = 100.0):
    """Run autonomous navigation mission with MuJoCo viewer."""

    # Load model
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # ---------- CREATE VIEWER ----------
    # This automatically opens the interactive MJ viewer window
    viewer = mujoco.viewer.launch_passive(model, data)
    # -----------------------------------

    # Create controller
    controller = EnhancedRoverController(model, data, mission_name=mission_name)

    # Define mission
    controller.define_exploration_mission(area_size=12.0, grid_spacing=4.0)

    # Simulation parameters
    dt = model.opt.timestep
    steps = int(duration / dt)

    print("\n" + "="*70)
    print("🚀 AUTONOMOUS NAVIGATION MISSION")
    print("="*70)
    print(f"Mission: {mission_name}")
    print(f"Duration: {duration}s")
    print(f"Waypoints: {len(controller.path_planner.waypoints)}")
    print(f"Timestep: {dt}s")
    print("="*70 + "\n")

    start_time = time.time()

    # Main simulation loop
    for step in range(steps):
        # If the viewer is closed, stop simulation.
        if viewer.is_running() is False:
            print("\n🛑 Viewer closed — stopping simulation.")
            break

        sim_time = step * dt

        # Controller step
        controller.step(sim_time, dt)

        # MuJoCo step
        mujoco.mj_step(model, data)

        # ---------- UPDATE VIEWER ----------
        viewer.sync()           # redraw frame
        time.sleep(dt)          # run in realtime
        # -----------------------------------

        # Status update
        if step % 1000 == 0:
            state = controller.get_state(sim_time)
            _, stability, _ = controller.stability_monitor.check_stability(state)

            waypoint = controller.path_planner.get_current_waypoint()
            if waypoint is not None:
                dist_to_wp = np.linalg.norm(state.chassis_pos[:2] - waypoint.position)
            else:
                dist_to_wp = 0.0

            print(f"t={sim_time:6.1f}s | "
                  f"Pos: ({state.chassis_pos[0]:5.1f}, {state.chassis_pos[1]:5.1f}) | "
                  f"WP: {controller.path_planner.current_waypoint_idx + 1}/"
                  f"{len(controller.path_planner.waypoints)} | "
                  f"Dist: {dist_to_wp:.1f}m | "
                  f"Stab: {stability:.2f} | "
                  f"Imgs: {controller.image_capture.capture_count}")

        # Check mission completion
        if controller.path_planner.is_mission_complete():
            print("\n🎉 Mission complete! All waypoints reached.")
            break

        if controller.emergency_stop:
            print("\n⚠️ Mission aborted due to stability issues")
            break

    real_time = time.time() - start_time

    # Save mission data
    controller.save_mission_report()

    print(f"\n✅ Mission completed in {real_time:.1f}s")
    print(f"📸 Total images captured: {controller.image_capture.capture_count}")
    print(f"📊 Check mission_data/{mission_name}/ for all data")

    return controller


if __name__ == "__main__":
    import sys
    
    xml_file = "rockie_bogie.xml"
    mission_name = "lunar_exploration"
    # use_viewer = True  # Default to using viewer
    
    if len(sys.argv) > 1:
        xml_file = sys.argv[1]
    if len(sys.argv) > 2:
        mission_name = sys.argv[2]
    if len(sys.argv) > 3:
        # Allow disabling viewer with --headless or --no-viewer
        use_viewer = sys.argv[3].lower() not in ['--headless', '--no-viewer', 'false', '0']
    
    print("\n🤖 Enhanced Rover Navigation System")
    print("Features: Stability control, Path planning, Image capture")
    # print(f"Mode: {'VIEWER ENABLED' if use_viewer else 'HEADLESS'}\n")
    
    try:
        controller = run_navigation_mission(
            xml_file,
            mission_name=mission_name,
            duration=200.0,
            # use_viewer=use_viewer
        )
        
        print("\n✅ All systems nominal!")
        
    except FileNotFoundError:
        print(f"❌ Error: XML file '{xml_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)