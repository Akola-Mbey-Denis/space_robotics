"""
Rocker-Bogie Rover Stability Control System with Video Recording
"""

import numpy as np
import mujoco
import time
import os
from dataclasses import dataclass
from typing import Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import imageio  # ensure video writing works
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


class RoverEKF:
    """
    Extended Kalman Filter for 2D State Estimation (X, Y, Theta).
    Fuses Wheel Odometry (Linear Velocity) with IMU (Angular Velocity).
    """
    def __init__(self, x=0, y=0, theta=0):
        # State Vector [x, y, theta]
        self.state = np.array([x, y, theta])
        
        # Covariance Matrix P (Uncertainty)
        self.P = np.eye(3) * 0.1 
        
        # Process Noise Covariance Q (Tune these!)
        self.Q = np.diag([0.01, 0.01, 0.005]) 
        
    def predict(self, v_linear, w_gyro, dt):
        """
        Prediction Step: Propagate state using motion model.
        v_linear: Linear velocity from wheel encoders (m/s)
        w_gyro: Angular velocity from IMU (rad/s)
        """
        theta = self.state[2]
        
        # --- 1. State Prediction (Motion Model) ---
        
        new_x = self.state[0] + v_linear * np.cos(theta) * dt
        new_y = self.state[1] + v_linear * np.sin(theta) * dt
        new_theta = self.state[2] + w_gyro * dt
        
        # Normalize theta to [-pi, pi]
        new_theta = (new_theta + np.pi) % (2 * np.pi) - np.pi
        
        self.state = np.array([new_x, new_y, new_theta])
        
        # Jacobian of the motion model with respect to state [x, y, theta]
        F = np.eye(3)
        F[0, 2] = -v_linear * np.sin(theta) * dt  # dx/dtheta
        F[1, 2] =  v_linear * np.cos(theta) * dt  # dy/dtheta
        
        # P_new = F * P * F.T + Q
        self.P = F @ self.P @ F.T + self.Q
        
        return self.state.copy()



@dataclass
class RoverState:
    """Current state of the rover"""
    chassis_pos: np.ndarray
    chassis_quat: np.ndarray
    chassis_vel: np.ndarray
    chassis_gyro: np.ndarray
    wheel_velocities: np.ndarray
    wheel_positions: np.ndarray


class StabilityMonitor:
    def __init__(self, moon_gravity=False):
        """Monitors rover stability"""

        if moon_gravity:
            # Allow more tilt on Moon
            self.max_pitch = np.deg2rad(75)  # instead of 45
            self.max_roll  = np.deg2rad(75)
        else:
            self.max_pitch = np.deg2rad(45)
            self.max_roll  = np.deg2rad(45)
        
    def get_roll_pitch_yaw(self, quat: np.ndarray) -> Tuple[float, float, float]:
        """Convert quaternion to roll, pitch, yaw"""
        w, x, y, z = quat
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(np.clip(sinp, -1, 1))
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw
    
    def check_stability(self, state: RoverState) -> Tuple[bool, float, str]:
        """Check if rover is stable"""
        roll, pitch, yaw = self.get_roll_pitch_yaw(state.chassis_quat)
        
        if abs(pitch) > self.max_pitch:
            return False, 0.0, f"Excessive pitch: {np.rad2deg(pitch):.1f}°"
        
        if abs(roll) > self.max_roll:
            return False, 0.0, f"Excessive roll: {np.rad2deg(roll):.1f}°"
        
        if state.chassis_pos[2] < 0.1:
            return False, 0.0, "Chassis too low - possible tip-over"
        
        angular_vel_magnitude = np.linalg.norm(state.chassis_gyro)
        if angular_vel_magnitude > 5.0:
            return False, 0.0, "Excessive angular velocity"
        
        pitch_margin = 1.0 - abs(pitch) / self.max_pitch
        roll_margin = 1.0 - abs(roll) / self.max_roll
        height_margin = min(1.0, (state.chassis_pos[2] - 0.1) / 0.2)
        
        stability_margin = min(pitch_margin, roll_margin, height_margin)
        
        return True, stability_margin, "Stable"


class VelocityController:
    """Controls wheel velocities with differential drive"""
    
    def __init__(self):
        self.wheel_radius = 0.05  # From XML geom
        # self.max_wheel_velocity = 20.0
        self.max_wheel_velocity = 3.3  # previously 20.0

        
    def differential_drive(self, linear_vel: float, angular_vel: float) -> np.ndarray:
        """Compute individual wheel velocities for differential drive"""
        track_width = 0.4  # Distance between left and right wheels
        
        v_left = linear_vel - (angular_vel * track_width / 2.0)
        v_right = linear_vel + (angular_vel * track_width / 2.0)
        
        omega_left = v_left / self.wheel_radius
        omega_right = v_right / self.wheel_radius
        
        omega_left = np.clip(omega_left, -self.max_wheel_velocity, self.max_wheel_velocity)
        omega_right = np.clip(omega_right, -self.max_wheel_velocity, self.max_wheel_velocity)
        
        wheel_velocities = np.array([
            omega_left,   # left_bogie_front
            omega_left,   # left_bogie_rear
            omega_left,   # left_rocker_rear
            omega_right,  # right_bogie_front
            omega_right,  # right_bogie_rear
            omega_right   # right_rocker_rear
        ])
        
        return wheel_velocities
    
    def adaptive_speed(self, stability_margin: float) -> float:
        """Adjust speed based on stability"""
        # Increased base speed to ensure movement
        base_speed = 1.5  # was 1.0
        
        if stability_margin > 0.8:
            speed_factor = 1.3
        elif stability_margin > 0.5:
            speed_factor = 1.0
        else:
            speed_factor = stability_margin
        
        return base_speed * speed_factor
    


class VideoRecorder:
    """Records RGB+depth video AND logs raw frames + depth + camera poses"""

    def __init__(self, model, data, fps=30, save_dir="video_output"):
        self.model = model
        self.data = data
        self.fps = fps
        self.frame_idx = 0  # ⭐ frame counter
        self.recording_enabled = False
        self.renderer = None
        self.video_writers = {}
        
        # Framebuffer resolution
        self.width = model.vis.global_.offwidth
        self.height = model.vis.global_.offheight
        # breakpoint()
        
        print(f"📹 Using framebuffer resolution: {self.width}x{self.height}")

        # Create save dir
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # ⭐ Create raw logging directories
        for sub in ["rgb", "depth", "poses"]:
            os.makedirs(os.path.join(save_dir, sub), exist_ok=True)

        # Init renderer
        self.renderer = mujoco.Renderer(model, height=self.height, width=self.width)

        # Detect cameras
        self.cameras = {}
        target_cameras = ['camera_left', 'camera_right']
        for i in range(model.ncam):
            cam_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            if cam_name in target_cameras:
                self.cameras[cam_name] = i
        
        if not self.cameras:
            print("⚠️ No target cams — using tracking cam")
            self.cameras = {'tracking': -1}


        for cam_name, cam_id in self.cameras.items():
            # breakpoint()
            self._save_intrinsics(cam_name)
            # np.savetxt(os.path.join(save_dir, f"K_{cam_name}.txt"), K)
            print(f"[OK] Saved intrinsics: K_{cam_name}.txt")


        # Init video writers
        self.video_writers = {}
        for cam_name in self.cameras.keys():
            # RGB
            rgb_path = os.path.join(save_dir, f"{cam_name}_rgb.mp4")
            self.video_writers[f"{cam_name}_rgb"] = imageio.get_writer(
                rgb_path, fps=fps, codec="h264_videotoolbox", quality=8
            )
            # Depth
            depth_path = os.path.join(save_dir, f"{cam_name}_depth.mp4")
            self.video_writers[f"{cam_name}_depth"] = imageio.get_writer(
                depth_path, fps=fps, codec="h264_videotoolbox", quality=8
            )

        self.recording_enabled = True

    def _save_intrinsics(self, cam_name):
        """Save 3x3 intrinsic matrix K per camera"""
        cam_id = self.cameras[cam_name]

        # Vertical FOV in radians
        fovy = np.deg2rad(self.model.cam_fovy[cam_id])

        # Focal lengths
        fy = (0.5 * self.height) / np.tan(0.5 * fovy)
        fx = fy * (self.width / self.height)  # fx accounts for aspect ratio

        # Principal point (center)
        cx = self.width / 2.0
        cy = self.height / 2.0

        K = np.array([
            [fx, 0,  cx],
            [0,  fy, cy],
            [0,  0,  1]
        ], dtype=np.float32)

        # Save
        K_dir = os.path.join(self.save_dir, "intrinsics")
        os.makedirs(K_dir, exist_ok=True)

        file = os.path.join(K_dir, f"{cam_name}.txt")
        with open(file, "w") as f:
            f.write(f"# Camera intrinsics for {cam_name}\n")
            np.savetxt(f, K, fmt="%.6f")


  


    def _save_pose(self, cam_name):
        """ Save 4x4 camera pose matrix to poses/{cam}.txt"""
        # Mujoco camera pose
        cam_id = self.cameras[cam_name]
        cam_pos = self.data.cam_xpos[cam_id]      # (3,)
        cam_mat = self.data.cam_xmat[cam_id].reshape(3,3)

        T = np.eye(4)
        T[:3,:3] = cam_mat
        T[:3,3] = cam_pos

        pose_file = os.path.join(self.save_dir, "poses", f"{cam_name}.txt")
        with open(pose_file, "a") as f:
            f.write(f"# frame {self.frame_idx}\n")
            np.savetxt(f, T, fmt="%.6f")
            f.write("\n")

    def capture_frame(self):
        if not self.recording_enabled: return
        
        for cam_name, cam_id in self.cameras.items():

            # ============== RGB ==============
            self.renderer.update_scene(self.data, camera=cam_id)
            rgb = self.renderer.render() # (H,W,3)
            self.video_writers[f"{cam_name}_rgb"].append_data(rgb)

            # ⭐ Save raw RGB PNG
            rgb_dir = os.path.join(self.save_dir, "rgb", cam_name)
            os.makedirs(rgb_dir, exist_ok=True)
            cv2.imwrite(
                os.path.join(rgb_dir, f"{self.frame_idx:06d}.png"),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            )

            # ============== DEPTH ==============

            scene = mujoco.MjvScene(self.model, maxgeom=1000)
            ctx = mujoco.MjrContext(self.model, 1)  # 1=offscreen

            # Camera setup
            cam = mujoco.MjvCamera()
            if cam_id >= 0:
                cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                cam.fixedcamid = cam_id
            else:
                cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                cam.trackbodyid = 0  # default tracking body
            cam.lookat = np.array([0, 0, 0])
            cam.distance = 2.0

            # Visualization options
            opt = mujoco.MjvOption()

            # Viewport
            viewport = mujoco.MjrRect(0, 0, self.width, self.height)

            # Allocate pixel buffers
            rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            depth = np.zeros((self.height, self.width), dtype=np.float32)

            # Update scene
            mujoco.mjv_updateScene(
                self.model, self.data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene
            )

            # Render to context
            mujoco.mjr_render(viewport, scene, ctx)

            # Read pixels into arrays
            mujoco.mjr_readPixels(rgb, depth, viewport, ctx)

            dep_norm = (depth - depth.min()) / ((depth.max() - depth.min())+ 1e-6)
            depth_rgb = (cm.get_cmap("Spectral_r")(dep_norm)[..., :3]*255).astype(np.uint8)
            self.video_writers[f"{cam_name}_depth"].append_data(depth_rgb)

            # ⭐ save raw depth .exr
            depth_dir = os.path.join(self.save_dir, "depth", cam_name)
            os.makedirs(depth_dir, exist_ok=True)

            cv2.imwrite(
                os.path.join(depth_dir, f"{self.frame_idx:06d}.exr"), depth
            )

            # ⭐ save pose
            self._save_pose(cam_name)

        self.frame_idx += 1  # ⭐ next frame

    def close(self):
        print("\n🎬 Finalizing logs & videos...")
        for writer in self.video_writers.values():
            writer.close()
        print(f"✅ Saved logs to {self.save_dir}")



class PIDController:
    """Simple PID controller"""
    def __init__(self, kp: float, ki: float, kd: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0
    
    def update(self, error: float, dt: float) -> float:
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error
        
        self.integral = np.clip(self.integral, -10.0, 10.0)
        
        return self.kp * error + self.ki * self.integral + self.kd * derivative


class RockerBogieController:
    """Main controller for rocker-bogie rover"""
    
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.stability_monitor = StabilityMonitor(moon_gravity= True)
        self.velocity_controller = VelocityController()

        # EKF Addition
        self.ekf = RoverEKF(x=0.0, y=0.0, theta=0.0)
        self.ekf_history = []  # To store the estimated path
        
        self._get_indices()
        
        self.target_linear_vel = 0.0
        self.target_angular_vel = 0.0
        self.emergency_stop = False
        self.stability_history = []
        
        # PID controllers for 6 wheels
        self.wheel_pid = [PIDController(kp=50.0, ki=5.0, kd=2.0) for _ in range(6)]
        
    def _get_indices(self):
        """Get indices for actuators and sensors from XML"""
        # Actuator names from XML
        self.motor_names = [
            'left_bogie_front_axle',
            'left_bogie_rear_axle',
            'left_rocker_rear_axle',
            'right_bogie_front_axle',
            'right_bogie_rear_axle',
            'right_rocker_rear_axle'
        ]
        
        self.motor_ids = []
        for name in self.motor_names:
            try:
                motor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                self.motor_ids.append(motor_id)
            except:
                print(f"Warning: Motor {name} not found")
        
        # Sensor names from XML
        self.sensor_vel_names = [
            'left_bogie_front_axle_v',
            'left_bogie_rear_axle_v',
            'left_rocker_rear_axle_v',
            'right_bogie_front_axle_v',
            'right_bogie_rear_axle_v',
            'right_rocker_rear_axle_v'
        ]
        
        self.sensor_pos_names = [
            'left_bogie_front_axle_p',
            'left_bogie_rear_axle_p',
            'left_rocker_rear_axle_p',
            'right_bogie_front_axle_p',
            'right_bogie_rear_axle_p',
            'right_rocker_rear_axle_p'
        ]
        
        print(f"✓ Found {len(self.motor_ids)} motors")
    
    def get_sensor_data(self, sensor_name: str) -> np.ndarray:
        """Get data from named sensor"""
        try:
            sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
            sensor_adr = self.model.sensor_adr[sensor_id]
            sensor_dim = self.model.sensor_dim[sensor_id]
            return self.data.sensordata[sensor_adr:sensor_adr + sensor_dim].copy()
        except:
            return np.zeros(1)
    
    def get_state(self) -> RoverState:
        """Get current rover state"""
        # Get wheel velocities and positions
        wheel_vels = np.array([self.get_sensor_data(name)[0] for name in self.sensor_vel_names])
        wheel_pos = np.array([self.get_sensor_data(name)[0] for name in self.sensor_pos_names])
        
        # Get orientation and gyro
        orientation = self.get_sensor_data('orientation')
        gyro = self.get_sensor_data('angular-velocity')
        
        # Get position from freejoint (first 3 elements of qpos)
        chassis_pos = self.data.qpos[0:3].copy()
        
        # Get velocity from freejoint
        chassis_vel = self.data.qvel[0:3].copy()
        
        return RoverState(
            chassis_pos=chassis_pos,
            chassis_quat=orientation if len(orientation) == 4 else np.array([1, 0, 0, 0]),
            chassis_vel=chassis_vel,
            chassis_gyro=gyro if len(gyro) == 3 else np.zeros(3),
            wheel_velocities=wheel_vels,
            wheel_positions=wheel_pos
        )
    
    def compute_control(self, dt: float) -> np.ndarray:
        """Main control loop"""
        state = self.get_state()
        
        is_stable, stability_margin, reason = self.stability_monitor.check_stability(state)
        self.stability_history.append(stability_margin)
        
        if not is_stable:
            print(f"⚠️  STABILITY WARNING: {reason}")
            if stability_margin < 0.2:
                print("🛑 EMERGENCY STOP")
                self.emergency_stop = True
                return np.zeros(6)
        
        if self.emergency_stop:
            if np.max(np.abs(state.wheel_velocities)) < 0.1:
                print("✓ Stopped safely")
            return np.zeros(6)
        
        max_speed = self.velocity_controller.adaptive_speed(stability_margin)
        
        if abs(self.target_linear_vel) > max_speed:
            scale = max_speed / abs(self.target_linear_vel)
            linear_vel = self.target_linear_vel * scale
            angular_vel = self.target_angular_vel * scale
        else:
            linear_vel = self.target_linear_vel
            angular_vel = self.target_angular_vel
        
        desired_wheel_vels = self.velocity_controller.differential_drive(linear_vel, angular_vel)
        
        motor_commands = np.zeros(6)
        for i in range(min(6, len(state.wheel_velocities))):
            error = desired_wheel_vels[i] - state.wheel_velocities[i]
            motor_commands[i] = self.wheel_pid[i].update(error, dt)
        
        # Clip to motor limits from XML
        motor_commands = np.clip(motor_commands, -5.12766, 5.12766)
        
        return motor_commands
    
    def set_velocity(self, linear: float, angular: float):
        """Set target velocities"""
        self.target_linear_vel = linear
        self.target_angular_vel = angular
        self.emergency_stop = False
    
    def step(self, dt: float):
        """Execute one control step"""
        motor_commands = self.compute_control(dt)

        #EKF addition
        state = self.get_state()
        
        # 1. Calculate Average Wheel Velocity (Linear Velocity)
        wheel_radius = 0.05 
        avg_wheel_omega = np.mean(state.wheel_velocities)
        v_linear_measured = avg_wheel_omega * wheel_radius
        
        # 2. Get Angular Velocity directly from IMU (Gyro)
        w_gyro_measured = state.chassis_gyro[2] 
        
        # 3. Update Kalman Filter
        estimated_state = self.ekf.predict(v_linear_measured, w_gyro_measured, dt)
        
        # 4. Log for plotting
        self.ekf_history.append(estimated_state)
        
        for i, motor_id in enumerate(self.motor_ids):
            if i < len(motor_commands):
                self.data.ctrl[motor_id] = motor_commands[i]


def plot_results(time_log, stability_log, position_log, ekf_log=None):
    """Plot simulation results"""
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Stability margin
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time_log, stability_log, 'b-', linewidth=2)
    ax1.axhline(y=0.5, color='r', linestyle='--', label='Warning threshold')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Stability Margin')
    ax1.set_title('Stability Over Time')
    ax1.grid(True)
    ax1.legend()
    ax1.set_ylim([0, 1.1])
    
    # XY trajectory
    positions = np.array(position_log)
    ax2 = fig.add_subplot(gs[0, 1])

    ax2.plot(positions[:, 0], positions[:, 1], 'g-', linewidth=2, label='Ground Truth')
    
    # --- CONTRIBUTION: Plot KF Estimate ---
    if ekf_log is not None:
        ekf_pos = np.array(ekf_log)
        # Plot EKF path in Red Dashed
        ax2.plot(ekf_pos[:, 0], ekf_pos[:, 1], 'r--', linewidth=2, label='EKF Estimate')
    
    #ax2.plot(positions[:, 0], positions[:, 1], 'g-', linewidth=2)
    
    ax2.plot(positions[0, 0], positions[0, 1], 'go', markersize=10, label='Start')
    ax2.plot(positions[-1, 0], positions[-1, 1], 'ro', markersize=10, label='End')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('XY Trajectory')
    ax2.grid(True)
    ax2.legend()
    ax2.axis('equal')
    
    # Height over time
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(time_log, positions[:, 2], 'm-', linewidth=2)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Height (m)')
    ax3.set_title('Chassis Height')
    ax3.grid(True)
    
    # Distance traveled
    ax4 = fig.add_subplot(gs[1, 0])
    distances = [0]
    for i in range(1, len(positions)):
        dist = np.linalg.norm(positions[i, :2] - positions[i-1, :2])
        distances.append(distances[-1] + dist)
    
    ax4.plot(time_log, distances, 'c-', linewidth=2)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Distance (m)')
    ax4.set_title('Total Distance Traveled')
    ax4.grid(True)
    
    # Velocity profile
    ax5 = fig.add_subplot(gs[1, 1])
    velocities = []
    for i in range(1, len(positions)):
        dt = time_log[i] - time_log[i-1]
        if dt > 0:
            vel = np.linalg.norm(positions[i, :2] - positions[i-1, :2]) / dt
            velocities.append(vel)
        else:
            velocities.append(0)
    velocities = [0] + velocities
    
    ax5.plot(time_log, velocities, 'orange', linewidth=2)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Velocity (m/s)')
    ax5.set_title('Rover Velocity')
    ax5.grid(True)
    
    # 3D trajectory
    ax6 = fig.add_subplot(gs[1, 2], projection='3d')
    ax6.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2)
    ax6.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='g', s=100, label='Start')
    ax6.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='r', s=100, label='End')
    ax6.set_xlabel('X (m)')
    ax6.set_ylabel('Y (m)')
    ax6.set_zlabel('Z (m)')
    ax6.set_title('3D Trajectory')
    ax6.legend()
    
    plt.savefig('rocker_bogie_analysis.png', dpi=150, bbox_inches='tight')
    print("\n📊 Results saved to: rocker_bogie_analysis.png")
    plt.show()



# def run_simulation(xml_path: str, duration: float = 10.0, record_video: bool = True):
#     """Run rocker-bogie simulation with stability control, live viewer, and video recording"""
    
#     # Load model
#     model = mujoco.MjModel.from_xml_path(xml_path)
#     data = mujoco.MjData(model)
    
#     # Create controller
#     controller = RockerBogieController(model, data)
    
#     # Create video recorder
#     video_recorder = None
#     if record_video:
#         video_recorder = VideoRecorder(model, data, fps=30)
    
#     # Simulation parameters
#     dt = model.opt.timestep
#     steps = int(duration / dt)
    
#     # Data logging
#     time_log = []
#     stability_log = []
#     position_log = []
    
#     print("\n" + "="*70)
#     print("ROCKER-BOGIE STABILITY CONTROL SIMULATION")
#     print("="*70)
#     print(f"Duration: {duration}s")
#     print(f"Timestep: {dt}s")
#     print(f"Total steps: {steps}")
#     print(f"Video recording: {'ON' if video_recorder and video_recorder.recording_enabled else 'OFF'}")
#     print("="*70 + "\n")
    
#     # Short coverage mission (~40s)
#     mission_phases = [
#         (0.0, 5.0, 0.5, 0.0),    # Move forward
#         (5.0, 10.0, 0.5, 0.3),   # Turn right
#         (10.0, 15.0, 0.5, 0.0),  # Straight
#         (15.0, 20.0, 0.5, -0.3), # Turn left
#         (20.0, 25.0, 0.5, 0.0),  # Straight
#         (25.0, 30.0, 0.5, 0.2),  # Gentle right
#         (30.0, 35.0, 0.5, 0.0),  # Straight
#         (35.0, 40.0, 0.5, -0.2), # Gentle left
# ]

#     start_time = time.time()
#     video_frame_interval = max(1, int(1.0 / (30 * dt)))  # Capture at ~30 fps

#     # Launch passive viewer (non-blocking)
#     with mujoco.viewer.launch_passive(model, data) as viewer:
#         for step in range(steps):
#             sim_time = step * dt
            
#             # Update mission phase
#             for t_start, t_end, v_linear, v_angular in mission_phases:
#                 if t_start <= sim_time < t_end:
#                     controller.set_velocity(v_linear, v_angular)
#                     break
            
#             # Control + physics step
#             controller.step(dt)
#             mujoco.mj_step(model, data)
            
#             # Capture video frame
#             if video_recorder and step % video_frame_interval == 0:
#                 video_recorder.capture_frame()
            
#             # Log telemetry
#             if step % 100 == 0:
#                 state = controller.get_state()
#                 _, stability_margin, _ = controller.stability_monitor.check_stability(state)
                
#                 time_log.append(sim_time)
#                 stability_log.append(stability_margin)
#                 position_log.append(state.chassis_pos.copy())
                
#                 roll, pitch, yaw = controller.stability_monitor.get_roll_pitch_yaw(state.chassis_quat)
                
#                 print(f"t={sim_time:5.2f}s | "
#                       f"Pos: ({state.chassis_pos[0]:5.2f}, {state.chassis_pos[1]:5.2f}, {state.chassis_pos[2]:5.2f}) | "
#                       f"Roll: {np.rad2deg(roll):5.1f}° | "
#                       f"Pitch: {np.rad2deg(pitch):5.1f}° | "
#                       f"Stability: {stability_margin:4.2f}")
            
#             # Update viewer (renders the live simulation)
#             viewer.sync()

#     real_time = time.time() - start_time
#     print(f"\n✓ Simulation completed in {real_time:.2f}s (real-time factor: {duration/real_time:.2f}x)")
    
#     # Close video recorder
#     if video_recorder:
#         video_recorder.close()
    
#     # Plot results
#     plot_results(time_log, stability_log, position_log)
    
#     # Print statistics
#     print("\n" + "="*70)
#     print("STABILITY STATISTICS")
#     print("="*70)
#     print(f"Average stability margin: {np.mean(stability_log):.3f}")
#     print(f"Minimum stability margin: {np.min(stability_log):.3f}")
#     print(f"Stability maintained > 0.5: {100 * np.sum(np.array(stability_log) > 0.5) / len(stability_log):.1f}%")
    
#     positions = np.array(position_log)
#     total_distance = 0
#     for i in range(1, len(positions)):
#         total_distance += np.linalg.norm(positions[i, :2] - positions[i-1, :2])
#     print(f"\nTotal distance traveled: {total_distance:.2f} m")
#     print(f"Average velocity: {total_distance/duration:.2f} m/s")
#     print("="*70 + "\n")
    
#     return controller, time_log, stability_log


def run_simulation(xml_path: str, duration: float = 10.0, record_video: bool = True):
    """Run rocker-bogie simulation with stability control and video recording"""
    
    # Load model
    model = mujoco.MjModel.from_xml_path(xml_path)
    
    data = mujoco.MjData(model)
    
    # Create controller
    controller = RockerBogieController(model, data)
    
    # Create video recorder
    video_recorder = None
    if record_video:
        video_recorder = VideoRecorder(model, data, fps=30)
    
    # Simulation parameters
    dt = model.opt.timestep
    steps = int(duration / dt)
    
    # Data logging
    time_log = []
    stability_log = []
    position_log = []
    
    print("\n" + "="*70)
    print("ROCKER-BOGIE STABILITY CONTROL SIMULATION")
    print("="*70)
    print(f"Duration: {duration}s")
    print(f"Timestep: {dt}s")
    print(f"Total steps: {steps}")
    print(f"Video recording: {'ON' if video_recorder and video_recorder.recording_enabled else 'OFF'}")
    print("="*70 + "\n")
    
    # # Mission profile
    # mission_phases = [
    #     (0.0, 2.0, 0.3, 0.0),    # Move forward slowly
    #     (2.0, 4.0, 0.6, 0.0),    # Increase speed
    #     (4.0, 6.0, 0.5, 0.2),    # Turn right
    #     (6.0, 8.0, 0.5, -0.2),   # Turn left
    #     (8.0, 10.0, 0.4, 0.0),   # Straight again
    # ]

    # Mission profile
    mission_phases = [
        (0.0, 10.0, 0.5, 0.0),     # Move forward
        (10.0, 15.0, 0.5, 0.3),    # Turn right
        (15.0, 25.0, 0.5, 0.0),    # Straight
        (25.0, 30.0, 0.5, -0.3),   # Turn left
        (30.0, 40.0, 0.5, 0.0),    # Straight
        (40.0, 45.0, 0.5, 0.3),    # Turn right
        (45.0, 55.0, 0.5, 0.0),    # Straight
        (55.0, 60.0, 0.5, -0.3),   # Turn left
        (60.0, 70.0, 0.5, 0.0),    # Straight
        (70.0, 75.0, 0.5, 0.3),    # Turn right
        (75.0, 85.0, 0.5, 0.0),    # Straight
        (85.0, 90.0, 0.5, -0.3),   # Turn left
        (90.0, 100.0, 0.5, 0.0),   # Final straight
    ]
    
    
    start_time = time.time()
    video_frame_interval = max(1, int(1.0 / (30 * dt)))  # Capture at ~30 fps
    
    for step in range(steps):
        sim_time = step * dt
        
        # Update mission phase
        for t_start, t_end, v_linear, v_angular in mission_phases:
            if t_start <= sim_time < t_end:
                controller.set_velocity(v_linear, v_angular)
                break
        
        # Control + physics step
        controller.step(dt)
        mujoco.mj_step(model, data)
        
        # Capture video frame
        if video_recorder and step % video_frame_interval == 0:
            video_recorder.capture_frame()
        
        # Log telemetry
        if step % 100 == 0:
            state = controller.get_state()
            _, stability_margin, _ = controller.stability_monitor.check_stability(state)
            
            time_log.append(sim_time)
            stability_log.append(stability_margin)
            position_log.append(state.chassis_pos.copy())
            
            roll, pitch, yaw = controller.stability_monitor.get_roll_pitch_yaw(state.chassis_quat)
            
            print(f"t={sim_time:5.2f}s | "
                  f"Pos: ({state.chassis_pos[0]:5.2f}, {state.chassis_pos[1]:5.2f}, {state.chassis_pos[2]:5.2f}) | "
                  f"Roll: {np.rad2deg(roll):5.1f}° | "
                  f"Pitch: {np.rad2deg(pitch):5.1f}° | "
                  f"Stability: {stability_margin:4.2f}")
    
    real_time = time.time() - start_time
    print(f"\n✓ Simulation completed in {real_time:.2f}s (real-time factor: {duration/real_time:.2f}x)")
    
    # Close video recorder
    if video_recorder:
        video_recorder.close()
    
    # Plot results
    plot_results(time_log, stability_log, position_log)
    
    # Print statistics
    print("\n" + "="*70)
    print("STABILITY STATISTICS")
    print("="*70)
    print(f"Average stability margin: {np.mean(stability_log):.3f}")
    print(f"Minimum stability margin: {np.min(stability_log):.3f}")
    print(f"Stability maintained > 0.5: {100 * np.sum(np.array(stability_log) > 0.5) / len(stability_log):.1f}%")
    
    positions = np.array(position_log)
    total_distance = 0
    for i in range(1, len(positions)):
        total_distance += np.linalg.norm(positions[i, :2] - positions[i-1, :2])
    print(f"\nTotal distance traveled: {total_distance:.2f} m")
    print(f"Average velocity: {total_distance/duration:.2f} m/s")
    print("="*70 + "\n")
    
    return controller, time_log, stability_log, position_log, controller.ekf_history


if __name__ == "__main__":
    import sys
    
    xml_file = "rockie_bogie.xml"
    
    if len(sys.argv) > 1:
        xml_file = sys.argv[1]
    
    print("\n🤖 Rocker-Bogie Stability Control System")
    print("With Video Recording\n")
    
    try:
        controller, time_log, stability_log = run_simulation(
            xml_file, 
            duration=100.0,
            record_video=True
        )
        
        print("\n✅ Simulation complete!")
        print("📹 Check the 'video_output' folder for recorded videos")
        print("📊 Check 'rocker_bogie_analysis.png' for telemetry plots")
        
    except FileNotFoundError:
        print(f"❌ Error: XML file '{xml_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)